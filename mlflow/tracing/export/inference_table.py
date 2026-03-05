import logging
from datetime import datetime, timezone
from typing import Any, Sequence

from cachetools import TTLCache
from opentelemetry.sdk.trace import ReadableSpan
from opentelemetry.sdk.trace.export import SpanExporter

from mlflow.entities.model_registry import PromptVersion
from mlflow.entities.trace import Trace
from mlflow.environment_variables import (
    MLFLOW_EXPERIMENT_ID,
    MLFLOW_TRACE_BUFFER_MAX_SIZE,
    MLFLOW_TRACE_BUFFER_TTL_SECONDS,
)
from mlflow.tracing.client import TracingClient
from mlflow.tracing.export.async_export_queue import AsyncTraceExportQueue, Task
from mlflow.tracing.export.utils import try_link_prompts_to_trace
from mlflow.tracing.fluent import _set_last_active_trace_id
from mlflow.tracing.trace_manager import InMemoryTraceManager
from mlflow.tracing.utils import add_size_stats_to_trace_metadata

_logger = logging.getLogger(__name__)


def pop_trace(request_id: str) -> dict[str, Any] | None:
    """
    Pop the completed trace data from the buffer. This method is used in
    the Databricks model serving so please be careful when modifying it.
    """
    wall_clock = datetime.now(timezone.utc).isoformat()
    if request_id not in _TRACE_BUFFER:
        _logger.debug(
            "[TRACE_DEBUG] pop_trace NOT FOUND | wall_clock=%s | request_id=%s | "
            "available_request_ids=%s",
            wall_clock,
            request_id,
            list(_TRACE_BUFFER.keys()),
        )
    else:
        trace_dict = _TRACE_BUFFER.get(request_id)
        trace_info = trace_dict.get("info", {}) if isinstance(trace_dict, dict) else {}
        _logger.debug(
            "[TRACE_DEBUG] pop_trace FOUND | wall_clock=%s | request_id=%s | "
            "trace_request_time=%s | trace_execution_duration=%s | "
            "trace_state=%s | num_spans=%s",
            wall_clock,
            request_id,
            trace_info.get("request_time"),
            trace_info.get("execution_duration"),
            trace_info.get("state"),
            len(trace_dict.get("data", {}).get("spans", [])) if isinstance(trace_dict, dict) else "N/A",
        )
    return _TRACE_BUFFER.pop(request_id, None)


# For Inference Table, we use special TTLCache to store the finished traces
# so that they can be retrieved by Databricks model serving. The values
# in the buffer are not Trace dataclass, but rather a dictionary with the schema
# that is used within Databricks model serving.
def _initialize_trace_buffer():  # Define as a function for testing purposes
    return TTLCache(
        maxsize=MLFLOW_TRACE_BUFFER_MAX_SIZE.get(),
        ttl=MLFLOW_TRACE_BUFFER_TTL_SECONDS.get(),
    )


_TRACE_BUFFER = _initialize_trace_buffer()


class InferenceTableSpanExporter(SpanExporter):
    """
    An exporter implementation that logs the traces to Inference Table.

    Currently the Inference Table does not use collector to receive the traces,
    but rather actively fetches the trace during the prediction process. In the
    future, we may consider using collector-based approach and this exporter should
    send the traces instead of storing them in the buffer.
    """

    def __init__(self):
        self._trace_manager = InMemoryTraceManager.get_instance()
        if MLFLOW_EXPERIMENT_ID.get():
            self._client = TracingClient("databricks")
            self._async_queue = AsyncTraceExportQueue()

    def export(self, spans: Sequence[ReadableSpan]):
        """
        Export the spans to Inference Table via the TTLCache buffer.

        Args:
            spans: A sequence of OpenTelemetry ReadableSpan objects passed from
                a span processor. Only root spans for each trace should be exported.
        """
        for span in spans:
            if span._parent is not None:
                _logger.debug("Received a non-root span. Skipping export.")
                continue

            manager_trace = self._trace_manager.pop_trace(span.context.trace_id)
            if manager_trace is None:
                _logger.debug(f"Trace for span {span} not found. Skipping export.")
                continue

            if manager_trace.is_remote_trace:
                _logger.warning(
                    f"Mlflow does not support exporting the span {span.name} that is created "
                    "in a remote process to Databricks InferenceTable."
                )
                continue

            trace = manager_trace.trace
            _set_last_active_trace_id(trace.info.trace_id)

            wall_clock_export = datetime.now(timezone.utc).isoformat()
            request_time_iso = (
                datetime.fromtimestamp(trace.info.request_time / 1000, tz=timezone.utc).isoformat()
                if trace.info.request_time
                else "None"
            )
            _logger.debug(
                "[TRACE_DEBUG] export ADD TO BUFFER | wall_clock=%s | "
                "client_request_id=%s | trace_id=%s | "
                "request_time_ms=%s (=%s) | execution_duration_ms=%s | "
                "state=%s | num_spans=%d",
                wall_clock_export,
                trace.info.client_request_id,
                trace.info.trace_id,
                trace.info.request_time,
                request_time_iso,
                trace.info.execution_duration,
                trace.info.state,
                len(trace.data.spans),
            )

            # Add the trace to the in-memory buffer so it can be retrieved by upstream
            # The key is Databricks request ID.
            _TRACE_BUFFER[trace.info.client_request_id] = trace.to_dict()
            _logger.debug(f"Added {trace.info.client_request_id} to TRACE_BUFFER")

            # Export to MLflow backend if experiment ID is set
            if MLFLOW_EXPERIMENT_ID.get():
                if trace.info.experiment_id is None:
                    _logger.debug(
                        f"{MLFLOW_EXPERIMENT_ID.name} is set, but trace {trace.info.trace_id} "
                        "has no experiment ID. Skipping export."
                    )
                    continue

                try:
                    # Log the trace to the MLflow backend asynchronously
                    self._async_queue.put(
                        task=Task(
                            handler=self._log_trace_to_mlflow_backend,
                            args=(trace, manager_trace.prompts),
                            error_msg=f"Failed to log trace {trace.info.trace_id}.",
                        )
                    )
                except Exception as e:
                    _logger.warning(
                        f"Failed to export trace to MLflow backend. Error: {e}",
                        stack_info=_logger.isEnabledFor(logging.DEBUG),
                    )

    def _log_trace_to_mlflow_backend(self, trace: Trace, prompts: Sequence[PromptVersion]):
        add_size_stats_to_trace_metadata(trace)

        wall_clock_before = datetime.now(timezone.utc).isoformat()
        request_time_iso = (
            datetime.fromtimestamp(trace.info.request_time / 1000, tz=timezone.utc).isoformat()
            if trace.info.request_time
            else "None"
        )
        _logger.debug(
            "[TRACE_DEBUG] _log_trace_to_mlflow_backend BEFORE start_trace | "
            "wall_clock=%s | trace_id=%s | request_time_ms=%s (=%s) | "
            "execution_duration_ms=%s | state=%s | num_spans=%d",
            wall_clock_before,
            trace.info.trace_id,
            trace.info.request_time,
            request_time_iso,
            trace.info.execution_duration,
            trace.info.state,
            len(trace.data.spans),
        )

        returned_trace_info = self._client.start_trace(trace.info)
        self._client._upload_trace_data(returned_trace_info, trace.data)

        _logger.debug(
            "[TRACE_DEBUG] _log_trace_to_mlflow_backend AFTER start_trace | "
            "wall_clock=%s | returned_trace_id=%s | returned_request_time=%s | "
            "returned_execution_duration=%s",
            datetime.now(timezone.utc).isoformat(),
            returned_trace_info.trace_id,
            returned_trace_info.request_time,
            returned_trace_info.execution_duration,
        )

        # Link prompt versions to the trace. Prompt linking is not critical for trace export
        # (if the prompt fails to link, the user's workflow is minorly affected), so we handle
        # errors gracefully without failing the entire trace export
        try_link_prompts_to_trace(
            client=self._client,
            trace_id=returned_trace_info.trace_id,
            prompts=prompts,
            synchronous=True,  # Run synchronously since we're already in an async task
        )
        _logger.debug(
            f"Finished logging trace to MLflow backend. TraceInfo: {returned_trace_info.to_dict()} "
        )

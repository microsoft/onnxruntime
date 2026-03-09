# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Main OneCollector log exporter implementation."""

import threading
from collections.abc import Sequence
from datetime import datetime, timezone
from time import time
from typing import TYPE_CHECKING, Any, Callable, Optional

import requests
from opentelemetry.sdk._logs import ReadableLogRecord
from opentelemetry.sdk._logs.export import LogExportResult, LogRecordExporter
from opentelemetry.sdk.resources import Resource

from olive.telemetry.library.callback_manager import CallbackManager
from olive.telemetry.library.event_source import event_source
from olive.telemetry.library.options import OneCollectorExporterOptions
from olive.telemetry.library.payload_builder import PayloadBuilder
from olive.telemetry.library.retry import RetryHandler
from olive.telemetry.library.serialization import CommonSchemaJsonSerializationHelper
from olive.telemetry.library.transport import HttpJsonPostTransport

if TYPE_CHECKING:
    from olive.telemetry.library.callback_manager import PayloadTransmittedCallbackArgs


class OneCollectorLogExporter(LogRecordExporter):
    """OpenTelemetry log exporter for Microsoft OneCollector.

    Implements the OpenTelemetry LogRecordExporter interface and sends logs
    to OneCollector using the Common Schema JSON format.
    """

    def __init__(
        self,
        options: Optional[OneCollectorExporterOptions] = None,
        excluded_attributes: Optional[set[str]] = None,
    ):
        """Initialize the OneCollector log exporter.

        Args:
            options: Exporter configuration options
            excluded_attributes: Attribute keys to exclude from log attributes

        """
        # Validate options
        options.validate()

        self._options = options
        self._shutdown_lock = threading.Lock()
        self._shutdown = False
        self._shutdown_event = threading.Event()
        if excluded_attributes is None:
            self._excluded_attributes = {
                "code.filepath",
                "code.function",
                "code.lineno",
                "code.file.path",
                "code.function.name",
                "code.line.number",
            }
        else:
            self._excluded_attributes = set(excluded_attributes)

        # Initialize transport
        transport_opts = options.transport_options

        # Create or get HTTP session
        if transport_opts.http_client_factory:
            self._session = transport_opts.http_client_factory()
        else:
            self._session = requests.Session()

        # Build iKey with tenant prefix
        self._ikey = f"{CommonSchemaJsonSerializationHelper.ONE_COLLECTOR_TENANCY_SYMBOL}:{options.tenant_token}"

        # Initialize callback manager
        self._callback_manager = CallbackManager()

        # Initialize transport with callback manager
        self._transport = HttpJsonPostTransport(
            endpoint=transport_opts.endpoint,
            ikey=options.instrumentation_key,
            compression=transport_opts.compression,
            session=self._session,
            callback_manager=self._callback_manager,
        )

        # Initialize payload builder
        self._payload_builder = PayloadBuilder(
            max_size_bytes=transport_opts.max_payload_size_bytes, max_items=transport_opts.max_items_per_payload
        )

        # Initialize retry handler
        self._retry_handler = RetryHandler(max_retries=6)

        # Initialize metadata
        self._metadata: dict[str, Any] = {}

        # Cache for resource (populated on first export)
        self._resource: Optional[Resource] = None

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        """Add custom metadata fields to all exported logs.

        Args:
            metadata: Dictionary of metadata fields to add

        """
        self._metadata.update(metadata)

    def register_payload_transmitted_callback(
        self, callback: Callable[["PayloadTransmittedCallbackArgs"], None], include_failures: bool = False
    ) -> Callable[[], None]:
        """Register a callback that will be invoked on payload transmission.

        Callbacks are invoked after each HTTP request completes. If retries are
        enabled, callbacks will be invoked for each retry attempt.

        Args:
            callback: Function to call when payload is transmitted.
                      Receives PayloadTransmittedCallbackArgs with transmission details.
            include_failures: If True, callback is invoked on both success and failure.
                             If False, callback is only invoked on success.

        Returns:
            Function to call to unregister the callback.

        Example:
            >>> def on_transmitted(args):
            ...     if args.succeeded:
            ...         print(f"✅ Sent {args.item_count} items ({args.payload_size_bytes} bytes)")
            ...     else:
            ...         print(f"❌ Failed: status={args.status_code}")
            >>>
            >>> unregister = exporter.register_payload_transmitted_callback(
            ...     on_transmitted,
            ...     include_failures=True
            ... )
            >>> # Later: unregister()

        """
        return self._transport.register_payload_transmitted_callback(callback, include_failures)

    def export(self, batch: Sequence[ReadableLogRecord]) -> LogExportResult:
        """Export a batch of log records.

        Args:
            batch: Sequence of log data records to export

        Returns:
            LogExportResult indicating success or failure

        """
        if self._shutdown:
            return LogExportResult.FAILURE

        try:
            # Get resource (cache for subsequent calls)
            if self._resource is None:
                first_item = batch[0] if batch else None
                resource = getattr(first_item, "resource", None)
                if resource is None and first_item is not None:
                    resource = getattr(first_item.log_record, "resource", None)
                self._resource = resource or Resource.create()

            # Serialize log records to JSON
            serialized_items = []
            for log_data in batch:
                try:
                    item_bytes = self._serialize_log_data(log_data)
                    serialized_items.append(item_bytes)
                except Exception as ex:
                    event_source.export_exception_thrown("ReadableLogRecord", ex)
                    # Continue with other items

            if not serialized_items:
                return LogExportResult.FAILURE

            # Build payloads respecting size/count limits
            payloads = self._build_payloads(serialized_items)

            # Send each payload with retry logic
            deadline_sec = time() + self._options.transport_options.timeout_seconds

            for payload in payloads:
                # Count items in this payload (approximation based on newlines)
                item_count = payload.count(b"\n") + 1 if payload else 0
                success = self._retry_handler.execute_with_retry(
                    operation=lambda payload=payload, item_count=item_count: self._transport.send(
                        payload, deadline_sec - time(), item_count=item_count
                    ),
                    deadline_sec=deadline_sec,
                    shutdown_event=self._shutdown_event,
                )

                if not success:
                    return LogExportResult.FAILURE

                # Check if shutdown occurred
                if self._shutdown:
                    return LogExportResult.FAILURE

            # Log success
            event_source.sink_data_written("ReadableLogRecord", len(batch), "OneCollector")

            return LogExportResult.SUCCESS

        except Exception as ex:
            event_source.export_exception_thrown("ReadableLogRecord", ex)
            return LogExportResult.FAILURE

    def _serialize_log_data(self, log_data: ReadableLogRecord) -> bytes:
        """Serialize a single log record to JSON bytes.

        Args:
            log_data: Log data to serialize

        Returns:
            UTF-8 encoded JSON bytes

        """
        log_record = log_data.log_record

        # Build data dictionary
        data = {}

        # Add resource attributes (if available)
        if self._resource and self._resource.attributes:
            for key, value in self._resource.attributes.items():
                # Map common resource attributes
                if key == "service.name" and "app_name" not in data:
                    data["app_name"] = value
                elif key == "service.version" and "app_version" not in data:
                    data["app_version"] = value
                elif key == "service.instance.id" and "app_instance_id" not in data:
                    data["app_instance_id"] = value
                else:
                    data[key] = value

        # Add log record attributes (override resource attributes)
        if log_record.attributes:
            data.update(
                {key: value for key, value in log_record.attributes.items() if key not in self._excluded_attributes}
            )

        # Add custom metadata
        data.update(self._metadata)

        # Format timestamp
        if log_record.timestamp:
            timestamp = datetime.fromtimestamp(log_record.timestamp / 1e9, tz=timezone.utc)
        else:
            timestamp = datetime.now(timezone.utc)

        # Create event envelope
        event_name = str(log_record.body) if log_record.body else "UnnamedEvent"

        envelope = CommonSchemaJsonSerializationHelper.create_event_envelope(
            event_name=event_name, timestamp=timestamp, ikey=self._ikey, data=data
        )

        # Serialize to JSON bytes
        return CommonSchemaJsonSerializationHelper.serialize_to_json_bytes(envelope)

    def _build_payloads(self, serialized_items: list[bytes]) -> list[bytes]:
        """Build payloads from serialized items respecting size and count limits.

        Args:
            serialized_items: List of serialized item bytes

        Returns:
            List of payload bytes

        """
        payloads = []
        self._payload_builder.reset()

        for item_bytes in serialized_items:
            if not self._payload_builder.can_add(item_bytes) and not self._payload_builder.is_empty:
                # Current payload is full, build it and start a new one
                payloads.append(self._payload_builder.build())
                self._payload_builder.reset()

            self._payload_builder.add(item_bytes)

        # Build final payload
        if not self._payload_builder.is_empty:
            payloads.append(self._payload_builder.build())

        return payloads

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        """Force flush any buffered data.

        Note: This exporter doesn't buffer data internally, so this is a no-op.

        Args:
            timeout_millis: Timeout in milliseconds

        Returns:
            True (always succeeds)

        """
        return True

    def shutdown(self) -> None:
        """Shutdown the exporter and release resources."""
        with self._shutdown_lock:
            if self._shutdown:
                return

            self._shutdown = True
            self._shutdown_event.set()

        # Close HTTP session
        if hasattr(self, "_session"):
            self._session.close()

        # Close callback manager
        if hasattr(self, "_callback_manager"):
            self._callback_manager.close()

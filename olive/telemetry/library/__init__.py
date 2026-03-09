# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""OneCollector Exporter for OpenTelemetry Python.

This package provides an OpenTelemetry exporter that sends telemetry data
to Microsoft OneCollector using the Common Schema JSON format.

Example usage:

    from onecollector_exporter import (
        OneCollectorLogExporter,
        OneCollectorExporterOptions,
        get_telemetry_logger,
    )

    # Option 1: Use with OpenTelemetry SDK directly
    options = OneCollectorExporterOptions(
        connection_string="InstrumentationKey=your-key-here"
    )
    exporter = OneCollectorLogExporter(options=options)

    # Add to logger provider
    from opentelemetry.sdk._logs import LoggerProvider
    from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

    provider = LoggerProvider()
    provider.add_log_record_processor(BatchLogRecordProcessor(exporter))

    # Option 2: Use the simplified telemetry logger
    logger = get_telemetry_logger(
        connection_string="InstrumentationKey=your-key-here"
    )
    logger.log("MyEvent", {"key": "value"})
    logger.shutdown()
"""

from olive.telemetry.library.callback_manager import CallbackManager, PayloadTransmittedCallbackArgs
from olive.telemetry.library.connection_string_parser import ConnectionStringParser
from olive.telemetry.library.event_source import OneCollectorEventId, OneCollectorEventSource, event_source
from olive.telemetry.library.exporter import OneCollectorLogExporter
from olive.telemetry.library.options import (
    CompressionType,
    OneCollectorExporterOptions,
    OneCollectorExporterValidationError,
    OneCollectorTransportOptions,
)
from olive.telemetry.library.payload_builder import PayloadBuilder
from olive.telemetry.library.retry import RetryHandler
from olive.telemetry.library.serialization import CommonSchemaJsonSerializationHelper
from olive.telemetry.library.telemetry_logger import (
    TelemetryLogger,
    get_telemetry_logger,
    log_event,
    shutdown_telemetry,
)
from olive.telemetry.library.transport import HttpJsonPostTransport, ITransport

__version__ = "0.0.1"

__all__ = [
    "CallbackManager",
    "CommonSchemaJsonSerializationHelper",
    "CompressionType",
    "ConnectionStringParser",
    "HttpJsonPostTransport",
    "ITransport",
    "OneCollectorEventId",
    "OneCollectorEventSource",
    "OneCollectorExporterOptions",
    "OneCollectorExporterValidationError",
    "OneCollectorLogExporter",
    "OneCollectorTransportOptions",
    "PayloadBuilder",
    "PayloadTransmittedCallbackArgs",
    "RetryHandler",
    "TelemetryLogger",
    "event_source",
    "get_telemetry_logger",
    "log_event",
    "shutdown_telemetry",
]

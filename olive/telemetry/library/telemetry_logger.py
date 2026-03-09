# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""High-level telemetry logger facade for easy usage."""

import logging
import uuid
from typing import Any, Callable, Optional

from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from olive.telemetry.library.exporter import OneCollectorLogExporter
from olive.telemetry.library.options import OneCollectorExporterOptions
from olive.version import __version__ as VERSION


class TelemetryLogger:
    """Singleton telemetry logger for simplified OneCollector integration.

    Provides a simple interface for logging telemetry events without
    needing to configure OpenTelemetry directly.
    """

    _instance: Optional["TelemetryLogger"] = None
    _default_logger: Optional["TelemetryLogger"] = None
    _logger: Optional[logging.Logger] = None
    _logger_exporter: Optional[OneCollectorLogExporter] = None
    _logger_provider: Optional[LoggerProvider] = None

    def __new__(cls, options: Optional[OneCollectorExporterOptions] = None):
        """Create or return the singleton instance.

        Args:
            options: Exporter options (only used on first instantiation)

        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(options)

        return cls._instance

    def _initialize(self, options: Optional[OneCollectorExporterOptions]) -> None:
        """Initialize the logger (called only once).

        Args:
            options: Exporter configuration options

        """
        try:
            # Create exporter
            self._logger_exporter = OneCollectorLogExporter(options=options)

            # Create logger provider
            self._logger_provider = LoggerProvider(
                resource=Resource.create(
                    {
                        "service.name": __name__.split(".", maxsplit=1)[0],
                        "service.version": VERSION,
                        "service.instance.id": str(uuid.uuid4()),  # Unique instance ID; can double as session ID
                    }
                )
            )

            # Set as global logger provider
            set_logger_provider(self._logger_provider)

            # Add batch processor
            self._logger_provider.add_log_record_processor(
                BatchLogRecordProcessor(
                    self._logger_exporter,
                    schedule_delay_millis=1000,
                )
            )

            # Create logging handler
            handler = LoggingHandler(level=logging.INFO, logger_provider=self._logger_provider)

            # Set up Python logger
            logger = logging.getLogger(__name__)
            logger.propagate = False
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            self._logger = logger

        except Exception:
            # Silently fail initialization - logger will be None
            self._logger = None
            self._logger_provider = None
            self._logger_exporter = None

    def add_global_metadata(self, metadata: dict[str, Any]) -> None:
        """Add metadata fields to all telemetry events.

        Args:
            metadata: Dictionary of metadata to add

        """
        if self._logger_exporter:
            self._logger_exporter.add_metadata(metadata)

    def register_payload_transmitted_callback(
        self, callback, include_failures: bool = False
    ) -> Optional[Callable[[], None]]:
        """Register a callback for payload transmission events."""
        if self._logger_exporter:
            return self._logger_exporter.register_payload_transmitted_callback(callback, include_failures)
        return None

    def log(self, event_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Log a telemetry event.

        Args:
            event_name: Name of the event
            attributes: Optional event attributes

        """
        if self._logger:
            extra = attributes if attributes else {}
            self._logger.info(event_name, extra=extra)

    def disable_telemetry(self) -> None:
        """Disable telemetry logging."""
        if self._logger:
            self._logger.disabled = True

    def enable_telemetry(self) -> None:
        """Enable telemetry logging."""
        if self._logger:
            self._logger.disabled = False

    def shutdown(self) -> None:
        """Shutdown the telemetry logger and flush pending data."""
        if self._logger_provider:
            self._logger_provider.shutdown()

    @classmethod
    def get_default_logger(cls, connection_string: Optional[str] = None) -> "TelemetryLogger":
        """Get or create the default telemetry logger.

        Args:
            connection_string: OneCollector connection string (only used on first call)

        Returns:
            TelemetryLogger instance

        """
        if cls._default_logger is None:
            options = None
            if connection_string:
                options = OneCollectorExporterOptions(connection_string=connection_string)
            cls._default_logger = cls(options=options)

        return cls._default_logger

    @classmethod
    def shutdown_default_logger(cls) -> None:
        """Shutdown the default telemetry logger."""
        if cls._default_logger:
            cls._default_logger.shutdown()
            cls._default_logger = None


def get_telemetry_logger(connection_string: Optional[str] = None) -> TelemetryLogger:
    """Get or create the default telemetry logger.

    Args:
        connection_string: OneCollector connection string (only used on first call)

    Returns:
        TelemetryLogger instance

    """
    return TelemetryLogger.get_default_logger(connection_string=connection_string)


def log_event(event_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    """Log a telemetry event using the default logger.

    Args:
        event_name: Name of the event
        attributes: Optional event attributes

    """
    logger = get_telemetry_logger()
    logger.log(event_name, attributes)


def shutdown_telemetry() -> None:
    """Shutdown the default telemetry logger."""
    TelemetryLogger.shutdown_default_logger()

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Configuration options for OneCollector exporter."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import requests

from olive.telemetry.library.connection_string_parser import ConnectionStringParser


class CompressionType(Enum):
    """HTTP compression types supported by OneCollector."""

    NO_COMPRESSION = "none"
    DEFLATE = "deflate"
    GZIP = "gzip"


@dataclass
class OneCollectorTransportOptions:
    """Transport configuration options for OneCollector exporter."""

    DEFAULT_ENDPOINT = "https://mobile.events.data.microsoft.com/OneCollector/1.0/"
    DEFAULT_MAX_PAYLOAD_SIZE_BYTES = 4 * 1024 * 1024  # 4MB
    DEFAULT_MAX_ITEMS_PER_PAYLOAD = 1500

    endpoint: str = DEFAULT_ENDPOINT
    max_payload_size_bytes: int = DEFAULT_MAX_PAYLOAD_SIZE_BYTES
    max_items_per_payload: int = DEFAULT_MAX_ITEMS_PER_PAYLOAD
    compression: CompressionType = CompressionType.DEFLATE
    timeout_seconds: float = 10.0
    http_client_factory: Optional[Callable[[], requests.Session]] = None

    def validate(self) -> None:
        """Validate the transport options.

        Raises:
            OneCollectorExporterValidationError: If any option is invalid

        """
        if not self.endpoint:
            raise OneCollectorExporterValidationError("Endpoint is required")

        if self.max_payload_size_bytes <= 0 and self.max_payload_size_bytes != -1:
            raise OneCollectorExporterValidationError("max_payload_size_bytes must be positive or -1 for unlimited")

        if self.max_items_per_payload <= 0 and self.max_items_per_payload != -1:
            raise OneCollectorExporterValidationError("max_items_per_payload must be positive or -1 for unlimited")

        if self.timeout_seconds <= 0:
            raise OneCollectorExporterValidationError("timeout_seconds must be positive")


@dataclass
class OneCollectorExporterOptions:
    """Configuration options for OneCollector exporter."""

    connection_string: Optional[str] = None
    transport_options: OneCollectorTransportOptions = field(default_factory=OneCollectorTransportOptions)

    # Internal fields populated during validation
    instrumentation_key: Optional[str] = field(default=None, init=False)
    tenant_token: Optional[str] = field(default=None, init=False)

    def validate(self) -> None:
        """Validate the exporter options and populate derived fields.

        Raises:
            OneCollectorExporterValidationError: If any option is invalid

        """
        if not self.connection_string:
            raise OneCollectorExporterValidationError("ConnectionString is required")

        # Parse connection string
        try:
            parser = ConnectionStringParser(self.connection_string)
        except ValueError as ex:
            raise OneCollectorExporterValidationError(str(ex)) from ex

        self.instrumentation_key = parser.instrumentation_key

        if not self.instrumentation_key:
            raise OneCollectorExporterValidationError("Instrumentation key not found in connection string")

        # Extract tenant token (part before first dash)
        dash_pos = self.instrumentation_key.find("-")
        if dash_pos < 0:
            raise OneCollectorExporterValidationError(f"Invalid instrumentation key format: {self.instrumentation_key}")

        self.tenant_token = self.instrumentation_key[:dash_pos]

        # Validate transport options
        self.transport_options.validate()


class OneCollectorExporterValidationError(Exception):
    """Exception raised when OneCollector exporter options validation fails."""

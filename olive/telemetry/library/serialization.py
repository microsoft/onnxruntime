# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""JSON serialization helper for Common Schema format."""

import base64
import json
from datetime import date, datetime, time, timedelta, timezone
from typing import Any
from uuid import UUID


class CommonSchemaJsonSerializationHelper:
    """Helper class for serializing values to Common Schema JSON format.

    Matches the .NET implementation in CommonSchemaJsonSerializationHelper.cs
    """

    # Common Schema constants
    ONE_COLLECTOR_TENANCY_SYMBOL = "o"
    SCHEMA_VERSION = "4.0"

    @staticmethod
    def serialize_value(value: Any) -> Any:
        """Serialize a Python value to JSON-compatible format.

        Args:
            value: The value to serialize

        Returns:
            JSON-serializable representation of the value

        """
        if value is None:
            return None

        # Boolean
        if isinstance(value, bool):
            return value

        # Numeric types
        if isinstance(value, (int, float)):
            return value

        # String
        if isinstance(value, str):
            return value

        # DateTime types
        if isinstance(value, datetime):
            # Convert to UTC ISO 8601 format with 'Z' suffix
            if value.tzinfo is None:
                # Assume naive datetime is UTC
                return value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
            utc_value = value.astimezone(timezone.utc)
            return utc_value.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        if isinstance(value, date):
            return value.isoformat()

        if isinstance(value, time):
            return value.isoformat()

        if isinstance(value, timedelta):
            # Format as ISO 8601 duration
            total_seconds = int(value.total_seconds())
            hours, remainder = divmod(abs(total_seconds), 3600)
            minutes, seconds = divmod(remainder, 60)
            sign = "-" if total_seconds < 0 else ""
            return f"{sign}{hours:02d}:{minutes:02d}:{seconds:02d}"

        # UUID/GUID
        if isinstance(value, UUID):
            return str(value)

        # Bytes - encode as base64
        if isinstance(value, (bytes, bytearray)):
            return base64.b64encode(bytes(value)).decode("ascii")

        # Arrays/Lists
        if isinstance(value, (list, tuple)):
            return [CommonSchemaJsonSerializationHelper.serialize_value(item) for item in value]

        # Dictionary/Map
        if isinstance(value, dict):
            result = {}
            for k, v in value.items():
                if k:  # Skip empty keys
                    result[str(k)] = CommonSchemaJsonSerializationHelper.serialize_value(v)
            return result

        # Default: convert to string
        try:
            return str(value)
        except Exception:
            return f"ERROR: type {type(value).__name__} is not supported"

    @staticmethod
    def create_event_envelope(
        event_name: str, timestamp: datetime, ikey: str, data: dict[str, Any], extensions: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Create a Common Schema event envelope.

        Args:
            event_name: Full event name (namespace.name)
            timestamp: Event timestamp
            ikey: Instrumentation key with tenant prefix
            data: Event data/attributes
            extensions: Optional extension fields

        Returns:
            Common Schema event envelope as dictionary

        """
        envelope = {
            "ver": CommonSchemaJsonSerializationHelper.SCHEMA_VERSION,
            "name": event_name,
            "time": CommonSchemaJsonSerializationHelper.serialize_value(timestamp),
            "iKey": ikey,
            "data": CommonSchemaJsonSerializationHelper.serialize_value(data),
        }

        if extensions:
            envelope["ext"] = CommonSchemaJsonSerializationHelper.serialize_value(extensions)

        return envelope

    @staticmethod
    def serialize_to_json_bytes(envelope: dict[str, Any]) -> bytes:
        """Serialize an envelope to JSON bytes.

        Args:
            envelope: Event envelope dictionary

        Returns:
            UTF-8 encoded JSON bytes

        """
        return json.dumps(envelope, ensure_ascii=False, separators=(",", ":")).encode("utf-8")

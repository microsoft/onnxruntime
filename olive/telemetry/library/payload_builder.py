# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Payload builder for batching telemetry items."""


class PayloadBuilder:
    """Builds payloads respecting size and item count limits.

    Matches the batching logic from the .NET implementation.
    """

    NEWLINE_SEPARATOR = b"\n"

    def __init__(self, max_size_bytes: int, max_items: int):
        """Initialize payload builder.

        Args:
            max_size_bytes: Maximum payload size in bytes (-1 for unlimited)
            max_items: Maximum number of items per payload (-1 for unlimited)

        """
        self.max_size_bytes = max_size_bytes
        self.max_items = max_items
        self.reset()

    def reset(self) -> None:
        """Reset the builder to start a new payload."""
        self.items: list[bytes] = []
        self.current_size = 0

    def can_add(self, item_bytes: bytes) -> bool:
        """Check if an item can be added to the current payload.

        Args:
            item_bytes: Serialized item bytes

        Returns:
            True if item can be added without exceeding limits

        """
        # Check item count limit
        if self.max_items != -1 and len(self.items) >= self.max_items:
            return False

        # Check size limit
        if self.max_size_bytes != -1:
            # Calculate new size including newline separator
            separator_size = len(self.NEWLINE_SEPARATOR) if self.items else 0
            new_size = self.current_size + len(item_bytes) + separator_size

            if new_size > self.max_size_bytes:
                return False

        return True

    def add(self, item_bytes: bytes) -> None:
        """Add an item to the current payload.

        Args:
            item_bytes: Serialized item bytes

        """
        self.items.append(item_bytes)
        self.current_size += len(item_bytes)

        # Account for newline separator (except for first item)
        if len(self.items) > 1:
            self.current_size += len(self.NEWLINE_SEPARATOR)

    def build(self) -> bytes:
        """Build the final payload.

        Returns:
            Newline-delimited payload bytes (x-json-stream format)

        """
        if not self.items:
            return b""

        return self.NEWLINE_SEPARATOR.join(self.items)

    @property
    def item_count(self) -> int:
        """Get the number of items in the current payload."""
        return len(self.items)

    @property
    def is_empty(self) -> bool:
        """Check if the payload is empty."""
        return len(self.items) == 0

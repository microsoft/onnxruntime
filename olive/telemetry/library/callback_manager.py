# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Callback manager for payload transmission events."""

import threading
from dataclasses import dataclass
from typing import Callable, Optional

from olive.telemetry.library.event_source import event_source


@dataclass
class PayloadTransmittedCallbackArgs:
    """Arguments passed to payload transmitted callbacks."""

    succeeded: bool
    """Whether the transmission succeeded."""

    status_code: Optional[int]
    """HTTP status code, if available."""

    payload_size_bytes: int
    """Size of the transmitted payload in bytes."""

    item_count: int
    """Number of items in the payload."""

    payload_bytes: Optional[bytes] = None
    """Raw payload bytes (uncompressed), if available."""


class CallbackManager:
    """Manages callbacks for payload transmission events.

    Allows registration of callbacks that are invoked when payloads
    are successfully transmitted or fail.
    """

    def __init__(self):
        """Initialize the callback manager."""
        self._callbacks: list[tuple[Callable[[PayloadTransmittedCallbackArgs], None], bool]] = []
        self._lock = threading.Lock()
        self._closed = False

    def register(
        self, callback: Callable[[PayloadTransmittedCallbackArgs], None], include_failures: bool = False
    ) -> Callable[[], None]:
        """Register a callback to be invoked on payload transmission.

        Args:
            callback: Function to call when payload is transmitted
            include_failures: Whether to invoke callback on transmission failures

        Returns:
            Function to call to unregister the callback

        """
        with self._lock:
            if self._closed:
                return lambda: None  # No-op unregister if disposed
            entry = (callback, include_failures)
            self._callbacks.append(entry)

        def unregister():
            """Unregister this callback."""
            with self._lock:
                try:
                    self._callbacks.remove(entry)
                except ValueError:
                    # The callback was already removed.
                    pass

        return unregister

    def notify(self, args: PayloadTransmittedCallbackArgs) -> None:
        """Notify all registered callbacks.

        Args:
            args: Callback arguments

        """
        # Get snapshot of callbacks to avoid holding lock during invocation
        with self._lock:
            if self._closed:
                return
            callbacks_snapshot = self._callbacks.copy()

        # Invoke callbacks
        for callback, include_failures in callbacks_snapshot:
            # Check if we should invoke this callback
            if not args.succeeded and not include_failures:
                continue

            try:
                callback(args)
            except Exception as ex:
                # Log but don't propagate exceptions from user code
                event_source.exception_thrown_from_user_code("PayloadTransmittedCallback", ex)

    def close(self) -> None:
        """Close the callback manager and prevent further registrations.

        This method is idempotent and can be called multiple times.
        """
        with self._lock:
            self._callbacks.clear()
            self._closed = True

# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""HTTP transport implementation for OneCollector exporter."""

import gzip
import zlib
from abc import ABC, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Callable, Optional

import requests

from olive.telemetry.library.event_source import event_source
from olive.telemetry.library.options import CompressionType

if TYPE_CHECKING:
    from olive.telemetry.library.callback_manager import CallbackManager, PayloadTransmittedCallbackArgs


class ITransport(ABC):
    """Abstract base class for transports."""

    @abstractmethod
    def send(self, payload: bytes, timeout_sec: float, item_count: int = 1) -> tuple[bool, Optional[int]]:
        """Send a payload.

        Args:
            payload: The data to send
            timeout_sec: Timeout in seconds
            item_count: Number of items in the payload (for callbacks)

        Returns:
            Tuple of (success, status_code)

        """

    @abstractmethod
    def register_payload_transmitted_callback(
        self, callback: Callable[["PayloadTransmittedCallbackArgs"], None], include_failures: bool = False
    ) -> Callable[[], None]:
        """Register a callback for payload transmission events.

        Args:
            callback: Function to call when payload is transmitted
            include_failures: Whether to invoke callback on failures

        Returns:
            Function to call to unregister the callback

        """


class HttpJsonPostTransport(ITransport):
    """HTTP JSON POST transport implementation.

    Sends telemetry data to OneCollector via HTTP POST with JSON payload.
    """

    def __init__(
        self,
        endpoint: str,
        ikey: str,
        compression: CompressionType,
        session: requests.Session,
        callback_manager: Optional["CallbackManager"] = None,
        sdk_version: str = "OTel-python-1.0.0",
    ):
        """Initialize the HTTP transport.

        Args:
            endpoint: OneCollector endpoint URL
            ikey: Instrumentation key
            compression: Compression type to use
            session: Requests session for connection pooling
            callback_manager: Optional callback manager for payload events
            sdk_version: SDK version string

        """
        self.endpoint = endpoint
        self.ikey = ikey
        self.compression = compression
        self.session = session
        self.sdk_version = sdk_version
        self.callback_manager = callback_manager

        # Build base headers
        self.headers = {
            "x-apikey": ikey,
            "User-Agent": "Python/3 HttpClient",
            "Host": "mobile.events.data.microsoft.com",
            "Content-Type": "application/x-json-stream; charset=utf-8",
            "sdk-version": sdk_version,
            "NoResponseBody": "true",
        }

        if compression != CompressionType.NO_COMPRESSION:
            self.headers["Content-Encoding"] = compression.value

    def register_payload_transmitted_callback(
        self, callback: Callable[["PayloadTransmittedCallbackArgs"], None], include_failures: bool = False
    ) -> Callable[[], None]:
        """Register a callback for payload transmission events.

        Args:
            callback: Function to call when payload is transmitted
            include_failures: Whether to invoke callback on failures

        Returns:
            Function to call to unregister the callback

        """
        if self.callback_manager is None:
            # Import here to avoid circular dependency
            from olive.telemetry.library.callback_manager import CallbackManager

            self.callback_manager = CallbackManager()

        return self.callback_manager.register(callback, include_failures)

    def send(self, payload: bytes, timeout_sec: float, item_count: int = 1) -> tuple[bool, Optional[int]]:
        """Send payload via HTTP POST.

        Args:
            payload: Uncompressed payload bytes
            timeout_sec: Request timeout in seconds
            item_count: Number of items in the payload (for callbacks)

        Returns:
            Tuple of (success, status_code)

        """
        payload_size_bytes = len(payload)

        try:
            # Compress payload
            compressed_payload = self._compress(payload)

            # Update headers with content length
            headers = {**self.headers, "Content-Length": str(len(compressed_payload))}

            # Send request
            try:
                response = self.session.post(
                    url=self.endpoint, data=compressed_payload, headers=headers, timeout=timeout_sec
                )
            except requests.exceptions.ConnectionError:
                # Retry once on connection error
                response = self.session.post(
                    url=self.endpoint, data=compressed_payload, headers=headers, timeout=timeout_sec
                )

            # Check response
            success = response.ok
            status_code = response.status_code

            # Invoke callbacks
            if self.callback_manager:
                from olive.telemetry.library.callback_manager import PayloadTransmittedCallbackArgs

                self.callback_manager.notify(
                    PayloadTransmittedCallbackArgs(
                        succeeded=success,
                        status_code=status_code,
                        payload_size_bytes=payload_size_bytes,
                        item_count=item_count,
                        payload_bytes=payload,
                    )
                )

            if success:
                return True, status_code
            else:
                # Log error response
                if event_source.is_error_logging_enabled:
                    collector_error = response.headers.get("Collector-Error", "")
                    error_details = response.text[:100] if response.text else ""
                    event_source.http_transport_error_response(
                        "HttpJsonPost", status_code, collector_error, error_details
                    )
                return False, status_code

        except requests.exceptions.Timeout:
            # Invoke failure callbacks
            if self.callback_manager:
                from olive.telemetry.library.callback_manager import PayloadTransmittedCallbackArgs

                self.callback_manager.notify(
                    PayloadTransmittedCallbackArgs(
                        succeeded=False,
                        status_code=None,
                        payload_size_bytes=payload_size_bytes,
                        item_count=item_count,
                        payload_bytes=payload,
                    )
                )

            event_source.transport_exception_thrown("HttpJsonPost", Exception("Request timeout"))
            return False, None
        except Exception as ex:
            # Invoke failure callbacks
            if self.callback_manager:
                from olive.telemetry.library.callback_manager import PayloadTransmittedCallbackArgs

                self.callback_manager.notify(
                    PayloadTransmittedCallbackArgs(
                        succeeded=False,
                        status_code=None,
                        payload_size_bytes=payload_size_bytes,
                        item_count=item_count,
                        payload_bytes=payload,
                    )
                )

            event_source.transport_exception_thrown("HttpJsonPost", ex)
            return False, None

    def _compress(self, data: bytes) -> bytes:
        """Compress data according to configured compression type.

        Args:
            data: Uncompressed data

        Returns:
            Compressed data

        """
        if self.compression == CompressionType.DEFLATE:
            # Raw deflate (no zlib header)
            compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
            compressed = compressor.compress(data)
            compressed += compressor.flush()
            return compressed

        elif self.compression == CompressionType.GZIP:
            gzip_buffer = BytesIO()
            with gzip.GzipFile(fileobj=gzip_buffer, mode="w") as gzip_file:
                gzip_file.write(data)
            return gzip_buffer.getvalue()

        else:  # NO_COMPRESSION
            return data

    @staticmethod
    def is_retryable(status_code: Optional[int]) -> bool:
        """Check if a response status code indicates the request should be retried.

        Args:
            status_code: HTTP status code, or None if request failed

        Returns:
            True if request should be retried

        """
        if status_code is None:
            return True  # Network errors are retryable

        # Retryable status codes
        return status_code in {408, 429, 500, 502, 503, 504}

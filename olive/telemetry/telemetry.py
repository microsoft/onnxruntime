# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Thin wrapper around the OneCollector telemetry logger with event helpers."""

import base64
import errno
import json
import os
import platform
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

from olive.telemetry.constants import CONNECTION_STRING
from olive.telemetry.deviceid import get_encrypted_device_id_and_status
from olive.telemetry.library.event_source import event_source
from olive.telemetry.library.telemetry_logger import TelemetryLogger, get_telemetry_logger
from olive.telemetry.utils import (
    _decode_cache_line,
    _encode_cache_line,
    _exclusive_file_lock,
    get_telemetry_base_dir,
)

if TYPE_CHECKING:
    from olive.telemetry.library.callback_manager import PayloadTransmittedCallbackArgs

# Default event names used by the high-level telemetry helpers.
HEARTBEAT_EVENT_NAME = "OliveHeartbeat"
ACTION_EVENT_NAME = "OliveAction"
ERROR_EVENT_NAME = "OliveError"

ALLOWED_KEYS = {
    HEARTBEAT_EVENT_NAME: {
        "device_id",
        "id_status",
        "os.name",
        "os.version",
        "os.release",
        "os.arch",
        "app_version",
        "app_instance_id",
        "initTs",
    },
    ACTION_EVENT_NAME: {
        "invoked_from",
        "action_name",
        "duration_ms",
        "success",
        "app_version",
        "app_instance_id",
        "initTs",
    },
    ERROR_EVENT_NAME: {
        "exception_type",
        "exception_message",
        "app_version",
        "app_instance_id",
        "initTs",
    },
}

CRITICAL_EVENTS = {HEARTBEAT_EVENT_NAME}
MAX_CACHE_SIZE_BYTES = 5 * 1024 * 1024
HARD_MAX_CACHE_SIZE_BYTES = 10 * 1024 * 1024
CACHE_FILE_NAME = "olive.json"


class TelemetryCacheHandler:
    """Handles caching of failed telemetry events for offline resilience.

    Design decisions:
    - Single shared cache file (olive.json) for simplicity
    - Cache writes are synchronous (fast JSON operations don't need async)
    - Cache flush runs in a separate thread (slow network I/O)
    - Flush triggered on success when cached events exist
    - All critical sections protected by lock to prevent race conditions
    - Newline-delimited JSON format for human readability and partial corruption recovery

    Assumptions:
    - File I/O (JSON lines) is fast enough for synchronous execution (~microseconds)
    - Network I/O is slow and should not block the callback thread
        - Successful send indicates network is available to retry cached events
    - Cache persists across sessions for offline resilience
    """

    def __init__(self, telemetry: "Telemetry") -> None:
        self._telemetry = telemetry
        # Single shared cache file for all processes
        self._cache_file_name = CACHE_FILE_NAME
        self._shutdown = False
        # Protects all shared state to prevent race conditions
        self._lock = threading.Lock()
        self._callback_condition = threading.Condition()
        self._callbacks_item_count = 0
        self._events_logged = 0
        # Prevents concurrent flush operations
        self._is_flushing = False

    def shutdown(self) -> None:
        """Signal shutdown to prevent new operations.

        Note: Does NOT flush the cache. Cache persists across sessions for
        offline resilience. If network is working, success callbacks already
        flushed. If network is down, flushing would fail anyway.
        """
        with self._lock:
            self._shutdown = True

    def __del__(self):
        """Cleanup cache handler resources on garbage collection.

        Safety net to ensure shutdown is called even if not done explicitly.
        """
        try:
            self.shutdown()
        except Exception:
            # Silently ignore errors during cleanup
            pass

    def on_payload_transmitted(self, args: "PayloadTransmittedCallbackArgs") -> None:
        """Telemetry payload transmission callback.

        Design decisions:
        - Ignore callbacks during flush (unlikely to fail during successful flush)
        - On success: flush cache if any cached events exist
        - On failure: write to cache immediately (synchronous for simplicity)

        Assumptions:
        - Successful transmission indicates network is available to retry cached events
        - If flush is in progress, we already successfully sent an event, so unlikely an event would suddenly fail
        - Multiple concurrent successes don't need multiple flush operations
        - Failed payloads should be cached immediately to avoid loss
        """
        try:
            payload = None
            should_flush = False

            with self._lock:
                if self._shutdown:
                    return

                # Skip callbacks from replayed events during flush
                # If a flush is in progress it means we successfully sent an event,
                # so it's unlikely that an event would suddenly fail and need to be cached
                # and we don't need to flush again.
                if self._is_flushing:
                    with self._callback_condition:
                        self._callbacks_item_count += args.item_count
                        self._callback_condition.notify_all()
                    return

                if args.succeeded:
                    # Only flush if cache exists and no flush is in progress
                    cache_path = self.cache_path
                    if cache_path and cache_path.exists():
                        should_flush = True
                else:
                    payload = args.payload_bytes

            if should_flush:
                # Release lock before scheduling (flush runs in separate thread)
                self._schedule_flush()
            elif payload:
                # Write synchronously - JSON operations are fast enough
                self._write_payload_to_cache(payload)
        except Exception:
            # Fail silently - telemetry should never crash the application
            pass
        finally:
            with self._callback_condition:
                self._callbacks_item_count += args.item_count
                self._callback_condition.notify_all()

    def wait_for_callbacks(self, timeout_sec: float) -> bool:
        deadline = time.time() + timeout_sec
        while True:
            with self._callback_condition:
                callbacks_item_count = self._callbacks_item_count
                expected_items = self._events_logged
                if not self.is_flushing and callbacks_item_count >= expected_items:
                    return True
            remaining = deadline - time.time()
            if remaining <= 0:
                return False
            with self._callback_condition:
                self._callback_condition.wait(timeout=remaining)

    def record_event_logged(self, count: int = 1) -> None:
        with self._callback_condition:
            self._events_logged += count

    def _schedule_flush(self) -> None:
        """Schedule cache flush in a separate thread to avoid blocking the callback.

        Design decisions:
        - Check _is_flushing before spawning thread to avoid unnecessary threads
        - Run flush in daemon thread (don't block process exit)
        - Acquire lock at start to set _is_flushing flag atomically
        - Always clear _is_flushing flag even if flush fails

        Assumptions:
        - Flush operations are slow (network I/O) and should not block callbacks
        - Daemon thread is acceptable (flush is best-effort)
        """
        # Check before spawning thread to avoid unnecessary thread creation
        with self._lock:
            if self._shutdown or self._is_flushing:
                return
            self._is_flushing = True

        def flush_task():
            try:
                self._flush_cache()
            except Exception:
                # Fail silently
                pass
            finally:
                # Always clear flag, even on exception
                with self._lock:
                    self._is_flushing = False

        thread = threading.Thread(target=flush_task, daemon=True)
        thread.start()

    @property
    def cache_path(self) -> Optional[Path]:
        """Get the path to the telemetry cache file.

        Returns:
            Optional[Path]: Path to cache file, or None if base directory unavailable.

        """
        telemetry_cache_dir = None
        if "OLIVE_TELEMETRY_CACHE_DIR" in os.environ:
            telemetry_cache_dir = os.environ["OLIVE_TELEMETRY_CACHE_DIR"]
        if not telemetry_cache_dir:
            telemetry_cache_dir = get_telemetry_base_dir() / "cache"
        return telemetry_cache_dir / self._cache_file_name

    def _write_payload_to_cache(self, payload: bytes) -> None:
        """Write failed telemetry payload to cache for later retry.

        Design decisions:
        - Parse payload to extract individual events (allows filtering)
        - Filter to only critical events near size limit (preserves important data)
        - Use file locking for multi-process safety (prevents corruption)
        - Use exponential backoff for file contention (avoids spinning)
        - Fail silently on errors (telemetry should never crash app)

        Assumptions:
        - JSON operations are fast enough for synchronous execution
        - File contention is rare and transient (retry a few times)
        - Cache size limits prevent unbounded growth
        - Critical events (heartbeat) are more important than others
        """
        try:
            cache_path = self.cache_path
            if cache_path is None:
                return

            # Parse payload into individual events for filtering
            entries = _parse_payload(payload)
            if not entries:
                return

            cache_path.parent.mkdir(parents=True, exist_ok=True)

            max_retries = 3
            for attempt in range(max_retries + 1):
                try:
                    cache_size = cache_path.stat().st_size if cache_path.exists() else 0

                    # Hard limit: stop caching entirely to prevent unbounded growth
                    if cache_size >= HARD_MAX_CACHE_SIZE_BYTES:
                        return

                    # Soft limit: keep only critical events to preserve space
                    if cache_size >= MAX_CACHE_SIZE_BYTES:
                        entries = [entry for entry in entries if entry["event_name"] in CRITICAL_EVENTS]
                        if not entries:
                            return

                    # Append base64-encoded newline-delimited entries
                    # Use exclusive file lock for multi-process safety
                    with _exclusive_file_lock(cache_path, mode="a") as cache_file:
                        for entry in entries:
                            plain = json.dumps(entry, ensure_ascii=False, separators=(",", ":"))
                            cache_file.write(_encode_cache_line(plain) + "\n")
                    return
                except OSError as exc:
                    # Retry only on transient access errors (file locked by another process)
                    if exc.errno not in {errno.EACCES, errno.EAGAIN, errno.EWOULDBLOCK, errno.EBUSY}:
                        return
                    if attempt >= max_retries:
                        return
                    # Exponential backoff: 50ms, 100ms, 200ms (aligned with C# implementation)
                    time.sleep(0.05 * (2**attempt))
        except Exception:
            # Fail silently - telemetry errors should not crash the application
            return

    def _flush_cache(self) -> None:
        """Flush this process's cached events back to telemetry service."""
        cache_path = self.cache_path
        if cache_path is None or not cache_path.exists():
            return

        self._flush_cache_file(cache_path)

    def _flush_cache_file(self, cache_path: Path) -> None:
        """Flush cached events back to telemetry service.

        Approach:
        1. Atomically rename cache → .flush (claims ownership, prevents concurrent flushes)
        2. Read all events from .flush file
        3. Queue all events for sending via telemetry logger
        4. Force flush with 2-second timeout
        5. On success: delete .flush file
        6. On failure: restore .flush → cache for retry

        Multi-process coordination:
        - `replace()` is atomic; only one process can successfully rename the cache file
        - If another process already renamed it, we get FileNotFoundError and abort
        - Stale .flush files from crashes are overwritten by the atomic rename

        Shutdown handling:
        - If shutdown flag set during flush, restore cache before returning
        - This preserves events even if callbacks don't fire during shutdown

        Callback behavior:
        - Queued events trigger callbacks with success/failure
        - Failed events are automatically re-cached via callbacks (unless shutting down)
        - The _is_flushing flag prevents re-caching of replayed events during flush
        """
        flush_path = None
        try:
            # Check shutdown before starting (under lock to prevent race)
            with self._lock:
                if self._shutdown:
                    return

            if not cache_path.exists():
                return

            # Atomically rename to .flush file to claim ownership
            # Overwrite any stale .flush file from crashed process (C# pattern)
            flush_path = cache_path.with_name(f"{cache_path.name}.flush")
            try:
                # On Windows/POSIX, replace() overwrites existing files atomically
                cache_path.replace(flush_path)
            except FileNotFoundError:
                # Cache already claimed by another flush or doesn't exist
                return

            # Read all cached entries (base64-decoded)
            entries = _read_cache_entries(flush_path)

            if not entries:
                # Empty cache, just delete the flush file
                flush_path.unlink(missing_ok=True)
                return

            # Replay all events through telemetry logger
            # Note: _is_flushing flag (set by caller) prevents these callbacks from re-caching or triggering nested flushes
            # (unlikely since we just successfully sent an event, indicating network is available)
            for entry in entries:
                try:
                    event_name = entry["event_name"]
                    event_data = entry["event_data"]
                    if not event_name or not event_data:
                        continue
                    attributes = json.loads(event_data)
                    if not isinstance(attributes, dict):
                        continue
                    # Preserve original timestamp
                    attributes["initTs"] = entry.get("initTs", entry["ts"])
                    self._telemetry.log(event_name, attributes, None)
                except Exception:
                    # Skip malformed entries
                    continue

            # Check if shutdown happened during flush
            with self._lock:
                if self._shutdown:
                    # Restore cache to avoid data loss during shutdown
                    if flush_path and flush_path.exists():
                        try:
                            cache_path.parent.mkdir(parents=True, exist_ok=True)
                            flush_path.replace(cache_path)
                        except Exception:
                            # Silently ignore errors during cleanup
                            pass
                    return

            # Cleanup based on flush result
            flush_success = False
            with self._callback_condition:
                callbacks_item_count = self._callbacks_item_count
                expected_items = self._events_logged
                if callbacks_item_count >= expected_items:
                    flush_success = True
            if flush_success:
                # Success: delete the flush file (events were sent)
                if flush_path:
                    flush_path.unlink(missing_ok=True)
            elif flush_path and flush_path.exists():
                # Failure: restore cache for retry later
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                flush_path.replace(cache_path)
        except Exception:
            # Best-effort restore on any exception to prevent data loss
            if flush_path and flush_path.exists():
                try:
                    cache_path.parent.mkdir(parents=True, exist_ok=True)
                    flush_path.replace(cache_path)
                except Exception:
                    # If restore fails, we lose the data (acceptable for telemetry)
                    pass
            return

    @property
    def is_flushing(self) -> bool:
        with self._lock:
            return self._is_flushing


class Telemetry:
    """Wrapper that wires environment configuration into the library logger.

    This is a singleton class - all instances share the same state.
    Use Telemetry() to get the singleton instance.
    """

    _instance: Optional["Telemetry"] = None
    _lock = threading.Lock()

    def __new__(cls):
        """Create or return the singleton instance.

        Thread-safe singleton implementation using double-checked locking.
        """
        if cls._instance is None:
            with cls._lock:
                # Double-check pattern to prevent race conditions
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        """Initialize the telemetry logger (only runs once for singleton)."""
        # Prevent re-initialization
        if self._initialized:
            return

        self._logger = self._create_logger()
        event_source.disable()

        self._cache_handler = TelemetryCacheHandler(self)
        self._initialized = True
        self._setup_payload_callbacks()
        self._log_heartbeat()
        if os.environ.get("OLIVE_DISABLE_TELEMETRY") == "1":
            self.disable_telemetry()

    def _create_logger(self) -> Optional[TelemetryLogger]:
        try:
            return get_telemetry_logger(base64.b64decode(CONNECTION_STRING).decode())
        except Exception:
            return None

    def _setup_payload_callbacks(self) -> None:
        # Register callback for payload transmission events
        # No need to store unregister function - logger shutdown will clean up callbacks
        self._logger.register_payload_transmitted_callback(
            self._cache_handler.on_payload_transmitted,
            include_failures=True,
        )

    def add_global_metadata(self, metadata: dict[str, Any]) -> None:
        """Add metadata to all telemetry events.

        Args:
            metadata: Dictionary of metadata key-value pairs to add to all events.
                     These will be included in every telemetry event sent.

        Example:
            >>> telemetry = Telemetry()
            >>> telemetry.add_global_metadata({"user_id": "12345", "environment": "production"})

        """
        self._logger.add_global_metadata(metadata)

    def log(
        self,
        event_name: str,
        attributes: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a telemetry event.

        Args:
            event_name: Name of the event to log (e.g., "UserLogin", "ModelTrained").
            attributes: Optional dictionary of event-specific attributes.
            metadata: Optional dictionary of additional metadata to merge with attributes.

        Example:
            >>> telemetry = Telemetry()
            >>> telemetry.log("ModelOptimized", {"model_type": "bert", "duration_ms": 1500})

        """
        attrs = _merge_metadata(attributes, metadata)
        self._logger.log(event_name, attrs)
        if self._cache_handler:
            self._cache_handler.record_event_logged()

    def _log_heartbeat(
        self,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a heartbeat event with system information.

        Args:
            metadata: Optional additional metadata to include.

        """
        encrypted_device_id, device_id_status = get_encrypted_device_id_and_status()
        attributes = {
            "device_id": encrypted_device_id,
            "id_status": device_id_status.value,
            "os": {
                "name": platform.system().lower(),
                "version": platform.version(),
                "release": platform.release(),
                "arch": platform.machine(),
            },
        }
        self.log(HEARTBEAT_EVENT_NAME, attributes, metadata)

    def disable_telemetry(self) -> None:
        """Disable all telemetry logging.

        After calling this method, no telemetry events will be sent until
        telemetry is explicitly re-enabled.
        """
        self._logger.disable_telemetry()

    def shutdown(self, timeout_millis: float = 10_000, callback_timeout_millis: float = 2_000) -> None:
        """Shutdown telemetry and flush pending events.

        Shutdown sequence:
        1. Wait for in-flight flush to complete (up to 1 second)
        2. Wait for callbacks + signal shutdown to cache handler
        3. Shutdown logger (cleans up callbacks automatically)
        """
        # Step 1: Wait for pending flush to complete (matches C# 1-second timeout)
        start_time = time.time()
        while time.time() - start_time < 1.0:
            if not self._cache_handler or not self._cache_handler.is_flushing:
                break
            time.sleep(0.05)

        # Step 2: Wait for callbacks/flush to complete before shutting down cache handler
        if self._cache_handler:
            # Nothing can be done if callbacks don't complete in time, so we ignore the result
            _ = self._cache_handler.wait_for_callbacks(callback_timeout_millis / 1000)
            self._cache_handler.shutdown()

        # Step 3: Shutdown logger (callbacks cleaned up automatically)
        self._logger.shutdown()

    def __del__(self):
        """Cleanup telemetry resources on garbage collection.

        This is a safety net to ensure resources are cleaned up even if
        shutdown() is not explicitly called. However, relying on __del__
        is not recommended - always call shutdown() explicitly when done.
        """
        try:
            self.shutdown()
        except Exception:
            # Silently ignore errors during cleanup
            pass


def _get_logger() -> Telemetry:
    """Get or create the singleton Telemetry instance."""
    return Telemetry()


def _merge_metadata(attributes: Optional[dict[str, Any]], metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(attributes or {})
    if metadata:
        merged.update(metadata)
    return merged


def _parse_payload(payload: bytes) -> list[dict[str, Any]]:
    """Parse telemetry payload into individual event entries.

    Design decisions:
    - Filter events to only allowed keys (privacy/security)
    - Store as minimal JSON (reduces cache size)
    - Fail silently on malformed data (telemetry should be robust)

    Assumptions:
    - Payload is newline-delimited JSON (OneCollector format)
    - Events have "name", "time", and "data" fields
    - Only whitelisted events and fields should be cached
    """
    entries = []
    try:
        payload_text = payload.decode("utf-8")
        lines = payload_text.splitlines()

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                event_name = event["name"]
                if not event_name:
                    continue
                # Filter to only allowed keys for privacy/security
                filtered_data = _filter_event_data(event_name, event["data"])
                if not filtered_data:
                    continue
                entries.append(
                    {
                        "ts": event["time"] or time.time(),
                        "event_name": event_name,
                        # Compact JSON to reduce cache size
                        "event_data": json.dumps(filtered_data, ensure_ascii=False, separators=(",", ":")),
                    }
                )
            except Exception:
                # Skip malformed lines
                continue
    except Exception:
        # If entire payload is malformed, return empty list
        return []

    return entries


def _filter_event_data(event_name: str, data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Filter event data to only allowed keys for privacy/security.

    Design decisions:
    - Whitelist approach (only explicitly allowed keys are included)
    - Support nested keys with dot notation (e.g., "os.name")
    - Return None if no allowed keys found (filters out unknown events)

    Assumptions:
    - ALLOWED_KEYS dict defines all cacheable events and their fields
    - Unknown events should not be cached (privacy/security)
    """
    if event_name not in ALLOWED_KEYS:
        return None
    allowed_keys = ALLOWED_KEYS[event_name]

    filtered: dict[str, Any] = {}
    for key in allowed_keys:
        value = _get_nested_value(data, key)
        if value is None:
            continue
        _set_nested_value(filtered, key, value)
    return filtered or None


def _get_nested_value(data: dict[str, Any], key: str) -> Any:
    current = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested_value(data: dict[str, Any], key: str, value: Any) -> None:
    current = data
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value


def _read_cache_entries(cache_path: Path) -> list[dict[str, Any]]:
    """Read all entries from a cache file, decoding each line.

    Design decisions:
    - Use file locking for multi-process safety
    - Continue reading past malformed entries (partial data recovery)
    - Return empty list on complete read failure (fail gracefully)
    - Each line is base64-decoded before JSON parsing.

    Assumptions:
    - Cache file contains newline-delimited base64-encoded entries (one per line)
    - Each line is independent (one malformed line doesn't affect others)
    - Empty or whitespace-only lines are skipped
    """
    entries = []
    try:
        with _exclusive_file_lock(cache_path, mode="r") as cache_file:
            for raw_line in cache_file:
                line = raw_line.strip()
                if not line:
                    continue
                try:
                    line = _decode_cache_line(line)
                    if isinstance(line, dict):
                        entries.append(line)
                except Exception:
                    # Malformed line, skip and continue
                    continue
    except Exception:
        # If file cannot be opened or read, return empty list
        return []
    return entries

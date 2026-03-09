# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Retry logic with exponential backoff for OneCollector exporter."""

import random
import threading
from time import time
from typing import Callable, Optional

from olive.telemetry.library.event_source import event_source
from olive.telemetry.library.transport import HttpJsonPostTransport


class RetryHandler:
    """Handles retry logic with exponential backoff and jitter.

    Implements retry strategy matching the .NET implementation.
    """

    def __init__(self, max_retries: int = 6, base_delay: float = 1.0, max_delay: float = 60.0):
        """Initialize retry handler.

        Args:
            max_retries: Maximum number of retry attempts
            base_delay: Base delay for exponential backoff (seconds)
            max_delay: Maximum delay between retries (seconds)

        """
        self.max_retries = max_retries
        self.base_delay = base_delay
        self.max_delay = max_delay

    def execute_with_retry(
        self,
        operation: Callable[[], tuple[bool, Optional[int]]],
        deadline_sec: float,
        shutdown_event: threading.Event,
    ) -> bool:
        """Execute an operation with retry logic.

        Args:
            operation: Function that returns (success, status_code)
            deadline_sec: Absolute deadline timestamp
            shutdown_event: Event to signal shutdown

        Returns:
            True if operation succeeded, False otherwise

        """
        for retry_num in range(self.max_retries):
            # Check if we've exceeded the deadline
            remaining_time = deadline_sec - time()
            if remaining_time <= 0:
                return False

            try:
                # Execute the operation
                success, status_code = operation()

                if success:
                    return True

                # Check if response is retryable
                if not HttpJsonPostTransport.is_retryable(status_code):
                    return False

            except Exception as ex:
                event_source.export_exception_thrown("RetryHandler", ex)

                # Last retry - don't wait
                if retry_num + 1 == self.max_retries:
                    return False

            # Last retry - failed
            if retry_num + 1 == self.max_retries:
                return False

            # Calculate backoff with exponential increase and jitter
            backoff = min(self.base_delay * (2**retry_num), self.max_delay)
            # Add +/-20% jitter
            backoff *= random.uniform(0.8, 1.2)

            # Don't wait longer than remaining time
            remaining_time = deadline_sec - time()
            wait_time = min(backoff, remaining_time)

            if wait_time <= 0:
                return False

            # Wait with ability to interrupt on shutdown
            if shutdown_event.wait(wait_time):
                # Shutdown occurred
                return False

        return False

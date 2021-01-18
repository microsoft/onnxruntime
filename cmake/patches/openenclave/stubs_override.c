// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <stdlib.h>
#include <unistd.h>
#include <sys/types.h>
#include <time.h>

// The following functions are included in Open Enclave's standard
// library and have an implementation that aborts the program.
// Here we override the implementation to avoid #ifdef'ing existing code.

// See https://github.com/Microsoft/openenclave/blob/master/libc/time.c.
size_t strftime(char *s, size_t max, const char *format, const struct tm *tm)
{
  (void)s;
  (void)max;
  (void)format;
  (void)tm;
  return 0;
}

// clock_gettime(CLOCK_MONOTONIC, ..) is called by
// std::chrono::high_resolution_clock::now().
// Open Enclave only supports CLOCK_REALTIME (via an ocall), every other
// clock would normally result in an abort.
// For now, let's pretend all clocks behave like CLOCK_REALTIME.
// clock_gettime is defined as weak alias for __clock_gettime,
// meaning we have a facility to provide an alternative implementation
// and also re-use the existing one.
// https://github.com/Microsoft/openenclave/blob/6ee7306a84b7/3rdparty/musl/musl/src/time/clock_gettime.c#L55
int __clock_gettime(clockid_t clk, struct timespec *ts);
int clock_gettime(clockid_t clk, struct timespec *ts) {
    if (clk != CLOCK_REALTIME) {
        clk = CLOCK_REALTIME;
    }
    return __clock_gettime(clk, ts);
}

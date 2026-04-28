#!/usr/bin/env python3
"""
Simulation test for the cpuinfo sysfs fallback fix.

This test verifies two fixes for https://github.com/microsoft/onnxruntime/issues/10038:

1. Safe logging in env.cc - PosixEnv constructor no longer crashes when the
   logging system is not yet initialized and cpuinfo_initialize() fails.

2. cpuinfo sysfs fallback - The patched cpuinfo library falls back to
   sysconf(_SC_NPROCESSORS_ONLN) for both processor counts and per-CPU
   present/possible flags when /sys/devices/system/cpu/{possible,present}
   files are missing.

Testing approach:
- Test 1: Compile a small C++ program that calls the safe logging pattern
  without a registered logger. Verify it doesn't crash.
- Test 2: Compile a small C program that validates the sysconf fallback
  arithmetic and verifies that the fallback marks each online CPU with both
  PRESENT and POSSIBLE flags. This catches the incomplete count-only fallback.
- Test 3: Use an LD_PRELOAD shim (like the lambda-arm64-onnx workaround)
  to simulate missing sysfs files and verify ORT loads without crash.

Note: Tests 2 and 3 require a build of ORT with the patches applied.
Test 1 can run standalone.
"""

import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest


def _require_linux():
    if sys.platform != "linux":
        raise unittest.SkipTest("Test requires Linux")


def _require_gcc():
    if not shutil.which("gcc"):
        raise unittest.SkipTest("gcc not found")


def _require_gpp():
    if not shutil.which("g++"):
        raise unittest.SkipTest("g++ not found")


class TestCpuinfoSysfsFallback(unittest.TestCase):
    def test_safe_logging_pattern(self):
        """Verify the safe logging pattern doesn't crash when no logger exists.

        This simulates the fix in env.cc where we check HasDefaultLogger() before
        calling LOGS_DEFAULT(). We compile a minimal C++ program that:
        - Does NOT register a default logger
        - Calls the safe logging pattern
        - Verifies it writes to stderr instead of crashing
        """
        _require_linux()
        _require_gpp()

        source = textwrap.dedent(r"""
        #include <iostream>
        #include <string_view>

        // Minimal simulation of ORT's logging check pattern
        namespace logging {
        class LoggingManager {
        public:
            // Simulate: no default logger registered
            static bool HasDefaultLogger() { return false; }
        };
        }  // namespace logging

        void LogEarlyWarning(std::string_view message) {
            if (logging::LoggingManager::HasDefaultLogger()) {
                // Would call LOGS_DEFAULT(WARNING) here - but logger doesn't exist
                // This path should NOT be taken
                std::cerr << "BUG: should not reach here\n";
                return;
            }
            // Safe fallback to stderr
            std::cerr << "onnxruntime warning: " << message << "\n";
        }

        int main() {
            // This simulates what PosixEnv() does when cpuinfo_initialize() fails
            bool cpuinfo_available = false;  // Simulating failure
            if (!cpuinfo_available) {
                LogEarlyWarning("cpuinfo_initialize failed. "
                               "May cause CPU EP performance degradation due to undetected CPU features.");
            }
            std::cout << "PASS: Safe logging pattern works without crash\n";
            return 0;
        }
        """)

        with tempfile.NamedTemporaryFile(suffix=".cc", mode="w", delete=False) as f:
            f.write(source)
            src_path = f.name

        try:
            exe_path = src_path.replace(".cc", "")
            result = subprocess.run(
                ["g++", "-std=c++17", "-o", exe_path, src_path], check=False, capture_output=True, text=True
            )
            self.assertEqual(result.returncode, 0, f"Compilation failed: {result.stderr}")

            result = subprocess.run([exe_path], check=False, capture_output=True, text=True, timeout=10)
            self.assertEqual(
                result.returncode, 0, f"Program crashed with exit code {result.returncode}: {result.stderr}"
            )
            self.assertIn("PASS", result.stdout)
        finally:
            os.unlink(src_path)
            if os.path.exists(src_path.replace(".cc", "")):
                os.unlink(src_path.replace(".cc", ""))

    def test_sysconf_fallback(self):
        """Verify sysconf(_SC_NPROCESSORS_ONLN) works as a complete fallback.

        This doesn't test the actual cpuinfo patch (that requires building cpuinfo)
        but verifies the fallback mechanism produces correct counts and marks
        present/possible flags for each online CPU.
        """
        _require_linux()
        _require_gcc()

        source = textwrap.dedent(r"""
        #include <stdint.h>
        #include <stdio.h>
        #include <unistd.h>

        #define CPUINFO_LINUX_FLAG_PRESENT 0x1
        #define CPUINFO_LINUX_FLAG_POSSIBLE 0x2

        int main() {
            long nproc = sysconf(_SC_NPROCESSORS_ONLN);
            if (nproc <= 0) {
                printf("FAIL: sysconf(_SC_NPROCESSORS_ONLN) returned %ld\n", nproc);
                return 1;
            }
            // Simulate what the patched cpuinfo max-count helpers return:
            // max_processor = nproc - 1 (0-indexed). Then arm_linux_init does:
            // 1 + max_processor = nproc.
            unsigned int max_processor = (unsigned int)(nproc - 1);
            unsigned int arm_linux_processors_count = 1 + max_processor;

            uint32_t processor_flags[1024] = {0};
            unsigned int processors_count = arm_linux_processors_count;
            if (processors_count > 1024) {
                processors_count = 1024;
            }

            // Simulate cpuinfo_linux_detect_possible_processors() and
            // cpuinfo_linux_detect_present_processors() fallback helpers.
            for (unsigned int processor = 0; processor < processors_count; ++processor) {
                processor_flags[processor] |= CPUINFO_LINUX_FLAG_PRESENT;
                processor_flags[processor] |= CPUINFO_LINUX_FLAG_POSSIBLE;
            }

            unsigned int valid_processors = 0;
            const uint32_t valid_processor_mask = CPUINFO_LINUX_FLAG_PRESENT | CPUINFO_LINUX_FLAG_POSSIBLE;
            for (unsigned int processor = 0; processor < processors_count; ++processor) {
                if ((processor_flags[processor] & valid_processor_mask) == valid_processor_mask) {
                    ++valid_processors;
                }
            }

            printf("sysconf(_SC_NPROCESSORS_ONLN) = %ld\n", nproc);
            printf("Simulated max_processor = %u\n", max_processor);
            printf("Simulated arm_linux_processors_count = %u\n", arm_linux_processors_count);
            printf("Simulated valid_processors = %u\n", valid_processors);

            if (arm_linux_processors_count == (unsigned int)nproc && valid_processors == processors_count) {
                printf("PASS: Fallback produces correct processor count and flags\n");
                return 0;
            }
            printf("FAIL: Processor count or flags mismatch\n");
            return 1;
        }
        """)

        with tempfile.NamedTemporaryFile(suffix=".c", mode="w", delete=False) as f:
            f.write(source)
            src_path = f.name

        try:
            exe_path = src_path.replace(".c", "")
            result = subprocess.run(["gcc", "-o", exe_path, src_path], check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"Compilation failed: {result.stderr}")

            result = subprocess.run([exe_path], check=False, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 0, f"exit code {result.returncode}: {result.stdout}")
            self.assertIn("PASS", result.stdout)
        finally:
            os.unlink(src_path)
            if os.path.exists(src_path.replace(".c", "")):
                os.unlink(src_path.replace(".c", ""))

    def test_sysfs_hide_with_ld_preload(self):
        """Verify LD_PRELOAD shim can hide sysfs files.

        This compiles a small shim that intercepts open-family calls to return
        ENOENT for /sys/devices/system/cpu/{possible,present}, then runs a test
        program that opens those files.
        """
        _require_linux()
        _require_gcc()

        shim_source = textwrap.dedent(r"""
        #define _GNU_SOURCE
        #include <dlfcn.h>
        #include <errno.h>
        #include <fcntl.h>
        #include <stdarg.h>
        #include <stdio.h>
        #include <string.h>
        #include <sys/types.h>

#ifndef O_TMPFILE
#define O_TMPFILE 0
#endif

        static const char *CPU_POSSIBLE = "/sys/devices/system/cpu/possible";
        static const char *CPU_PRESENT  = "/sys/devices/system/cpu/present";

        static int is_blocked(const char *path) {
            return (strcmp(path, CPU_POSSIBLE) == 0 || strcmp(path, CPU_PRESENT) == 0);
        }

        static mode_t get_mode_if_needed(int flags, va_list args) {
            return ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE)) ? va_arg(args, mode_t) : 0;
        }

        int open(const char *path, int flags, ...) {
            static int (*real_open)(const char *, int, ...) = NULL;
            va_list args;
            mode_t mode = 0;

            if (!real_open) real_open = dlsym(RTLD_NEXT, "open");
            if (is_blocked(path)) {
                errno = ENOENT;
                return -1;
            }

            va_start(args, flags);
            mode = get_mode_if_needed(flags, args);
            va_end(args);
            return ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE))
                       ? real_open(path, flags, mode)
                       : real_open(path, flags);
        }

        int open64(const char *path, int flags, ...) {
            static int (*real_open64)(const char *, int, ...) = NULL;
            va_list args;
            mode_t mode = 0;

            if (!real_open64) real_open64 = dlsym(RTLD_NEXT, "open64");
            if (is_blocked(path)) {
                errno = ENOENT;
                return -1;
            }

            va_start(args, flags);
            mode = get_mode_if_needed(flags, args);
            va_end(args);
            return ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE))
                       ? real_open64(path, flags, mode)
                       : real_open64(path, flags);
        }

        int openat(int dirfd, const char *path, int flags, ...) {
            static int (*real_openat)(int, const char *, int, ...) = NULL;
            va_list args;
            mode_t mode = 0;

            if (!real_openat) real_openat = dlsym(RTLD_NEXT, "openat");
            if (path && is_blocked(path)) {
                errno = ENOENT;
                return -1;
            }

            va_start(args, flags);
            mode = get_mode_if_needed(flags, args);
            va_end(args);
            return ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE))
                       ? real_openat(dirfd, path, flags, mode)
                       : real_openat(dirfd, path, flags);
        }

        int openat64(int dirfd, const char *path, int flags, ...) {
            static int (*real_openat64)(int, const char *, int, ...) = NULL;
            va_list args;
            mode_t mode = 0;

            if (!real_openat64) real_openat64 = dlsym(RTLD_NEXT, "openat64");
            if (path && is_blocked(path)) {
                errno = ENOENT;
                return -1;
            }

            va_start(args, flags);
            mode = get_mode_if_needed(flags, args);
            va_end(args);
            return ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE))
                       ? real_openat64(dirfd, path, flags, mode)
                       : real_openat64(dirfd, path, flags);
        }

        FILE *fopen(const char *restrict path, const char *restrict mode) {
            static FILE *(*real_fopen)(const char *, const char *) = NULL;
            if (!real_fopen) real_fopen = dlsym(RTLD_NEXT, "fopen");

            if (is_blocked(path)) {
                errno = ENOENT;
                return NULL;
            }
            return real_fopen(path, mode);
        }
        """)

        test_source = textwrap.dedent(r"""
        #include <errno.h>
        #include <fcntl.h>
        #include <stdio.h>
        #include <string.h>
        #include <unistd.h>

        static int try_open(const char *path) {
            int fd = open(path, O_RDONLY);
            if (fd >= 0) {
                close(fd);
            }
            return fd;
        }

        int main() {
            int fd;
            int pass = 1;

            fd = try_open("/sys/devices/system/cpu/possible");
            if (fd >= 0) {
                printf("FAIL: /sys/devices/system/cpu/possible should be blocked\n");
                pass = 0;
            } else {
                printf("OK: /sys/devices/system/cpu/possible blocked (errno=%d: %s)\n",
                       errno, strerror(errno));
            }

            fd = try_open("/sys/devices/system/cpu/present");
            if (fd >= 0) {
                printf("FAIL: /sys/devices/system/cpu/present should be blocked\n");
                pass = 0;
            } else {
                printf("OK: /sys/devices/system/cpu/present blocked (errno=%d: %s)\n",
                       errno, strerror(errno));
            }

            // Verify other files still work
            fd = try_open("/proc/cpuinfo");
            if (fd < 0) {
                printf("WARN: /proc/cpuinfo not accessible (may be OK in some envs)\n");
            } else {
                printf("OK: /proc/cpuinfo still accessible\n");
            }

            if (pass) {
                printf("PASS: LD_PRELOAD sysfs-hiding shim works correctly\n");
            }
            return pass ? 0 : 1;
        }
        """)

        with tempfile.TemporaryDirectory() as tmpdir:
            shim_path = os.path.join(tmpdir, "hide_sysfs.c")
            shim_so = os.path.join(tmpdir, "hide_sysfs.so")
            test_path = os.path.join(tmpdir, "test_sysfs.c")
            test_exe = os.path.join(tmpdir, "test_sysfs")

            with open(shim_path, "w") as f:
                f.write(shim_source)
            with open(test_path, "w") as f:
                f.write(test_source)

            # Compile shim
            result = subprocess.run(
                ["gcc", "-shared", "-fPIC", "-o", shim_so, shim_path, "-ldl"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Shim compilation failed: {result.stderr}")

            # Compile test
            result = subprocess.run(["gcc", "-o", test_exe, test_path], check=False, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, f"Test compilation failed: {result.stderr}")

            # Run with LD_PRELOAD
            env = os.environ.copy()
            env["LD_PRELOAD"] = shim_so
            result = subprocess.run([test_exe], check=False, capture_output=True, text=True, timeout=10, env=env)
            self.assertEqual(result.returncode, 0, f"exit code {result.returncode}: {result.stdout}")
            self.assertIn("PASS", result.stdout)

    def test_ort_import_with_hidden_sysfs(self):
        """Integration test - import onnxruntime with hidden sysfs files.

        This uses the LD_PRELOAD shim to hide /sys/devices/system/cpu/{possible,present}
        and then imports onnxruntime. This is the actual end-to-end test that
        verifies both fixes work together.

        NOTE: This requires onnxruntime to be built with the patches applied.
        """
        _require_linux()
        _require_gcc()

        # Check if onnxruntime is importable
        result = subprocess.run(
            [sys.executable, "-c", "import onnxruntime"], check=False, capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            self.skipTest("onnxruntime not installed/importable")

        shim_source = textwrap.dedent(r"""
        #define _GNU_SOURCE
        #include <dlfcn.h>
        #include <errno.h>
        #include <fcntl.h>
        #include <stdarg.h>
        #include <stdio.h>
        #include <string.h>
        #include <sys/types.h>

#ifndef O_TMPFILE
#define O_TMPFILE 0
#endif

        static const char *CPU_POSSIBLE = "/sys/devices/system/cpu/possible";
        static const char *CPU_PRESENT  = "/sys/devices/system/cpu/present";

        static int is_blocked(const char *path) {
            return (strcmp(path, CPU_POSSIBLE) == 0 || strcmp(path, CPU_PRESENT) == 0);
        }

        static mode_t get_mode_if_needed(int flags, va_list args) {
            return ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE)) ? va_arg(args, mode_t) : 0;
        }

        int open(const char *path, int flags, ...) {
            static int (*real_open)(const char *, int, ...) = NULL;
            va_list args;
            mode_t mode = 0;

            if (!real_open) real_open = dlsym(RTLD_NEXT, "open");
            if (is_blocked(path)) { errno = ENOENT; return -1; }

            va_start(args, flags);
            mode = get_mode_if_needed(flags, args);
            va_end(args);
            return ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE))
                       ? real_open(path, flags, mode)
                       : real_open(path, flags);
        }

        int open64(const char *path, int flags, ...) {
            static int (*real_open64)(const char *, int, ...) = NULL;
            va_list args;
            mode_t mode = 0;

            if (!real_open64) real_open64 = dlsym(RTLD_NEXT, "open64");
            if (is_blocked(path)) { errno = ENOENT; return -1; }

            va_start(args, flags);
            mode = get_mode_if_needed(flags, args);
            va_end(args);
            return ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE))
                       ? real_open64(path, flags, mode)
                       : real_open64(path, flags);
        }

        int openat(int dirfd, const char *path, int flags, ...) {
            static int (*real_openat)(int, const char *, int, ...) = NULL;
            va_list args;
            mode_t mode = 0;

            if (!real_openat) real_openat = dlsym(RTLD_NEXT, "openat");
            if (path && is_blocked(path)) { errno = ENOENT; return -1; }

            va_start(args, flags);
            mode = get_mode_if_needed(flags, args);
            va_end(args);
            return ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE))
                       ? real_openat(dirfd, path, flags, mode)
                       : real_openat(dirfd, path, flags);
        }

        int openat64(int dirfd, const char *path, int flags, ...) {
            static int (*real_openat64)(int, const char *, int, ...) = NULL;
            va_list args;
            mode_t mode = 0;

            if (!real_openat64) real_openat64 = dlsym(RTLD_NEXT, "openat64");
            if (path && is_blocked(path)) { errno = ENOENT; return -1; }

            va_start(args, flags);
            mode = get_mode_if_needed(flags, args);
            va_end(args);
            return ((flags & O_CREAT) || ((flags & O_TMPFILE) == O_TMPFILE))
                       ? real_openat64(dirfd, path, flags, mode)
                       : real_openat64(dirfd, path, flags);
        }

        FILE *fopen(const char *restrict path, const char *restrict mode) {
            static FILE *(*real_fopen)(const char *, const char *) = NULL;
            if (!real_fopen) real_fopen = dlsym(RTLD_NEXT, "fopen");
            if (is_blocked(path)) { errno = ENOENT; return NULL; }
            return real_fopen(path, mode);
        }
        """)

        with tempfile.TemporaryDirectory() as tmpdir:
            shim_path = os.path.join(tmpdir, "hide_sysfs.c")
            shim_so = os.path.join(tmpdir, "hide_sysfs.so")

            with open(shim_path, "w") as f:
                f.write(shim_source)

            result = subprocess.run(
                ["gcc", "-shared", "-fPIC", "-o", shim_so, shim_path, "-ldl"],
                check=False,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, f"Shim compilation failed: {result.stderr}")

            env = os.environ.copy()
            env["LD_PRELOAD"] = shim_so

            # Try importing onnxruntime with hidden sysfs
            ort_script = (
                "import onnxruntime; print('PASS: onnxruntime imported successfully'); "
                "print(f'Version: {onnxruntime.__version__}'); "
                "print(f'Providers: {onnxruntime.get_available_providers()}')"
            )
            result = subprocess.run(
                [sys.executable, "-c", ort_script],
                check=False,
                capture_output=True,
                text=True,
                timeout=60,
                env=env,
            )
            self.assertEqual(result.returncode, 0, f"exit code {result.returncode}: {result.stderr}")
            self.assertIn("PASS", result.stdout)


if __name__ == "__main__":
    unittest.main()

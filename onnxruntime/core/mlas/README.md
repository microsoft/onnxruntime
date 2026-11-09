# About MLAS
MLAS is a compute library containing processor optimized GEMM kernels and platform specific threading code.

# Unit tests for MLAS
Unit tests for the SGEMM kernels are available under onnxruntime\test\mlas. These tests run over a range of inputs that then execute the various special cases for aligned and unaligned outputs. The tests have failed if any "mismatch" strings are printed.

Kernels behind runtime CPUID dispatch (AVX-512, AMX, AVX-VNNI-INT8, AVX-NE-CONVERT) only run where the hardware supports them; [docs/MLAS_ISA_Validation.md](../../../docs/MLAS_ISA_Validation.md) describes how to exercise and verify those tiers with Intel SDE on machines that lack them.

On Linux x64, AMX initialization also checks that tile data survives CPU migration in a temporary thread. If tile data is lost or the check cannot run (for example, fewer than two CPUs are allowed, affinity changes are blocked, or the CPU affinity mask exceeds `CPU_SETSIZE`), MLAS keeps the non-AMX QGEMM dispatch. The check does not change the caller's CPU affinity.

The probe thread is always joinable. On Linux libcs that support pthread cancellation, caller cancellation is temporarily disabled until the worker has finished using its stack-owned context, then the caller's previous cancellation state is restored. Android/bionic does not support pthread cancellation, so it uses the same probe and join without the cancellation-only guard and tests. The `AmxPlatform` regression tests exercise permission masks, migration failure, preservation decisions and pending caller cancellation through architecture-independent probe callbacks. These callbacks do not validate a hypervisor's actual AMX state handling; the hardware-dependent dispatch test requires AMX-TILE and AMX-INT8.

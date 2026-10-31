# About MLAS
MLAS is a compute library containing processor optimized GEMM kernels and platform specific threading code.

# Unit tests for MLAS
Unit tests for the SGEMM kernels are available under onnxruntime\test\mlas. These tests run over a range of inputs that then execute the various special cases for aligned and unaligned outputs. The tests have failed if any "mismatch" strings are printed.

Kernels behind runtime CPUID dispatch (AVX-512, AMX, AVX-VNNI-INT8, AVX-NE-CONVERT) only run where the hardware supports them; [docs/MLAS_ISA_Validation.md](../../../docs/MLAS_ISA_Validation.md) describes how to exercise and verify those tiers with Intel SDE on machines that lack them.


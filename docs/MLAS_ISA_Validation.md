# Validating MLAS ISA-dispatched kernels with Intel SDE

MLAS selects kernels at runtime from CPUID, so a test suite passing on a given machine only
proves the kernels that machine can reach. Several dispatch tiers (AVX-VNNI-INT8,
AVX-NE-CONVERT, AMX, and AVX-512 on non-AVX-512 development machines) may never execute in a
local run or on a given CI pool, while the suite still reports green because the fallback ran
instead. Intel's Software Development Emulator (SDE) closes that gap: it emulates a chosen CPU
generation, restricting or providing CPUID so the dispatch selects the tier under test. Issue
[#29862](https://github.com/microsoft/onnxruntime/issues/29862) has measured examples of
kernels no CI machine currently reaches.

## Setup

SDE is a free download from Intel (the Software Development Emulator kit). Unpack it anywhere;
the binary is `sde64` (`sde64.exe` in the native Windows kit). On Windows hosts where the Pin instrumentation SDE depends on is blocked
by endpoint security software, running the Linux SDE inside WSL2 against a Linux build of
`onnxruntime_mlas_test` works and is the setup the numbers below come from.

Build the MLAS test binary as usual:

```
python3 tools/ci_build/build.py --config Release --build_dir build/sde --parallel \
  --cmake_generator Ninja --target onnxruntime_mlas_test --skip_tests
```

## Running a tier

```
sde64 -<chip> -- build/sde/Release/onnxruntime_mlas_test --gtest_filter=<filter>
```

Useful chip flags and what they exercise on top of a plain AVX2 baseline:

| flag | emulates | reaches |
| --- | --- | --- |
| `-hsw` | Haswell | plain AVX2 tables (no AVX-VNNI) |
| `-adl` | Alder Lake | AVX2 + AVX-VNNI tables |
| `-skx` | Skylake-X | AVX-512 core tables |
| `-clx` | Cascade Lake | AVX-512 VNNI |
| `-spr` | Sapphire Rapids | AMX (`TDPB*`) plus the AVX-512 tiers |
| `-srf` | Sierra Forest | AVX-VNNI-INT8 and AVX-NE-CONVERT |
| `-gnr` | Granite Rapids | the server superset without the client-only tiers |

Two behaviors worth knowing:

- Running `sde64` with no chip flag exposes every feature SDE knows about. That is not a
  native baseline; always pass a chip flag when the point is to pin a tier.
- The slowdown is modest for correctness runs, roughly 2 to 3x over native for the MLAS
  suite, so full-suite runs per tier are practical.

## Proving the kernel actually ran

A green suite does not by itself prove the intended kernel executed: if the dispatch gate did
not open, the fallback ran and also passes. SDE's instruction histogram settles it:

```
sde64 -<chip> -mix -omix mix.txt -- <binary> --gtest_filter=<filter>
```

Count the tier's signature instructions in the dynamic sections of `mix.txt`, for example
`vpdpbusd` for the AVX-VNNI tiers, `vcvtneeph2ps` for AVX-NE-CONVERT, or `tdpbusd` for AMX.
Only the `EMIT_DYNAMIC_STATS` sections reflect executed instructions; the `$static-counts`
sections are disassembly of the binary and count instructions whether or not they ran, so
summing both double counts.

A zero signature count together with a green suite is exactly the blind spot this recipe
exists to catch: the gate did not open and only the fallback was tested.

## What this catches

- Dispatch gates that never open (a wrong CPUID bit, a misfiring build probe): compare
  signature counts between the tier's chip flag and a flag one tier below it.
- Kernels that no local machine or CI pool reaches: the suite passes identically whether or
  not the ISA kernel executed, distinguishable only by the histogram.
- Behavior of a new kernel across tiers: run the same filter under two chip flags and compare;
  the suites already assert against references, so per-tier runs double as byte-exactness
  checks where the tests are exact.

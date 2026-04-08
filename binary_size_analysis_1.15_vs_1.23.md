# Binary Size Analysis: ORT 1.15 vs 1.23 (ARM32 Speech SDK Extension)

**Files compared:**
- `libMicrosoft.CognitiveServices.Speech.extension.onnxruntime_1.15.so` — 13 MB
- `libMicrosoft.CognitiveServices.Speech.extension.onnxruntime_1.23.so` — 23 MB

**Architecture:** ARM32 hard-float ELF (`ld-linux-armhf.so.3`)

---

## Section-Level Breakdown

| Section | 1.15 | 1.23 | Delta |
|---------|-----:|-----:|------:|
| `.text` (code) | 7.9 MB | 15.2 MB | **+7.3 MB** |
| `.rodata` (read-only data) | 678 KB | 872 KB | +190 KB |
| `.ARM.extab` (exception tables) | 528 KB | 674 KB | +146 KB |
| `.ARM.exidx` (exception index) | 91 KB | 127 KB | +36 KB |
| `.rel.dyn` (relocations) | 107 KB | 169 KB | +61 KB |
| `.data.rel.ro` | 48 KB | 78 KB | +29 KB |
| `.debug_*` (new in 1.23) | 0 | ~40 KB | +40 KB |

The `.text` section (executable code) accounts for **~75% of the total ~10 MB growth**.

## Symbol-Level Analysis

| Metric | 1.15 | 1.23 |
|--------|-----:|-----:|
| Total text symbols | 16,240 | 22,912 |
| Symbols only in 1.15 | 3,894 | — |
| Symbols only in 1.23 | — | 10,566 |
| Shared symbols | 12,346 | 12,346 |

Total code size of the 10,566 **new-only** symbols: **7.2 MB** — this accounts for nearly all of the `.text` growth.

## Top-Level Namespace Growth

| Namespace | 1.15 | 1.23 | Delta |
|-----------|-----:|-----:|------:|
| `onnxruntime::` | 5.1 MB | 9.6 MB | **+4.3 MB** |
| `std::` (incl. `void std::`, `bool std::`) | 1.2 MB | 2.6 MB | **+1.4 MB** |
| `onnx::` | 904 KB | 1,455 KB | +550 KB |
| `void onnxruntime::` | 441 KB | 907 KB | +464 KB |
| `absl::` | 113 KB | 299 KB | +186 KB |
| `Eigen::` + `void Eigen::` | 240 KB | 544 KB | +304 KB |
| `OrtApis` | 76 KB | 188 KB | +112 KB |

## Root Causes (Ranked by Impact)

### 1. New ONNX Runtime Operators & Contrib Ops (~1.6 MB new code)

`onnxruntime::contrib` grew from **634 KB → 1,364 KB** (+730 KB). Major new operators:

| New Operator | Size |
|-------------|-----:|
| `GatherBlockQuantized` | 72 KB |
| `QMoECPU` | 54 KB |
| `SkipLayerNorm` | 35 KB |
| `MatMulNBits` | 27 KB |
| `RotaryEmbedding` | 16 KB |
| `MoE` | 14 KB |
| `SparseAttention` | 11 KB |
| `GroupQueryAttention` | 10 KB |
| `DecoderMaskedMultiHeadAttention` | 9 KB |

New kernel registrations: **961 new `KernelCreateInfo` / `BuildKernelCreateInfo`** entries totaling **437 KB**.

Transformer/LLM inference infrastructure (Whisper, BeamSearch, T5) was significantly expanded.

### 2. Static Linking of libstdc++ / libgcc (~1.4 MB)

| Dynamic Dependency | 1.15 | 1.23 |
|-------------------|:----:|:----:|
| `libstdc++.so.6` | Yes (shared) | **No (static)** |
| `libgcc_s.so.1` | Yes (shared) | **No (static)** |
| `librt.so.1` | No | Yes (new) |

1.23 statically links the C++ standard library and GCC runtime. This adds **5,454 new `std::` symbols** totaling **~1.4 MB** of template instantiations baked into the binary.

### 3. ONNX Opset & Schema Growth (~950 KB)

- 961 new `onnx::OpSchema` / `onnx::GetOpSchema` symbols (475 KB)
- `onnx::` namespace total grew 904 KB → 1,455 KB (+550 KB)
- Reflects support for newer ONNX opsets (likely opset 19–21+)

### 4. Abseil (absl) Library Upgrade (~710 KB)

- 1.15: older abseil version
- 1.23: `absl::lts_20250512`
- 1,422 new absl symbols, likely from broader adoption of abseil containers/utilities

### 5. ML Operators & Eigen Template Expansion (~700 KB)

- `onnxruntime::ml` grew 308 KB → 626 KB (+318 KB) — includes new `LabelEncoder_4`, tree ensemble updates
- New Eigen template instantiations: 185 symbols, ~400 KB (notably `Eigen::half` GEMV kernels, tensor shuffling)
- `onnxruntime::QDQ` (quantize/dequantize) grew 31 KB → 102 KB (+71 KB)

### 6. Sub-namespace Growth within `onnxruntime::`

| Sub-namespace | 1.15 | 1.23 | Delta |
|--------------|-----:|-----:|------:|
| `contrib` | 634 KB | 1,364 KB | +730 KB |
| `KernelCreateInfo` | 429 KB | 802 KB | +373 KB |
| `common` | 424 KB | 767 KB | +342 KB |
| `ml` | 308 KB | 626 KB | +318 KB |
| `BuildKernelCreateInfo` | 191 KB | 336 KB | +145 KB |
| `Graph` | 94 KB | 199 KB | +105 KB |
| `QDQ` | 31 KB | 102 KB | +71 KB |
| `InferenceSession` | 72 KB | 139 KB | +67 KB |
| `utils` | 44 KB | 101 KB | +57 KB |
| `Initializer` (new) | — | 48 KB | +48 KB |
| `optimizer_utils` (new) | — | 46 KB | +46 KB |

## CMake Build Configuration Analysis

Investigation of the ORT cmake build system (`cmake/`) to determine which configurations control the static linking and binary size growth.

### Static Linking of libstdc++ / libgcc: NOT controlled by ORT cmake

**Key finding:** The ORT cmake build system does **not** set `-static-libstdc++` or `-static-libgcc` anywhere.

Files thoroughly searched with no matches:
- `cmake/CMakeLists.txt` — main build configuration
- `cmake/onnxruntime.cmake` — shared library link configuration
- `cmake/adjust_global_compile_flags.cmake` — global compile/link flags
- `cmake/linux_arm32_crosscompile_toolchain.cmake` — ARM32 cross-compilation toolchain
- `tools/ci_build/build.py` — build script
- All CI pipeline YAML files in `tools/ci_build/github/`

The ARM32 toolchain file (`cmake/linux_arm32_crosscompile_toolchain.cmake`) only sets:
```cmake
SET(CMAKE_C_COMPILER arm-none-linux-gnueabihf-gcc)
SET(CMAKE_CXX_COMPILER arm-none-linux-gnueabihf-g++)
```
No static linking flags are added.

**Conclusion:** The `-static-libstdc++ -static-libgcc` flags must be injected by the **Speech SDK build system** (the consumer of the ORT library), not by ORT's own cmake. This is done externally, likely via `CMAKE_CXX_FLAGS`, `CMAKE_SHARED_LINKER_FLAGS`, or a custom toolchain file provided by the Speech SDK build. The dynamic dependency diff confirms this:
- 1.15 links `libstdc++.so.6` and `libgcc_s.so.1` **dynamically**
- 1.23 omits both — they are **statically linked** into the binary

### ORT CMake Options Relevant to Binary Size

The following options in `cmake/CMakeLists.txt` directly affect which code is compiled and linked:

#### Operator / Type Disabling Options (line 167–174, 186)

| CMake Option | Default | Effect |
|-------------|---------|--------|
| `onnxruntime_DISABLE_CONTRIB_OPS` | OFF | Disables all contrib ops (GQA, RotaryEmbedding, MatMulNBits, etc.) — **saves ~730 KB** |
| `onnxruntime_DISABLE_ML_OPS` | OFF | Disables traditional ML ops (TreeEnsemble, LabelEncoder, etc.) — **saves ~318 KB** |
| `onnxruntime_DISABLE_GENERATION_OPS` | OFF | Disables generation contrib ops (BeamSearch, WhisperBeamSearch, GreedySearch, Sampling) |
| `onnxruntime_DISABLE_SPARSE_TENSORS` | OFF | Disables sparse tensor data types |
| `onnxruntime_DISABLE_OPTIONAL_TYPE` | OFF | Disables optional type |
| `onnxruntime_DISABLE_FLOAT8_TYPES` | OFF | Disables float8 types (Float8E4M3FN, Float8E5M2, etc.) |
| `onnxruntime_DISABLE_FLOAT4_TYPES` | OFF | Disables float4 types |
| `onnxruntime_DISABLE_STRING_TYPE` | OFF | Disables string type support |
| `onnxruntime_REDUCED_OPS_BUILD` | OFF | Registers only a reduced set of kernels (via source file modification) |

#### Minimal Build Options (line 175–185)

| CMake Option | Default | Effect |
|-------------|---------|--------|
| `onnxruntime_MINIMAL_BUILD` | OFF | Excludes as much as possible; ORT flatbuffers models only (no ONNX format) |
| `onnxruntime_EXTENDED_MINIMAL_BUILD` | OFF | Minimal build + runtime kernel compilation |
| `onnxruntime_DISABLE_RTTI` | ON* | Disables RTTI (auto ON when not using Python/CUDA) |
| `onnxruntime_DISABLE_EXCEPTIONS` | ON* | Disables exception handling (requires MINIMAL_BUILD) |
| `onnxruntime_CLIENT_PACKAGE_BUILD` | OFF | Default settings more appropriate for client/on-device workloads |
| `onnxruntime_DISABLE_ABSEIL` | OFF | Redefs abseil containers to STL (still links abseil) |

\* These are `cmake_dependent_option` — their defaults depend on other options.

#### Linker-Level Size Optimizations (adjust_global_compile_flags.cmake)

| Setting | Location | Effect |
|---------|----------|--------|
| `-ffunction-sections -fdata-sections` | Line 14–16 | Enables per-function/data sections for GC |
| `LINKER:--gc-sections` | onnxruntime.cmake line 153 | Garbage-collects unreferenced sections |
| `LINKER:--no-undefined` | onnxruntime.cmake line 153 | Errors on unresolved symbols |
| Strip on minimal build | onnxruntime.cmake line 190–195 | Auto-strips on Android / minimal Unix builds |
| `onnxruntime_ENABLE_LTO` | CMakeLists.txt line 132 | Link-time optimization (interprocedural) |

#### Version Script (Symbol Visibility)

Both 1.15 and 1.23 export exactly 3 dynamic symbols (`OrtGetApiBase`, `OrtSessionOptionsAppendExecutionProvider_CPU`, version tag). The version script (`onnxruntime.lds`) generated by `tools/ci_build/gen_def.py` is used via `LINKER:--version-script` (onnxruntime.cmake line 153). This correctly hides internal symbols but cannot reduce binary size of the static code linked in.

### Key Finding: Where the External Build System Likely Injects Static Linking

Since ORT's cmake does **not** set static linking flags, the Speech SDK build must pass them via one of:
1. `CMAKE_CXX_FLAGS="-static-libstdc++ -static-libgcc"` on the cmake command line
2. `CMAKE_SHARED_LINKER_FLAGS="-static-libstdc++ -static-libgcc"` on the cmake command line
3. A custom toolchain file (replacing `cmake/linux_arm32_crosscompile_toolchain.cmake`) that sets these flags
4. Direct linker flags in the Speech SDK's CMakeLists.txt that wraps ORT

## Recommendations

If binary size reduction is a priority:

### Linking Configuration (External to ORT cmake)
1. **Dynamically link libstdc++** — remove `-static-libstdc++ -static-libgcc` from the external build flags. Saves **~1.4 MB** if the target environment has a compatible `libstdc++.so.6`

### ORT CMake Options to Set
2. **`-Donnxruntime_DISABLE_CONTRIB_OPS=ON`** — Disables all contrib ops. Saves **~730 KB** (GatherBlockQuantized, QMoECPU, SkipLayerNorm, MatMulNBits, RotaryEmbedding, MoE, SparseAttention, GQA, etc.). If only some contrib ops are needed, use `onnxruntime_REDUCED_OPS_BUILD` instead.
3. **`-Donnxruntime_DISABLE_ML_OPS=ON`** — Disables traditional ML ops. Saves **~318 KB** (TreeEnsemble, LabelEncoder, LinearClassifier, etc.)
4. **`-Donnxruntime_DISABLE_GENERATION_OPS=ON`** — Disables BeamSearch, WhisperBeamSearch, GreedySearch, Sampling ops specifically.
5. **`-Donnxruntime_DISABLE_FLOAT8_TYPES=ON -Donnxruntime_DISABLE_FLOAT4_TYPES=ON`** — Disables float8/float4 type support. Removes many type-dispatched template instantiations.
6. **`-Donnxruntime_REDUCED_OPS_BUILD=ON`** — Registers only a reduced set of kernels. Requires generating a reduced ops config file via `tools/ci_build/op_registration_utils.py`. This is the most effective way to eliminate unused kernel registrations (currently **961 new `KernelCreateInfo` entries** totaling 437 KB in 1.23).
7. **`-Donnxruntime_ENABLE_LTO=ON`** — Enable link-time optimization for cross-module dead code elimination.
8. **`-Donnxruntime_MINIMAL_BUILD=ON`** — The most aggressive option. Removes ONNX format parsing, graph optimizers, and most infrastructure. Only supports ORT flatbuffers format models. Enables auto-stripping on Unix and disables exceptions (with `onnxruntime_DISABLE_EXCEPTIONS`).
9. **`-Donnxruntime_CLIENT_PACKAGE_BUILD=ON`** — Enables default settings more appropriate for client/on-device workloads. 

### Compiler Flags (External to ORT cmake)
10. **Use `-Oz` optimization** — ARM32 builds can benefit significantly from size-optimized codegen
11. **Strip the binary** (`-s` linker flag) — 1.23 includes `.debug_*` sections (~40 KB, minor)

---

*Analysis performed on April 7, 2026 using `nm`, `objdump`, `readelf`, `size`, and cmake source inspection on the two ARM32 shared libraries.*

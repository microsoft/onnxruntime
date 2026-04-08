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

## Recommendations

If binary size reduction is a priority:

1. **Dynamically link libstdc++** — saves ~1.4 MB if the target environment has a compatible `libstdc++.so.6`
2. **Disable unused ops** via ORT's `--include_ops_by_config` build flag — many new LLM/transformer ops (BeamSearch, Whisper, MoE, SparseAttention) may be unnecessary for Speech SDK workloads
3. **Use `-Oz` optimization** — ARM32 builds can benefit significantly from size-optimized codegen
4. **Strip the binary** — 1.23 includes `.debug_*` sections (~40 KB, minor)
5. **Evaluate abseil usage** — the lts_20250512 upgrade pulled in substantial new code; consider if all abseil features are needed
6. **Reduce Eigen template bloat** — `Eigen::half` GEMV and tensor shuffling templates are heavily instantiated; consider limiting type support if `float16` is not used

---

*Analysis performed on April 7, 2026 using `nm`, `objdump`, `readelf`, and `size` on the two ARM32 shared libraries.*

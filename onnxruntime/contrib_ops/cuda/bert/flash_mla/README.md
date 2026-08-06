# FlashMLA (vendored)

Upstream: https://github.com/deepseek-ai/FlashMLA (MIT, Copyright (c) 2025 DeepSeek).
Imported via TensorRT-LLM `cpp/tensorrt_llm/kernels/flashMLA`, which carries the NVIDIA
Apache-2.0 modifications. Both license headers are preserved verbatim in every file.

These files are **unmodified** copies. Keep them that way: any ORT-side adaptation belongs in
`../flash_mla_latent.cu`, not in here, so the next upstream refresh stays a straight copy.

## What was imported

| file | why |
| --- | --- |
| `flash_mla.h` | public API: `Flash_fwd_mla_params`, `Mla_metadata_params`, the two entry points |
| `flash_fwd_mla_kernel.h` | the kernel itself (Hopper WGMMA + TMA) |
| `flash_fwd_mla_bf16_sm90.cu` | explicit instantiation `run_mha_fwd_splitkv_mla<bfloat16_t, bfloat16_t, 576>` |
| `flash_fwd_mla_metadata.cu` | `get_mla_metadata_func` — the tile scheduler / split planner |
| `softmax.h`, `utils.h`, `named_barrier.h`, `static_switch.h`, `fp8_transpose_v.h` | kernel-private helpers |

The fp16 and fp8 instantiation units were **not** imported. DeepSeek-V4-Flash decodes in bf16, and
each instantiation is expensive to compile; add them the same way if another model needs them.

## Build

The only external dependency is CUTLASS/CuTe headers, which ORT already vendors, so nothing new is
fetched. `cmake/onnxruntime_cuda_source_filters.cmake` routes `bert/flash_mla/*.cu` into the
`onnxruntime_providers_cuda_sm90_tma` object library, which compiles at `90a-real`.

`sm_90a`, not `sm_90`, is required: the kernel uses `wgmma` and the Hopper-only TMA paths. Because
that object library only exists when the build targets SM90+, the call site is compiled out via
`ORT_ENABLE_FLASH_MLA`; do not reference these symbols without that guard.

Warnings are suppressed per-source (`-w`) rather than by editing the files, since ORT builds with
`-Werror all-warnings` and this is third-party code.

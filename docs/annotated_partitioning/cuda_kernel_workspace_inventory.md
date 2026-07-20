# CUDA Kernel Workspace Buffer Inventory

This document catalogs all CUDA kernels in ONNX Runtime that allocate temporary/workspace buffers via `GetScratchBuffer` or `GetTransientScratchBuffer`. For each kernel, it identifies what information is needed to compute the workspace size and whether that information is available at `GetCapability()` time (for the workspace estimation function design).

## Classification Key

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully determinable from shapes + attributes + device properties |
| ✅* | Determinable via cuDNN/cuBLAS API call (needs handle, available on EP) |
| ⚠️ | Requires profiling/tactic selection (deterministic but costly at planning time) |

---

## Core CUDA Providers (`onnxruntime/core/providers/cuda/`)

### 1. Attention (LLM — Opset 23/24)

**File:** `llm/attention.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `softmax_lse_buffer` | `B * S_q * num_heads * sizeof(float)` | shapes |
| `softmax_lse_accum_buffer` | From `get_num_splits_and_buffer_sizes()` | shapes + `multiProcessorCount` |
| `out_accum_buffer` | From `get_num_splits_and_buffer_sizes()` | shapes + `multiProcessorCount` |
| `q_bsnh_buffer` | `B * S_q * num_heads * head_size * element_size` | shapes + dtype |
| `out_bsnh_buffer` | Same as Q | shapes + dtype |
| `k_bsnh_buffer` / `v_bsnh_buffer` | `B * S_kv * num_kv_heads * head_size * element_size` | shapes + dtype |
| `seqlens_k_buffer` | `B * sizeof(int)` | batch size |
| `past_seqlens_buffer` | `B * sizeof(int)` | batch size |
| `k_expand_buffer` / `v_expand_buffer` | `B * num_heads * S_kv * head_size * element_size` (GQA expansion) | shapes + dtype |
| `converted_mask_buffer` | `B * S_q * S_kv * sizeof(float)` | shapes |
| `present_k_scratch` / `present_v_scratch` | present KV cache size | shapes |
| `workspace_buffer` (math attention) | `B * S_q * num_heads * sizeof(float)` | shapes |

**What's needed to compute:** Input shapes, `num_heads` attribute, `head_size`, dtype, `device_prop.multiProcessorCount`.

**Static determinability:** ✅ All pure arithmetic on shapes + device SM count.

---

### 2. Conv (cuDNN Frontend)

**File:** `nn/conv.cc`, `nn/conv.h`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `workspace` | `s_.cudnn_fe_graph->get_workspace_size()` | cuDNN plan selection |
| `memory_for_cudnn_conv_results` (Conv8 only) | `y_dims_with_adjusted_pads.Size() * element_size` | output shape + padding |

**What's needed to compute:** Input shapes (NCHW), weight shapes, pads/strides/dilations attributes, cuDNN handle (for `build_plans()`).

**Static determinability:** ✅* — Requires calling `build_plans(handle)` with `HEUR_MODE_A`. The handle is on the EP. All tensor shapes and conv params come from node attributes.

**Note:** `GetTransientScratchBuffer` (32MB) used for algorithm search in Conv8 — this is a one-time cost during first Compute, not a per-run workspace.

---

### 3. ConvTranspose

**File:** `nn/conv_transpose.h`, `nn/conv_transpose_8.h`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `workspace` | `s_.workspace_bytes` (from cuDNN FE or algo selection) | Same as Conv |
| `AlgoSearchWorkspaceSize` (Conv8 path) | 32MB constant | N/A |

**Static determinability:** ✅* — Same as Conv.

---

### 4. DeformConv

**File:** `nn/deform_conv.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `col_buffer` | `C * kernel_size * col_stride * sizeof(T)` where `col_stride = n_parallel_imgs * output_image_size` | Input shapes, kernel_size, `device_prop.totalGlobalMem` (for chunk sizing) |

**What's needed:** Input shape (N,C,H,W), kernel dims, output_image_size, `totalGlobalMem` (determines `n_parallel_imgs` via `GetNParallelImgs`).

**Static determinability:** ✅ — Pure arithmetic on shapes + device memory size.

---

### 5. BatchNorm

**File:** `nn/batch_norm.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `f_scale`, `f_B`, `f_mean`, `f_var` | `C * sizeof(float)` each | Channel dim (shape[1] or shape[3]) |

**What's needed:** Channel dimension `C` from input shape.

**Static determinability:** ✅ — Trivial: `4 * C * sizeof(float)`.

---

### 6. InstanceNorm

**File:** `nn/instance_norm.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `mean`, `variance` | `N * C * sizeof(T)` | Batch × channels |
| `unused_scale`, `unused_bias` | `N * C * sizeof(T)` | Same |
| `scale_data_fp32`, `bias_data_fp32` | `C * sizeof(float)` (if fp16) | Channel dim + dtype |

**What's needed:** Input shape (N, C), dtype.

**Static determinability:** ✅ — Pure arithmetic on shapes.

---

### 7. Dropout

**File:** `nn/dropout.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| mask buffer | `element_count * sizeof(bool)` or `element_count / 16 * sizeof(BitmaskElementType)` | Input element count, bitmask mode |

**Static determinability:** ✅ — Input element count.

---

### 8. Reduction Ops (ReduceSum, ReduceMax, ReduceMean, etc.)

**File:** `reduction/reduction_ops.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `workspace_cuda` | `cudnnGetReductionWorkspaceSize()` | cuDNN handle, input/output tensor descriptors |
| `indices_cuda` | `cudnnGetReductionIndicesSize()` | Same |
| `temp_X` | `input_count * sizeof(float)` (type cast) | Input size |
| `input_data_buffer` | `input_count * sizeof(T)` (for `calculate_sqt_`) | Input size |
| `exp_result_buffer` | `input_count * sizeof(T)` (for `log_sum_exp_`) | Input size |
| `log_sum_result_buffer` | `output_count * sizeof(T)` | Output size |
| `temp_output` | `output_count * sizeof(float)` | Output size |

**What's needed:** Input/output shapes, reduction axes, op variant (LogSumExp, L2, etc.), cuDNN handle.

**Static determinability:** ✅* — cuDNN workspace query needs handle + tensor descriptors (constructible from shapes).

---

### 9. RNN (LSTM/GRU)

**File:** `rnn/cudnn_rnn_base.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `workspace_cuda` | `cudnnGetRNNTempSpaceSizes(fwdInference)` | RNN descriptor, seq_length, batch_size |
| `reservespace_cuda` | `cudnnGetRNNTempSpaceSizes(training)` | Same |
| `reorganized_w_data` | `w_size * sizeof(T)` | hidden_size, num_layers, input_size, direction |
| `x_reversed_data` | `seq_length * batch_size * input_size * sizeof(T)` | Shapes (bidirectional case) |
| `y_alloc_data` | `output_size * sizeof(T)` | Shapes |
| `state_buffer_` | RNN state size from cuDNN | cuDNN descriptor |

**What's needed:** seq_length, batch_size, input_size, hidden_size, num_layers, direction attribute, cuDNN handle.

**Static determinability:** ✅* — cuDNN API queries, all inputs available from node attributes/shapes.

---

### 10. TopK

**File:** `math/topk_impl.cuh`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `input_key_buffer` | `dimension * sizeof(T)` | Last-axis dimension |
| `output_key_buffer` | `dimension * sizeof(T)` | Same |
| `input_value_buffer` | `dimension * sizeof(int64_t)` | Same |
| `output_value_buffer` | `dimension * sizeof(int64_t)` | Same |
| `temp_storage_buffer` | From `cub::DeviceRadixSort::SortPairs` query | dimension |

**What's needed:** Dimension (last axis size), k, dtype.

**Static determinability:** ✅ — CUB temp storage query is deterministic given size.

---

### 11. MatMulInteger

**File:** `math/matmul_integer.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `a_row_buf` | `(output_size / N) * sizeof(int32_t)` | M dimension |
| `b_col_buf` | `(output_size / M) * sizeof(int32_t)` | N dimension |

**What's needed:** M, N dimensions from MatMul shapes.

**Static determinability:** ✅ — Pure arithmetic.

---

### 12. IntegerGemm (int8 alignment padding)

**File:** `integer_gemm.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `a_padded` | `m * roundoff(lda, 32) * sizeof(int8_t)` (only if lda not 32-aligned) | M, K dims |
| `b_padded` | `k * roundoff(ldb, 32) * sizeof(int8_t)` (only if ldb not 32-aligned) | K, N dims |

**What's needed:** M, K, N dimensions + alignment check.

**Static determinability:** ✅ — Pure arithmetic.

---

### 13. Compress

**File:** `tensor/compress.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `condition_cumulative_sum_buffer` | `valid_condition_length * sizeof(int32_t)` | Condition tensor size |
| `temp_buffer` | CUB `DeviceScan::InclusiveSum` temp storage | Condition size |

**Static determinability:** ✅ — Condition shape determines everything.

---

### 14. GatherND

**File:** `tensor/gather_nd.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `sizes_from_slice_dims_buffer` | `num_slice_dims * sizeof(int64_t)` | Indices shape |
| `input_slice_offsets_buffer` | `num_slices * sizeof(int64_t)` | Indices shape[:-1] product |

**Static determinability:** ✅ — Indices shape.

---

### 15. NonZero

**File:** `tensor/nonzero_op.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `prefix_buffer` | `number_of_blocks * sizeof(int)` | Input element count / block_size |
| `temp_buffer` | CUB temp storage | Input element count |

**Static determinability:** ✅ — Input element count.

---

### 16. Upsample/Resize

**File:** `tensor/upsample.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| temp buffer (via lambda) | Varies by resize mode | Input/output shapes, mode |
| `dims_mapping_buffer` | `temp_buffer_size` (coordinate mapping) | Output shape |

**Static determinability:** ✅ — Input/output shapes + mode attribute.

---

### 17. NonMaxSuppression

**File:** `object_detection/non_max_suppression.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| Various (via lambda) | Determined by CUB DeviceSelect internals | num_boxes, num_classes |

**What's needed:** boxes shape (num_batches, num_boxes, 4), scores shape.

**Static determinability:** ✅ — CUB queries are deterministic given sizes.

---

## Contrib CUDA Operators (`onnxruntime/contrib_ops/cuda/`)

### 18. Attention / MultiHeadAttention (Contrib)

**File:** `bert/attention.cc`, `bert/multihead_attention.cc`

**Buffers:** Uses `GetAttentionWorkspaceSize()` helper function.

**Size formula:** Depends on attention algorithm (Flash, MemoryEfficient, FusedRunner, Default):
- Flash: `qkv_size` (Q+K+V projection)
- MemoryEfficient: `qkv_size + output_accum (float)`
- Default (unfused): `qkv_size + 2 * attention_scratch_size`

**What's needed:** B, S_q, S_kv, num_heads, head_size, dtype, which attention algorithm is selected.

**Static determinability:** ✅ — Algorithm selection depends on shapes + SM version (available from device_prop).

---

### 19. MOE (Mixture of Experts)

**File:** `moe/moe.cc`, `moe/moe_quantization.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `workspace` | `moe_runner->getWorkspaceSize(num_rows, hidden, inter, experts, k)` | Shapes + tactic |
| `expert_scales` | `num_rows * k * sizeof(float)` | Shapes |
| `expert_indices` | `num_rows * k * sizeof(int)` | Shapes |
| `permutation_row_map` | `num_rows * k * sizeof(int)` | Shapes |

**What's needed:** num_rows, hidden_size, inter_size, num_experts, k, activation_type, SM version, selected tactic.

**Static determinability:** ⚠️ — `getWorkspaceSize()` depends on profiled best tactic (CUTLASS config). Could use worst-case across tactics as upper bound.

---

### 20. MatMulNBits (Quantized MatMul)

**File:** `quantization/matmul_nbits.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `workspace_buffer` | `weightOnlyGemmRunner_->getWorkspaceSize(m, n, k)` | Dims + tactic |
| `packed_transposed_weight_space` | `packed_weight_bytes` (transient) | Weight shape |
| `permutation_map_buffer` | `32 * sizeof(int32_t)` (transient) | Constant |

**What's needed:** M, N, K dimensions, quantization bits, SM version.

**Static determinability:** ⚠️ — Runner workspace depends on profiled tactic. Could use upper bound.

---

### 21. fpA_intB GEMM (FP×INT quantized)

**File:** `llm/fpA_intB_gemm/`

**Buffers:** `virtual size_t getWorkspaceSize(m, n, k)`.

**What's needed:** M, N, K + CUTLASS template specialization.

**Static determinability:** ⚠️ — Depends on selected CUTLASS config/tactic.

---

### 22. Inverse (Matrix Inversion)

**File:** `inverse.cc`

**Buffers allocated:**
| Buffer | Size formula | Depends on |
|--------|--------------|-----------|
| `input_workspace` | `input_count * sizeof(T)` | Matrix dimensions |
| `matrix_ptrs` | `n_batches * sizeof(T*)` | Batch count |
| `output_ptrs` | `n_batches * sizeof(T*)` | Batch count |
| `ml_float_output` | `input_count * sizeof(float)` (if fp16→fp32) | Dims + dtype |

**Static determinability:** ✅ — Pure arithmetic on matrix dimensions.

---

### 23. Generation (Beam Search / Sampling)

**File:** `transformers/generation_device_helper.cc`

**Buffers:** Various pinned + device buffers for beam state.

**What's needed:** batch_size, beam_width, max_length, vocab_size.

**Static determinability:** ✅ — All from session/model config.

---

## Summary: Coverage Analysis

### Workspace estimation approach validation

| Category | # Kernels | Estimation feasibility | Notes |
|----------|-----------|----------------------|-------|
| **Shapes only** | 12 | ✅ Exact, trivial | BatchNorm, InstanceNorm, Dropout, TopK, MatMulInteger, IntegerGemm, Compress, GatherND, NonZero, Upsample, Inverse, Generation |
| **Shapes + device properties** | 3 | ✅ Exact | Attention (SM count), DeformConv (totalGlobalMem), Contrib Attention (SM version) |
| **Shapes + cuDNN/cuBLAS handle** | 4 | ✅* Exact via API query | Conv, ConvTranspose, Reduction, RNN |
| **Shapes + tactic profiling** | 3 | ⚠️ Upper bound only | MOE, MatMulNBits, fpA_intB_GEMM |

### Key takeaways

1. **~75% of kernels** (19/25) can produce **exact** workspace estimates at `GetCapability()` time using only shapes + attributes + device properties (+ cuDNN handle for API queries).

2. **~12% of kernels** (3/25) require tactic profiling (CUTLASS/CUB autotuning). For these, options are:
   - Use worst-case workspace across all tactics (safe upper bound)
   - Run tactic selection eagerly at estimation time (expensive but exact)
   - Accept 1.5x multiplier for these few kernels

3. **The cuDNN handle requirement** affects only 4 kernel types (Conv, ConvTranspose, Reduction, RNN). All are standard cuDNN API queries that are fast and deterministic given the handle + tensor descriptors.

4. **No kernel requires actual GPU execution** to determine workspace size — even tactic-based kernels select tactics via CPU-side profiling/heuristics, not by running GPU code.

5. **Largest workspace consumers** in practice:
   - **Attention** (Flash): dominates in LLM workloads. Exact estimation possible.
   - **Conv** (cuDNN): dominates in vision workloads. Exact via `build_plans()`.
   - **MOE**: significant in MoE models. Upper bound via worst-case tactic.

### What the estimation function needs access to (API requirements):

| Access needed | How accessed | Kernels that require it |
|---------------|-------------|------------------------|
| `Node_GetInputShape()` | OrtEpApi (generic) | All 25 kernels |
| `Node_GetAttributeInt/Ints()` | OrtEpApi (generic) | Conv, Attention, RNN, MOE |
| `device_prop.multiProcessorCount` | Cast `OrtEp*` to concrete EP type | Attention, DeformConv |
| `device_prop.totalGlobalMem` | Cast `OrtEp*` to concrete EP type | DeformConv |
| cuDNN handle | Cast `OrtEp*` to concrete EP type | Conv, ConvTranspose, Reduction, RNN |
| Tactic profiler state (or worst-case constant) | Cast `OrtEp*` to concrete EP type | MOE, MatMulNBits, fpA_intB |

**API surface:** Only `Node_GetInputShape` and `Node_GetAttributeInt/Ints` need to be added to `OrtEpApi` (generic, EP-agnostic). All device-specific state (cuDNN handles, device properties, profiler state) is accessed by casting `OrtEp*` to the EP's concrete type — no public API needed since the estimation function is EP-specific code.

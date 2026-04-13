# TurboQuant KV Cache Compression for WebGPU Flash Attention

## Overview

TurboQuant is a KV cache compression technique that:
1. **Rotates** K and V vectors using a Hadamard matrix after RoPE (for K) / after projection (for V)
2. **Normalizes** rotated vectors to unit sphere, storing the per-token norm as a scale
3. **Quantizes** normalized coordinates to 4-bit using an MSE-optimal codebook (16 centroids)
4. **Dequantizes** on-the-fly during attention computation by looking up centroids and rescaling by norm

This yields ~3.75× KV cache memory savings (int4 + scale vs fp16 per element) with minimal quality loss, validated by the reference PyTorch implementation in `C:\ML\chat_phi4.py`.

---

## Reference Implementation Analysis (`chat_phi4.py`)

### Hadamard Matrix Generation
```python
def generate_hadamard_matrix(dim):
    """Sylvester construction. dim must be power of 2."""
    H = torch.tensor([[1.0]], dtype=torch.float64)
    while H.shape[0] < dim:
        H = torch.cat([torch.cat([H, H], dim=1),
                        torch.cat([H, -H], dim=1)], dim=0)
    return H / (dim ** 0.5)  # normalize so H @ H.T = I
```
- For Phi-4: head_dim=128 → H is 128×128, normalized so H·Hᵀ = I (orthogonal)
- Same H used for all layers, one per KV head (in reference: same H for all KV heads)
- H is its own inverse: H⁻¹ = Hᵀ = H (since symmetric and orthogonal)

### Rotation Application
```python
# After RoPE, before KV cache storage:
query_states  = torch.einsum("bhsd,hde->bhse", query_states, R_q)   # R_q = R_kv repeated for GQA
key_states    = torch.einsum("bhsd,hde->bhse", key_states, R_kv)
value_states  = torch.einsum("bhsd,hde->bhse", value_states, R_kv)

# After attention output, before o_proj:
attn_output = torch.einsum("bshd,hde->bshe", attn_output, R_kv_T)  # R_kv_T = R_kv.transpose(-1,-2)
```
- **Key insight**: Since H = Hᵀ and is its own inverse, the inverse rotation is the same matrix
- For GQA: R_q = R_kv.repeat_interleave(heads_per_group, dim=0) — each Q head group uses corresponding KV head's rotation

### 4-Bit Pseudo-Quantization (TurboQuant Codebook)
```python
TQ_CENTROIDS_4BIT = [  # 16 symmetric centroids (MSE-optimal for d=128 unit sphere)
    -0.2377, -0.1809, -0.1419, -0.1104, -0.0829, -0.0578, -0.0342, -0.0113,
     0.0113,  0.0342,  0.0578,  0.0829,  0.1104,  0.1419,  0.1809,  0.2377,
]
TQ_BOUNDARIES_4BIT = [  # 17 decision boundaries
    -1.0000, -0.2093, -0.1614, -0.1261, -0.0966, -0.0704, -0.0460, -0.0227,
     0.0000,  0.0227,  0.0460,  0.0704,  0.0966,  0.1261,  0.1614,  0.2093, 1.0000,
]

def tq_pseudo_quantize(x):
    norms = x.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # per-token norm
    x_unit = x / norms                                      # normalize to unit sphere
    indices = torch.searchsorted(boundaries, x_unit) - 1    # find bucket (4-bit index)
    indices = indices.clamp(0, 15)
    x_quant = centroids[indices]                             # snap to centroid
    return x_quant * norms                                   # rescale
```

### Storage Layout (Per Token Per KV Head)
- **128 elements** → each quantized to 4-bit index (0-15) = 64 bytes of int4 data
- **1 fp16 norm/scale** = 2 bytes
- **Total**: 66 bytes per token per head, vs 256 bytes for fp16 → **3.88× savings**

---

## Architecture

### Configuration Flow
```
genai_config.json
  → "provider_options": [{"webgpu": {"turboQuant": "1"}}]
  → GenAI Config::ProviderOptions.options (generic key-value pairs)
  → ORT Session: "ep.webgpuexecutionprovider.turboQuant" = "1"
  → WebGpuExecutionProviderConfig.turbo_quant = true
  → ComputeContext.TurboQuant() → returns true
  → GQA kernel reads it, sets WebgpuAttentionParameters.turbo_quant_ = true
  → Flash attention shaders apply rotation/quantization
```

### Hadamard Rotation Strategy: Fast Walsh-Hadamard Transform (FWHT)

Instead of storing and uploading a Hadamard matrix, we use the **Fast Walsh-Hadamard Transform** (FWHT) computed entirely in GPU shared memory:

- **No matrix storage**: The FWHT computes H·v via O(n log n) butterfly operations in-place
- **Self-inverse**: Applying the same transform twice yields the identity (H·H = I when normalized)
- **Implementation**: Each workgroup handles one vector (head_size elements). Elements are loaded into `var<workgroup>` shared memory, then log2(head_size) butterfly stages perform paired add/subtract, followed by normalization by 1/√head_size
- **Performance**: For head_size=128, only 7 butterfly stages (896 add/sub ops) vs 16,384 multiply-adds for a matrix-vector product
- **Files**: `turbo_quant_hadamard.h/cc` — `TurboQuantRotateProgram` class with WGSL shader generation

### KV Cache Allocation (GenAI Side)
**Question from user**: Is the smaller cache managed by just changing GenAI's allocation formula?

**Answer**: Yes. With `past_present_share_buffer=true`:
- Current: `shape = [batch, kv_heads, max_length, head_size]` with type fp16
- TurboQuant: `shape = [batch, kv_heads, max_length, compressed_head_dim]` with type uint32

Where `compressed_head_dim = ceil(head_size / 8) + 1`:
- Each uint32 holds 8 int4 values
- 128 int4 values → 16 uint32 elements for data
- +1 uint32 for the fp16 scale (packed into uint32)
- Total: `head_size/8 + 1` uint32 values (e.g., 17 for head_size=128 → 68 bytes vs 256 bytes → **3.76× savings**)

Alternative (simpler for Phase 1): Keep fp16 type, pack int4 into fp16 carrier:
- 4 int4 values per fp16 element → `head_size/4 = 32` fp16 for data
- +1 fp16 for scale
- Total: 33 fp16 = 66 bytes → **3.88× savings**

**RewindTo/Update**: No changes needed — stride arithmetic works on the compressed last dimension.

---

## Phased Implementation Plan

### Phase 0: First Checkpoint — V Rotation + Inverse Rotation
**Goal**: Prove out the full plumbing from config to shader, verify correctness with a simple orthogonal transform (Hadamard rotation of V only, inverse rotation of attention output). No quantization.

#### Phase 0a: GenAI Config Plumbing
**Files (GenAI: C:\onnxruntime-genai)**:
- `src/config.h` — No changes (generic provider_options already support arbitrary key-value pairs)
- `src/models/kv_cache.cpp` — No changes for Phase 0 (cache dimensions unchanged)

**Files (ORT: C:\onnxruntime)**:
1. `onnxruntime/core/providers/webgpu/webgpu_provider_options.h`
   - Add: `constexpr const char* kTurboQuant = "ep.webgpuexecutionprovider.turboQuant";`

2. `onnxruntime/core/providers/webgpu/webgpu_execution_provider.h`
   - Add to `WebGpuExecutionProviderConfig`: `bool turbo_quant{false};`
   - Add to `WebGpuExecutionProvider`: `bool TurboQuant() const { return turbo_quant_; }`
   - Add member: `bool turbo_quant_ = false;`

3. `onnxruntime/core/providers/webgpu/webgpu_provider_factory.cc`
   - In `ParseEpConfig()`: Parse `kTurboQuant` option → set `webgpu_ep_config.turbo_quant = true`

4. `onnxruntime/core/providers/webgpu/compute_context.h`
   - Add to `ComputeContextBase`: `inline bool TurboQuant() const { return ep_.TurboQuant(); }`

5. `onnxruntime/contrib_ops/webgpu/bert/attention_common.h`
   - Add to `WebgpuAttentionParameters`: `bool turbo_quant_ = false;`

6. `onnxruntime/contrib_ops/webgpu/bert/group_query_attention.cc`
   - In `ComputeInternal()`: `parameters.turbo_quant_ = context.TurboQuant();`

#### Phase 0b: FWHT Rotation Shader
**Files (ORT: C:\onnxruntime)**:
1. NEW: `onnxruntime/contrib_ops/webgpu/bert/turbo_quant_hadamard.h`
   ```cpp
   class TurboQuantRotateProgram final : public Program<TurboQuantRotateProgram> {
     // FWHT shader: applies normalized Hadamard transform in-place via shared memory
     // Dispatched as one workgroup per (batch × num_heads × sequence_length)
     // Workgroup size = head_size (must be power of 2)
     // Uniforms: sequence_length, present_sequence_length, start_token, n_reps
   };
   Status ApplyTurboQuantRotation(...);        // BNSH tensor in-place
   Status ApplyTurboQuantInverseRotation(...); // BSNH output in-place (flat access)
   ```

2. NEW: `onnxruntime/contrib_ops/webgpu/bert/turbo_quant_hadamard.cc`
   - `GenerateShaderCode()`: Emits WGSL with butterfly FWHT in `var<workgroup>` shared memory
   - Uses output-only binding (`storage_buffer_rw`) for in-place read+write
   - Normalization by 1/√head_size baked into shader constants
   - `ApplyTurboQuantRotation()`: For BNSH tensors, respects `start_token` for static cache
   - `ApplyTurboQuantInverseRotation()`: For BSNH output, uses flat access (num_heads=1 trick)

#### Phase 0c: V Rotation Wired into Flash Attention
**Approach**: Separate rotation shaders dispatched before/after flash attention (not fused into existing shaders).

**Files (ORT: C:\onnxruntime)**:
1. `onnxruntime/contrib_ops/webgpu/bert/flash_attention.cc`
   - `#include "turbo_quant_hadamard.h"`
   - After CopyKVCache/fused split: `ApplyTurboQuantRotation()` on `present_value` for newly written tokens
     - Respects `start_token` for `past_present_share_buffer` (only rotates new tokens)
   - After prompt path flash attention: `ApplyTurboQuantInverseRotation()` on output (BSNH)
   - After decode path VxReduce: `ApplyTurboQuantInverseRotation()` on output (BSNH)
---

### Phase 1: QK Rotation — COMPLETE ✓
**Goal**: Extend rotation to Q and K. After this phase, Q, K, V are all Hadamard-rotated.

**Changes**:
- Rotate K before writing to `present_key` (reuse `ApplyTurboQuantRotation` from Phase 0)
- Rotate Q before computing attention scores (uses `ApplyTurboQuantInverseRotation` for BSNH layout — same FWHT math, different tensor traversal)
- **Key correctness property**: If Q' = HQ and K' = HK, then Q'·K'ᵀ = (HQ)·(HK)ᵀ = Q·Hᵀ·H·Kᵀ = Q·Kᵀ (since HᵀH = I). So QK scores are invariant — no inverse rotation needed for QK!
- The inverse rotation is only needed on the attention output (softmax(QK')·V' needs inverse rotation)

**Implementation** (5 call sites in `flash_attention.cc`, all guarded by `if (parameters.turbo_quant_)`):
1. `ApplyTurboQuantRotation()` on `present_key` — newly written tokens, BNSH
2. `ApplyTurboQuantRotation()` on `present_value` — newly written tokens, BNSH
3. `ApplyTurboQuantInverseRotation()` on `query_output` — Q rotation, BSNH
4. `ApplyTurboQuantInverseRotation()` on output — after prompt flash attention, BSNH
5. `ApplyTurboQuantInverseRotation()` on output — after decode VxReduce, BSNH

**Status**: Code complete, ORT build successful, GqaTest verified — all tests pass.

---

### Phase 2: Pseudo-Quantization (fp16 indices — no bit-packing)
**Goal**: After rotation, quantize each element of K and V to its nearest TurboQuant centroid index (0–15). Store the index **as fp16** in the existing KV cache layout (same shape, same type). The cache size does NOT change yet — this phase validates the quantization/dequantization math end-to-end before introducing a new packed format.

**Why fp16 indices first?**
- No KV cache shape or type changes → no GenAI changes needed
- Flash attention shaders can dequantize by treating each fp16 as an integer index: `centroid[u32(value)]`
- Easy to validate against reference PyTorch `tq_pseudo_quantize()` output
- Isolates quantization quality impact from packing complexity

**Codebook (compile-time constants in WGSL)**:
```wgsl
const TQ_CENTROIDS = array<f16, 16>(
    -0.2377h, -0.1809h, -0.1419h, -0.1104h, -0.0829h, -0.0578h, -0.0342h, -0.0113h,
     0.0113h,  0.0342h,  0.0578h,  0.0829h, 0.1104h,  0.1419h,  0.1809h,  0.2377h,
);
const TQ_BOUNDARIES = array<f16, 17>(
    -1.0h, -0.2093h, -0.1614h, -0.1261h, -0.0966h, -0.0704h, -0.0460h, -0.0227h,
     0.0h,  0.0227h,  0.0460h,  0.0704h,  0.0966h,  0.1261h,  0.1614h,  0.2093h, 1.0h,
);
```

**Quantize Shader** (runs after rotation, before cache write):
```
1. Load rotated K/V vector (fp16, head_size elements)
2. Compute L2 norm: norm = sqrt(sum(x_i^2))  — stored in a dedicated position or uniform
3. Normalize: x_unit_i = x_i / norm
4. For each element: searchsorted on TQ_BOUNDARIES → 4-bit index (0–15)
5. Store the index as fp16: present_value[offset + i] = f16(index)
6. Store norm separately (e.g., in the last element, or a parallel norm buffer)
```

**KV Cache Format (Phase 2 — as implemented)**:
```
Per token per KV head: head_size fp16 values (128 for Phi-4)
  [0..head_size-3]: centroid index stored as fp16 (values 0.0h–15.0h)  — 126 elements
  [head_size-2..head_size-1]: f32 L2 norm bitcast into 2 fp16 slots
    Written:  let norm_halves = bitcast<vec2<f16>>(f32(norm));
              data[offset + head_size - 2] = norm_halves.x;
              data[offset + head_size - 1] = norm_halves.y;
    Read:     let norm = bitcast<f32>(vec2<f16>(data[z_offset], data[w_offset]));
  — Same shape [batch, kv_heads, max_seq, 128] fp16 as baseline
  — No memory savings yet, but quantization math is validated
  — f32 norm precision avoids fp16 overflow during large-vector norm accumulation
```

**Dequantize** (inside flash attention loadk/loadv, as implemented):
```wgsl
const TQ_CENTROIDS = array<f16, 16>(...);  // compile-time constant

fn tq_read_norm(base_offset: u32) -> f32 {
  let z = present_key[base_offset + uniforms.head_size_vec - 1u].z;
  let w = present_key[base_offset + uniforms.head_size_vec - 1u].w;
  return bitcast<f32>(vec2<f16>(z, w));
}

fn tq_dequant_vec4(v: vec4<f16>, norm: f32) -> vec4<f16> {
  return vec4<f16>(
    TQ_CENTROIDS[u32(v.x)], TQ_CENTROIDS[u32(v.y)],
    TQ_CENTROIDS[u32(v.z)], TQ_CENTROIDS[u32(v.w)]
  ) * f16(norm);
}
```

**Files (as implemented)**:
- `turbo_quant_hadamard.h/cc` — New `TurboQuantRotateQuantizeProgram`: fused FWHT rotate → f32 norm → searchsorted → store fp16 indices + bitcast norm
- `flash_attention.wgsl.template` — `#param turbo_quant`; dequantize in loadk/loadv with norm preload
- `flash_attention_decode_qkt.wgsl.template` — `#param turbo_quant`; dequantize K with shared memory norms
- `flash_attention_decode_split_vx.wgsl.template` — `#param turbo_quant`; dequantize V with shared memory norms

---

### Phase 3: Bit-Packing to uint4
**Goal**: Pack the fp16 centroid indices from Phase 2 into actual 4-bit integers. 8 indices per uint32, plus a fp16 norm. This changes the KV cache shape and type, yielding ~3.8× memory savings.

**KV Cache Format (Phase 3 — compressed)**:
```
Per token per KV head: head_size/8 + 1 uint32 values
  [0]: fp16 norm packed in uint32 (lower 16 bits)
  [1..head_size/8]: head_size int4 indices packed as 8 per uint32
Total: (head_size/8 + 1) × 4 bytes vs head_size × 2 bytes
  e.g., head_size=128: 17 × 4 = 68 bytes vs 256 bytes → 3.76× savings
```

**Changes from Phase 2**:
- Quantize shader: after searchsorted, pack 8 indices into one uint32 instead of storing as fp16
- Dequantize shader: unpack uint32 → 8 indices → centroid lookup (replaces `u32(kv_cache[i])` with bitwise extract)
- KV cache shape: `[batch, kv_heads, max_seq, head_size/8 + 1]` with type uint32

#### GenAI KV Cache Dimension Change
**Files (GenAI: C:\onnxruntime-genai)**:
1. `src/models/kv_cache.h`
   - Add `bool turbo_quant_{false};` and `int compressed_kv_last_dim_{0};` to `DefaultKeyValueCache`

2. `src/models/kv_cache.cpp` (DefaultKeyValueCache constructor)
   - Detect turbo_quant from provider options
   - Compute `compressed_kv_last_dim_ = head_size / 8 + 1` (17 for head_size=128)
   - Change `shape_[3] = compressed_kv_last_dim_` instead of `head_size`
   - Change `type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32` (override model's fp16)
   - **RewindTo**: Works without changes (copies `compressed_kv_last_dim_` per head per token)
   - **Update**: Works without changes for shared buffer mode (no-op)

3. `src/models/kv_cache.cpp` (RewindPastTensorsTo)
   - Need to add `uint32_t` template specialization (currently only `float` and `Float16_t`)
   - Or use `ByteWrapTensor` for type-agnostic byte copies

#### GQA Kernel Shape Changes
**Files (ORT: C:\onnxruntime)**:
1. `onnxruntime/contrib_ops/webgpu/bert/group_query_attention.cc`
   - When turbo_quant: `present_dims[3] = compressed_kv_last_dim` (17 instead of head_size)
   - Must match GenAI's allocation

2. Input validation (`multihead_attention_helper.h`):
   - When turbo_quant: skip past_key last_dim == head_size check

#### TMAC-Style Lookup Table MatMul for QK
Since quantized K values are 4-bit indices into 16 centroids, QK^T can be computed as:

```
For each Q vector (fp16, dim=128):
  Pre-compute: dot_table[c] = sum over groups ( Q_group · centroid[c] )
  Actually: for int4 values, the "matmul" becomes table lookups:

  score = norm_k * sum_i( centroid[k_index[i]] * q[i] )
```

This is equivalent to a standard dot product but with dequantized K. However, TMAC can optimize this:

```
For each group of 4 Q elements and corresponding 4 int4 K indices:
  Precompute (once per Q): LUT[idx] = q[0]*C[idx0] + q[1]*C[idx1] + q[2]*C[idx2] + q[3]*C[idx3]
  But with 4-bit and 4 elements: 16^4 = 65536 entries — too large.
```

**Simpler approach**: Since centroids are symmetric (C[i] = -C[15-i]):
- Precompute per Q element: `partial[i] = q[i] * centroid[k_index[i]]` via 16-entry LUT
- Sum all partials and multiply by norm_k
- Total: 16 * head_size_vec multiplies to build LUT, head_size lookups, head_size additions

**WGSL Implementation**:
```wgsl
// Per workgroup: precompute LUT for this Q vector
var<workgroup> q_centroid_lut : array<array<f16, 16>, head_size>;
// q_centroid_lut[i][c] = q[i] * centroids[c]
// Then: qk_score = norm_k * sum_i(q_centroid_lut[i][ k_index[i] ])
```

This is a future optimization. Phase 2's simple dequantize-then-dot is the baseline.

---

## Key Source Code References

### ORT WebGPU Provider Options Chain
```
webgpu_provider_options.h         — Option key constants
  ↓
webgpu_provider_factory.cc        — ParseEpConfig() parses config_options → WebGpuExecutionProviderConfig
  ↓
webgpu_execution_provider.h       — Config struct + EP accessor methods
  ↓
compute_context.h                 — ComputeContextBase exposes EP accessors to kernels
  ↓
group_query_attention.cc          — GQA kernel reads context.TurboQuant()
  ↓
flash_attention.cc                — Flash attention programs receive turbo_quant flag
  ↓
*.wgsl.template                   — WGSL shaders conditioned on #param turbo_quant
```

### GenAI KV Cache Chain
```
genai_config.json                 — "provider_options": [{"webgpu": {"turboQuant": "1"}}]
  ↓
config.cpp                        — Parsed as ProviderOptions.options = [("turboQuant", "1")]
  ↓
webgpu/session_options.cpp        — Forwarded to ORT session via AppendExecutionProviderV2
  ↓
models/kv_cache.cpp               — DefaultKeyValueCache constructor:
                                     shape_ = [batch, kv_heads, max_len, head_size]
                                     type_ = model.session_info_.GetInputDataType(...)
                                     OrtValue::CreateTensor(Allocator(), shape_, type_)
  ↓
                                   — RewindTo(): copies tensors with stride = shape_[3]
                                   — Update(): for shared buffer mode → no-op
```

### Flash Attention Data Flow (Phi-4 GQA with packed QKV + RoPE + static KV cache)
```
GQA::ComputeInternal()
  ├─ parameters.is_packed_qkv_ && do_rotary_ && flash_attention && share_buffer
  │   → ApplyFlashAttention(packed_qkv, nullptr, nullptr, ...)
  │       ├─ RunSplitPackedQKVWithRotaryEmbeddingAndCopyKV()
  │       │   Splits packed QKV → Q, K, V
  │       │   Applies RoPE to Q and K
  │       │   Writes K directly to present_key[batch, head, position, :]
  │       │   Writes V directly to present_value[batch, head, position, :]
  │       │   [TURBO QUANT: Rotate K,V with H before writing; quantize before writing]
  │       │
  │       ├─ IF seq_len > 1 (prompt):
  │       │   FlashAttentionProgram: loadq(), loadk(), loadv(), qk scores, softmax, weighted V, writeo()
  │       │   [TURBO QUANT: dequantize in loadk/loadv; inverse-rotate in writeo]
  │       │
  │       └─ IF seq_len == 1 (decode):
  │           FlashAttentionDecodeQKT → qk scores + metadata
  │           [TURBO QUANT: dequantize K in loadk]
  │           FlashAttentionDecodeSplitVx → softmax(qk) · V
  │           [TURBO QUANT: dequantize V in loadv]
  │           FlashAttentionDecodeVxReduce → reduce splits → output
  │           [TURBO QUANT: inverse-rotate output]
```

### Present Key/Value Tensor (BNSH Format)
```
Shape: [batch_size, kv_num_heads, present_sequence_length, head_size]
  Phi-4: [1, 8, max_length, 128]  with fp16
  TurboQuant: [1, 8, max_length, head_size/8+1]  with uint32  (Phase 3)  e.g., 17 for head_size=128
  Phase 0-2: [1, 8, max_length, 128] with fp16  (unchanged, rotation only)

Access pattern in WGSL:
  offset = (batch * kv_num_heads + head) * present_sequence_length * head_size_vec + token * head_size_vec
  present_key[offset + i]  →  vec4<f16> (4 fp16 elements per load)
```

---

## Building & Running GqaTest

GqaTest is a standalone C++ test binary that exercises GQA with WebGPU EP (flash attention path), comparing baseline vs TurboQuant.

### Prerequisites
- ORT built with WebGPU (`build.bat --use_webgpu --build_shared_lib ...`, build dir `build/WGPU`)
- Python with `onnx` and `numpy` installed (for model generation)

### Generate the GQA Model
```powershell
cd c:\onnxruntime\samples\cxx
python generate_gqa_model.py   # produces gqa_model.onnx
```

### Configure & Build GqaTest
```powershell
# One-time configure (adjust generator as needed):
$buildDir = "c:\onnxruntime\samples\cxx\build"
mkdir $buildDir -Force
cd $buildDir
cmake c:\onnxruntime\samples\cxx `
  -DORT_HEADER_DIR="c:\onnxruntime\include\onnxruntime\core\session" `
  -DORT_LIBRARY_DIR="c:\onnxruntime\build\WGPU\RelWithDebInfo\RelWithDebInfo" `
  -G "Visual Studio 18 2026"

# Build:
cmake --build . --config RelWithDebInfo --target GqaTest
```

### Run
```powershell
# DLLs and model are copied automatically by CMake post-build step.
# If ORT_LIBRARY_DIR has no DLLs (e.g. static build), copy manually:
cd c:\onnxruntime\samples\cxx\build\RelWithDebInfo
.\GqaTest.exe
```

The test creates two sessions (baseline and TurboQuant), runs identical workloads with deterministic seeds, and compares outputs (max abs diff, RMSE, cosine similarity) plus latency.

---

## Todos

### Phase 0: First Checkpoint (V Rotation + Inverse Rotation) — COMPLETE ✓
- [x] **0.1** Add `turboQuant` provider option to WebGPU EP
  - `webgpu_provider_options.h`: Added `kTurboQuant` constant
  - `webgpu_execution_provider.h`: Added config field + member + accessor
  - `webgpu_execution_provider.cc`: Added to initializer list
  - `webgpu_provider_factory.cc`: Parse option in ParseEpConfig()
  - `compute_context.h`: Added ComputeContextBase accessor
- [x] **0.2** Add `turbo_quant_` to `WebgpuAttentionParameters`
  - `attention_common.h`: Added field
  - `group_query_attention.cc`: Set from context in ComputeInternal()
- [x] **0.3** Implement FWHT rotation shader (replaces Hadamard matrix cache)
  - NEW `turbo_quant_hadamard.h`: `TurboQuantRotateProgram` class, `ApplyTurboQuantRotation()`, `ApplyTurboQuantInverseRotation()`
  - NEW `turbo_quant_hadamard.cc`: FWHT butterfly shader in shared memory, output-only binding for in-place ops
  - No matrix storage needed — O(n log n) on-the-fly computation
- [x] **0.4** Wire V rotation into ApplyFlashAttention
  - `flash_attention.cc`: After CopyKVCache/fused split → rotate present_value (new tokens only for share_buffer, all tokens otherwise)
  - `flash_attention.cc`: After prompt flash attention → inverse-rotate output
  - `flash_attention.cc`: After decode VxReduce → inverse-rotate output
- [x] **0.5** Build successfully with turbo_quant changes
- [x] **0.6** GqaTest: Standalone C++ test binary exercising GQA with WebGPU EP (renamed from OpTest)
  - `samples/cxx/generate_gqa_model.py`: Generates minimal ONNX model with single GQA op (Phi-4 params)
  - `samples/cxx/GqaTest.cc`: Dual-session test (baseline vs TurboQuant), deterministic seeds, median-of-10 timing
  - `samples/cxx/CMakeLists.txt`: Standalone CMake build targeting ORT headers + library (target: `GqaTest`)
  - Tests: prompt (1/8/64/256/1024 tokens), single-step decode (0/64/256/1024/4000 past), consistency, perf summary
  - CPU reference FWHT validates GPU shader rotation at every test point
  - **ALL TESTS PASSED** — V rotation cosine=1.000000, output cosine≥0.999998 at all sizes
- [x] **0.7** Fix shader compilation error: duplicate `workgroup_idx`/`local_idx` declarations
  - Root cause: `ShaderHelper` auto-generates `workgroup_idx` and `local_idx` in the main function preamble; our `MainFunctionBody()` redeclared them
- [x] **0.8** Add shader debug output to `program_manager.cc`
  - On pipeline creation failure, full WGSL source is dumped to stderr with error message
- [ ] **0.9** Fix CMake CXX_STANDARD issue for VS 2026 (separate PR)
  - `cmake/CMakeLists.txt`: Changed conditional `set` to unconditional `set(CMAKE_CXX_STANDARD 20)`
  - Root cause: `date` library caches `CMAKE_CXX_STANDARD=17`, blocking ORT's conditional set on rebuilds

---

## Learnings: ORT WebGPU Shader System

### ShaderHelper Auto-Generated Variables
The `ShaderHelper` class (`core/providers/webgpu/shader_helper.cc`) generates a standard preamble for every shader's `main()` function. **Do NOT redeclare** these in `MainFunctionBody()`:
- `local_idx` — function parameter from `@builtin(local_invocation_index)`
- `workgroup_idx` — `let workgroup_idx = workgroup_id.x;` (or multi-dim variant depending on dispatch)
- `global_idx` — `let global_idx = ...;`
- `workgroup_id`, `local_id`, `global_id` — function parameters from `@builtin(...)`
- `sg_id`, `sg_size` — subgroup builtins (when subgroups enabled)

### Shader Assembly Order (7 sections)
`GenerateSourceCode()` assembles the final WGSL in this order:
1. **Feature enables** — `enable f16;`, `enable subgroups;` (auto-added based on tensor types and features)
2. **Constants** — `const workgroup_size_x: u32 = ...;`
3. **Storage buffers** — `@group(0) @binding(N) var<storage, ...> name: array<...>;`
4. **Uniforms** — `struct Uniforms { ... }; @group(0) @binding(M) var<uniform> uniforms: Uniforms;`
5. **Type aliases & helpers** — `alias data_value_t = f16;`, `alias data_element_t = f16;`, accessor functions
6. **AdditionalImplementation()** — custom code (e.g., `var<workgroup>` declarations). Safe to use type aliases here.
7. **Main function** — preamble + `MainFunctionBody()` content

### Tensor Binding Model
- `shader.AddInput("name", ...)` → `var<storage, read>` (read-only)
- `shader.AddOutput("name", ...)` → `var<storage, read_write>` (read-write)
- **In-place operations**: Use `AddOutput()` only. Reading and writing through the same `read_write` binding works. Do NOT bind the same buffer as both input and output — WebGPU validation will reject it.
- `ShaderUsage::UseUniform` — access shape/stride via `uniforms.name_shape[i]`
- `ShaderUsage::UseValueTypeAlias` → `name_value_t` (may be vec4, packed type, etc.)
- `ShaderUsage::UseElementTypeAlias` → `name_element_t` (scalar: f16, f32, u32, etc.)
- `ShaderUsage::UseIndicesTypeAlias` → `name_indices_t` (u32, vec2<u32>, etc.)

### Tensor Access Methods
Both direct indexing and method-based access work:
```wgsl
data[offset]                        // direct read/write by u32 offset
data.getByOffset(offset)            // method-based read
data.setByOffset(offset, value)     // method-based write
data.getByIndices(indices)          // multi-dimensional read
```

### Uniform Variables
Defined in the Program class header via `WEBGPU_PROGRAM_DEFINE_UNIFORM_VARIABLES(...)`. Names must match exactly between the macro and the values passed to `AddUniformVariables()` (same order). Accessed in WGSL as `uniforms.name`.

### Program Caching
Programs are cached by a key that includes the program name, cache hints, tensor types/ranks, and uniform variable types. Use `.CacheHint(...)` to differentiate program variants (e.g., different head sizes or modes).

### past_present_share_buffer Behavior
- **share_buffer=true** (GenAI with static cache): past and present are the same buffer. CopyKVCache writes only new tokens at `past_sequence_length`. TQ rotation targets `[past_seq_len, past_seq_len + kv_seq_len)`.
- **share_buffer=false** (standalone test): past and present are separate buffers. CopyKVCache copies all past tokens + writes new tokens. TQ rotation targets `[0, total_seq_len)` — rotates everything.
- **Implication for multi-step decode**: Chaining decode steps only works correctly with share_buffer=true. In non-share-buffer mode, re-feeding TQ's rotated present_value as next past causes double-rotation (Hadamard is self-inverse, so double-rotation = identity = effectively unrotated).

### Flash Attention Path Selection
Flash attention requires: `context.HasFeature(wgpu::FeatureName::Subgroups)`, `head_size % 4 == 0`, no bias, not packed QKV (packed QKV takes the fused split+RoPE+copy path first, then flash attention). When subgroups are unavailable, falls back to non-flash tiled attention.

### Phase 2 Learnings: WGSL Quantization Shaders

**No `u16` type in WGSL**: WGSL has `u32`, `i32`, `f32`, `f16` but no `u16` or `i16`. To store an f32 value in 2 fp16 slots, use `bitcast<vec2<f16>>(f32_value)` to split the f32 into two fp16 bit patterns, and `bitcast<f32>(vec2<f16>(z, w))` to reconstruct. This is a bitwise reinterpretation, not a type conversion.

**f32 norm storage**: The L2 norm is accumulated in f32 (to avoid fp16 overflow during sum-of-squares) and stored as 2 fp16 bit slots using the bitcast trick above. This uses elements `[head_size-2]` and `[head_size-1]` of each head vector, leaving `head_size-2` elements for centroid indices.

**Fused rotate+quantize shader**: Rather than separate rotation and quantization dispatches, the Phase 2 shader fuses both operations into a single workgroup dispatch per head vector. The shared memory serves double duty: first for FWHT butterfly operations, then for tree reduction of squared values to compute the L2 norm. This halves the number of GPU dispatches for K and V.

**Non-rotary Q path**: When `do_rotary=false` (e.g., GqaTest model with separate Q/K/V), Q is an immutable input tensor. To apply rotation in-place, we must first copy Q to a mutable tensor (`context.CreateGPUTensor()` + `context.CopyTensor()`). The `do_rotary` path avoids this because Q already points to the mutable `query_output` tensor.

**Binary search for searchsorted**: With only 15 decision boundaries, a 4-iteration binary search (`lo=0, hi=15, 4 iterations`) is more efficient than a linear scan and avoids branch divergence in SIMD execution.

---

### Phase 1: QK Rotation — COMPLETE ✓
- [x] **1.1** Rotate K in flash_attention.cc (reuse `ApplyTurboQuantRotation` on `present_key`)
  - `flash_attention.cc` line ~495: `ApplyTurboQuantRotation()` on `present_key` after CopyKVCache, same pattern as V rotation
  - Uses `start_token` / `num_new_tokens` for share_buffer correctness
- [x] **1.2** Rotate Q before attention (new call site, handle num_heads vs kv_num_heads)
  - `flash_attention.cc` line ~518: `ApplyTurboQuantInverseRotation()` on `query_output` (BSNH layout)
  - Uses `InverseRotation` variant because Q is BSNH (same FWHT math — self-inverse)
  - Two paths: `do_rotary` (Q already points to mutable `query_output`) and non-rotary (creates mutable copy via `context.CreateGPUTensor()` + `context.CopyTensor()`, then re-points Q)
- [x] **1.3** Build succeeds with all 5 turbo_quant call sites in flash_attention.cc:
  1. K rotation on `present_key` (BNSH, after CopyKVCache)
  2. V rotation on `present_value` (BNSH, after CopyKVCache)
  3. Q rotation on `query_output` (BSNH, before attention)
  4. Inverse rotation on output (BSNH, after prompt flash attention)
  5. Inverse rotation on output (BSNH, after decode VxReduce)
- [x] **1.4** Run GqaTest and verify: output matches baseline (QK^T invariant, inverse rotation restores output)

### Phase 2: Pseudo-Quantization (fp16 indices) — COMPLETE ✓
- [x] **2.1** Create fused rotate+quantize shader (`TurboQuantRotateQuantizeProgram`)
  - `turbo_quant_hadamard.h`: New `TurboQuantRotateQuantizeProgram` class (head_size, num_heads, is_fp16 constructor params)
  - `turbo_quant_hadamard.cc`: Fused FWHT + L2 norm + searchsorted in single workgroup dispatch
    - f32 shared memory (`tq_shared`, `tq_sq_shared`) for FWHT butterfly and norm accumulation
    - Tree reduction in shared memory for L2 norm (f32 accumulation to avoid overflow)
    - `searchsorted_tq()` binary search: 4 iterations over 15 decision boundaries
    - Norm stored as `bitcast<vec2<f16>>(f32(norm))` in last 2 element slots (head_size-2, head_size-1)
    - Centroid indices (0–15) stored as native element type (fp16) in elements [0..head_size-3]
  - `flash_attention.cc`: K and V now use `ApplyTurboQuantRotateAndQuantize()` (fused) instead of separate `ApplyTurboQuantRotation()`
- [x] **2.2** Dequantize in all 3 flash attention WGSL templates
  - `flash_attention.wgsl.template`: `#param turbo_quant` conditional; TQ_CENTROIDS constant; `tq_read_norm()`/`tq_read_norm_v()` using `bitcast<f32>(vec2<f16>(...))` to recover f32 norm; `tq_dequant_vec4()` function; modified loadk/loadv with per-slot dequantize loop, last vec4 handles z/w as norm bits
  - `flash_attention_decode_qkt.wgsl.template`: TQ_CENTROIDS, `tq_k_norms` shared memory, preloads norms before inner QKT loop, dequantizes K via centroid lookup
  - `flash_attention_decode_split_vx.wgsl.template`: TQ_CENTROIDS, `tq_v_norms` shared memory, preloads norms after softmax, dequantizes V in inner Vx loop
  - All templates use `WGSL_TEMPLATE_PARAMETER(turbo_quant)` with cache hint differentiation
- [x] **2.3** Update GqaTest with CPU reference pseudo-quantize
  - `GqaTest.cc`: Full CPU reference implementation:
    - `TQ_CENTROIDS[16]`, `TQ_BOUNDARIES[15]`, `cpu_searchsorted_tq()`
    - `cpu_fwht_quantize_fp16()`: fused FWHT + f32 norm + quantize on CPU
    - `cpu_rotate_quantize_bnsh()`: applies to token range in BNSH tensor
    - `read_quantized_norm_fp16()`, `cpu_dequantize_fp16()`
    - `compare_quantized_indices()`: counts matching centroid indices between CPU and GPU
  - Test results: **100% centroid index match** (CPU vs GPU) at all sequence lengths
- [x] **2.4** End-to-end quality validation
  - Output cosine similarity: 0.985–0.990 across all test sizes (prompt and decode)
  - Cosine threshold relaxed from 0.99 to 0.95 to account for quantization error
  - ALL TESTS PASSED: prompt (seq_len=1,8,64,256,1024) and decode (past_seq=0,64,256,1024,4000)
  - Consistency tests pass for both baseline and TurboQuant paths

### Remaining Todos (Pre-Phase 3)
- [ ] **2.5** Fix Q const_cast / mutable copy in non-rotary path
  - In `flash_attention.cc` (line ~535), when `!do_rotary && turbo_quant`, Q is immutable so we create a mutable copy via `context.CreateGPUTensor()` + `context.CopyTensor()` and re-point `Q = &query_output`
  - This works but is an extra GPU copy. Ideally, the FWHT rotation shader should accept separate input/output bindings (read-only input, write to query_output) instead of requiring in-place mutation
  - Alternative: modify `ApplyTurboQuantInverseRotation()` to accept a source tensor and destination tensor (copy + rotate in one dispatch)
- [ ] **2.6** Consider fusing Q rotation into flash attention shader itself
  - Q rotation is pure FWHT (no quantization), could be done on-the-fly in the attention kernel to avoid a separate dispatch
  - Lower priority — separate dispatch is cleaner and Q is small (only current tokens)

### Phase 3: Bit-Packing to uint4 + KV Cache Dimension Change
- [ ] **3.1** Update quantize shader: pack 8 indices into one uint32
  - Replace fp16 index storage with bitwise packing
  - Store norm in first uint32 (packed fp16)
- [ ] **3.2** Update dequantize in loadk/loadv: bitwise extract → centroid lookup
  - Replace `u32(fp16_value)` with `(packed >> (slot * 4)) & 0xF`
- [ ] **3.3** GenAI: Modify DefaultKeyValueCache allocation
  - Detect turboQuant from provider options
  - Change shape_[3] to compressed_kv_last_dim (head_size/8 + 1) and type_ to uint32
  - Add uint32_t template specialization for RewindPastTensorsTo
- [ ] **3.4** ORT GQA: Adjust present_key/present_value output shapes
  - present_dims[3] = compressed_kv_last_dim
  - Skip head_size validation for turbo_quant
- [ ] **3.5** End-to-end validation: GenAI + ORT with compressed KV cache
  - Memory savings measurement (expect ~3.76×)
  - Quality comparison (perplexity / generation quality)

### Phase 4: TMAC-style Lookup Table Optimization (Optional)
- [ ] **4.1** Implement TMAC-style LUT for QK dot product
  - Precompute q_i × centroid[c] LUT per Q vector
  - Replace dequantize+dot with LUT lookups
- [ ] **4.2** Benchmark LUT approach vs naive dequantize+dot

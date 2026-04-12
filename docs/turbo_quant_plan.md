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
- Total: 17 uint32 = 68 bytes vs 256 bytes for fp16 → **3.76× savings**

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

### Phase 1: QK Rotation
**Goal**: Extend rotation to Q and K. After this phase, Q, K, V are all Hadamard-rotated.

**Changes**:
- Rotate K before writing to `present_key` (reuse `ApplyTurboQuantRotation` from Phase 0)
- Rotate Q before computing attention scores (separate dispatch on Q tensor, BNSH or BSNH depending on qkv_format)
- **Key correctness property**: If Q' = HQ and K' = HK, then Q'·K'ᵀ = (HQ)·(HK)ᵀ = Q·Hᵀ·H·Kᵀ = Q·Kᵀ (since HᵀH = I). So QK scores are invariant — no inverse rotation needed for QK!
- The inverse rotation is only needed on the attention output (softmax(QK')·V' needs inverse rotation)

**Files**:
- `flash_attention.cc`: Add `ApplyTurboQuantRotation()` call on `present_key` (same as V)
- `flash_attention.cc`: Add Q rotation before prompt/decode attention (new call site)
- May need a variant of `ApplyTurboQuantRotation` for Q's shape (num_heads vs kv_num_heads)

---

### Phase 2: Pseudo-Quantization
**Goal**: After rotation, quantize K and V to 4-bit using TurboQuant codebook. Store compressed in KV cache.

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

**Quantize Shader** (during CopyKV / fused path):
```
1. Load rotated K/V vector (fp16, head_size dims)
2. Compute L2 norm: norm = sqrt(sum(x_i^2))
3. Normalize: x_unit_i = x_i / norm
4. For each element: find bucket via searchsorted on boundaries → 4-bit index
5. Pack 8 indices into one uint32
6. Store: [norm_as_fp16_in_uint32, packed_indices[0..15]] → 17 uint32 per token per head
```

**Dequantize Shader** (during attention computation):
```
1. Load norm from first uint32 (unpack fp16)
2. For each group of 8 elements: unpack uint32 → 8 indices → lookup centroids
3. Multiply each centroid by norm → reconstructed fp16 values
4. Use for dot product (QK^T) or weighted sum (softmax·V)
```

**KV Cache Format**:
```
Per token per KV head: 17 uint32 values
  [0]: fp16 norm packed in uint32 (upper 16 bits unused or used for fp16 second value)
  [1..16]: 128 int4 indices packed as 8 per uint32
```

**Files**:
- `onnxruntime/contrib_ops/webgpu/bert/turbo_quant_quantize.wgsl.template` — NEW
- `onnxruntime/contrib_ops/webgpu/bert/flash_attention_decode_qkt.wgsl.template` — Modified loadk with dequantize
- `onnxruntime/contrib_ops/webgpu/bert/flash_attention_decode_split_vx.wgsl.template` — Modified loadv with dequantize
- `onnxruntime/contrib_ops/webgpu/bert/flash_attention.wgsl.template` — Modified loadk/loadv with dequantize

---

### Phase 3: Adjusting KV Cache Dimensions in GenAI + TMAC-style Lookup Table MatMul
**Goal**: Actually allocate smaller KV cache buffers. Use lookup table for QK dot product.

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
  TurboQuant: [1, 8, max_length, 17]  with uint32  (Phase 3)
  Phase 0-2: [1, 8, max_length, 128] with fp16  (unchanged, rotation only)

Access pattern in WGSL:
  offset = (batch * kv_num_heads + head) * present_sequence_length * head_size_vec + token * head_size_vec
  present_key[offset + i]  →  vec4<f16> (4 fp16 elements per load)
```

---

## Todos

### Phase 0: First Checkpoint (V Rotation + Inverse Rotation)
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
  - `flash_attention.cc`: After CopyKVCache/fused split → rotate present_value (new tokens only)
  - `flash_attention.cc`: After prompt flash attention → inverse-rotate output
  - `flash_attention.cc`: After decode VxReduce → inverse-rotate output
- [x] **0.5** Build successfully with turbo_quant changes
- [ ] **0.6** Test: Run with `turboQuant=1` in genai_config.json
  - Verify output matches non-rotated version (rotation + inverse = identity for V path)
  - Verify no numeric drift across sequence lengths
- [ ] **0.7** Fix CMake CXX_STANDARD issue for VS 2026 (separate PR)
  - `cmake/CMakeLists.txt`: Changed conditional `set` to unconditional `set(CMAKE_CXX_STANDARD 20)`
  - Root cause: `date` library caches `CMAKE_CXX_STANDARD=17`, blocking ORT's conditional set on rebuilds

### Phase 1: QK Rotation
- [ ] **1.1** Rotate K in flash_attention.cc (reuse `ApplyTurboQuantRotation` on `present_key`)
- [ ] **1.2** Rotate Q before attention (new call site, handle num_heads vs kv_num_heads)
- [ ] **1.3** Verify: QK^T scores identical with and without rotation (since H is orthogonal)
- [ ] **1.4** Verify: output differs only by inverse rotation (which we apply)

### Phase 2: Pseudo-Quantization
- [ ] **2.1** Create quantize shader: rotate → normalize → searchsorted → pack int4 + scale
  - NEW `turbo_quant_quantize.wgsl.template`
  - Fuse with rotation for efficiency
- [ ] **2.2** Modify flash attention loadk/loadv to dequantize
  - Unpack int4 → lookup centroid → multiply by scale
  - Template parameter `#param turbo_quant` in each WGSL template
- [ ] **2.3** Validate against reference PyTorch implementation
  - Compare attention scores with/without quantization
  - Measure perplexity impact

### Phase 3: KV Cache Dimension Change + TMAC
- [ ] **3.1** GenAI: Modify DefaultKeyValueCache allocation
  - Detect turboQuant from provider options
  - Change shape_[3] and type_ for compressed storage
  - Add uint32_t template specialization for RewindPastTensorsTo
- [ ] **3.2** ORT GQA: Adjust present_key/present_value output shapes
  - present_dims[3] = compressed_kv_last_dim
  - Skip head_size validation for turbo_quant
- [ ] **3.3** Update quantize/dequantize shaders for uint32 packed format
- [ ] **3.4** (Optional) Implement TMAC-style LUT for QK dot product
  - Precompute q_i × centroid[c] LUT per Q vector
  - Replace dequantize+dot with LUT lookups
- [ ] **3.5** End-to-end validation: GenAI + ORT with compressed KV cache
  - Memory savings measurement
  - Quality comparison (perplexity / generation quality)

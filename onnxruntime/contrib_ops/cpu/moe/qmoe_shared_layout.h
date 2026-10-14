// SharedExpertBlockV1 and NVFP4 RowMajor CPU/GPU shared expert layouts and kernels.
// See dev/docs/qmoe_offload_design_v3.md (Sections 5, 6, 8, and F12).
// This header is intentionally independent of ORT internal headers so it can be unit-tested standalone.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>

namespace onnxruntime {
namespace contrib {
namespace qmoe_shared {

constexpr int64_t kBlockSize = 32;          // K values per FP16 scale (SharedExpertBlockV1 INT4)
constexpr int64_t kNvfp4BlockSize = 16;     // K values per Float8E4M3FN scale (NVFP4)
constexpr int64_t kColumnsInterleaved = 4;  // ColumnMajorTileInterleave<64,4>
constexpr int64_t kRowsPerTile = 64;        // K values per interleave tile
constexpr size_t kBankAlignment = 256;
constexpr int kMaxRowsPerCall = 8;

constexpr int64_t kRequiredWeightsPrepacked = 1;
constexpr const char* kSharedLayoutId = "ort.qmoe.w4a16.block32.shared.v1";
constexpr const char* kNvfp4RowMajorLayoutId = "ort.qmoe.nvfp4.block16.rowmajor.v1";
constexpr const char* kNvfp4SharedLayoutId = "ort.qmoe.nvfp4.block16.shared.v1";

bool IsAcceptedPrepackContract(int64_t weights_prepacked, int64_t block_size);
bool IsAcceptedNvfp4PrepackContract(int64_t weights_prepacked, int64_t block_size);

// Standalone checked arithmetic helpers (F12).
inline bool CheckedMulSize(size_t a, size_t b, size_t* out) noexcept {
  if (out == nullptr) return false;
  if (a != 0 && b > std::numeric_limits<size_t>::max() / a) return false;
  *out = a * b;
  return true;
}

inline bool CheckedAddSize(size_t a, size_t b, size_t* out) noexcept {
  if (out == nullptr) return false;
  if (b > std::numeric_limits<size_t>::max() - a) return false;
  *out = a + b;
  return true;
}

inline bool CheckedAlignUp(size_t v, size_t alignment, size_t* out) noexcept {
  if (out == nullptr || alignment == 0 || (alignment & (alignment - 1)) != 0) return false;
  const size_t mask = alignment - 1;
  if (v > std::numeric_limits<size_t>::max() - mask) return false;
  *out = (v + mask) & ~mask;
  return true;
}

// Validates that E * stride and external [offset, offset + length) cannot overflow (F12).
bool ValidateExpertStorageRange(int64_t num_experts, size_t stride, uint64_t external_offset,
                                uint64_t external_length, size_t* total_bytes = nullptr) noexcept;

// Lease descriptor recording actual pointer alignment alongside length (F12).
struct ExpertBlockLease {
  const uint8_t* data = nullptr;
  size_t length = 0;
  size_t alignment = 0;
};

size_t ComputePointerAlignment(const void* ptr) noexcept;
bool MakeExpertBlockLease(const void* ptr, size_t length, size_t min_required_alignment,
                          ExpertBlockLease* lease) noexcept;

// Byte offsets of the six banks inside one SharedExpertBlockV1 expert block.
// FC1 is the fused interleaved gate/up projection [2I,H]; FC2 is [H,I].
// Scales are FP16 block-major [K/32,N]; biases are FP16 [N].
struct ExpertBlockLayout {
  int64_t hidden = 0;
  int64_t inter = 0;
  size_t fc1_weights = 0;
  size_t fc1_scales = 0;
  size_t fc1_bias = 0;
  size_t fc2_weights = 0;
  size_t fc2_scales = 0;
  size_t fc2_bias = 0;
  size_t stride = 0;
};

// Returns false if the shape cannot use the layout or any intermediate product overflows (F12).
bool MakeExpertBlockLayout(int64_t hidden, int64_t inter, ExpertBlockLayout* layout);

// Byte offsets inside a contiguous NVFP4 row-major expert block (optional packed block mode).
struct Nvfp4ExpertBlockLayout {
  int64_t hidden = 0;
  int64_t inter = 0;
  size_t fc1_weights = 0;       // [2I, H/2] uint8 E2M1 row-major
  size_t fc1_scales = 0;        // [2I, H/16] uint8 Float8E4M3FN row-major
  size_t fc1_global_scale = 0;  // [1] float
  size_t fc2_weights = 0;       // [H, I/2] uint8 E2M1 row-major
  size_t fc2_scales = 0;        // [H, I/16] uint8 Float8E4M3FN row-major
  size_t fc2_global_scale = 0;  // [1] float
  size_t stride = 0;
};

bool MakeNvfp4ExpertBlockLayout(int64_t hidden, int64_t inter, Nvfp4ExpertBlockLayout* layout);

// Zero-copy view of one NVFP4 expert (works with either separate bank tensors or a contiguous block).
struct Nvfp4ExpertView {
  const uint8_t* fc1_weights = nullptr;  // [2*inter, hidden/2]
  const uint8_t* fc1_scales = nullptr;   // [2*inter, hidden/16] E4M3
  float fc1_global_scale = 1.0f;
  const uint16_t* fc1_bias = nullptr;    // optional [2*inter] in BF16 or FP16
  const uint8_t* fc2_weights = nullptr;  // [hidden, inter/2]
  const uint8_t* fc2_scales = nullptr;   // [hidden, inter/16] E4M3
  float fc2_global_scale = 1.0f;
  const uint16_t* fc2_bias = nullptr;    // optional [hidden] in BF16 or FP16
  bool bias_is_bf16 = true;
};

// Original K offset (0..31) of the low nibble of byte b (0..15) of a 32-K block; the high nibble
// holds that offset plus 8.
inline int LowNibbleK(int b) { return (b >> 1) + 16 * (b & 1); }

// Natural-order index -> index in the [low16 | high16] arrangement used by the INT4 CPU kernels.
int ReorderedPosition(int k_offset_in_block);

// Reorders one activation row (length k, multiple of 32) into the per-block INT4 decode order.
void ReorderActivationRow(const float* src, float* dst, int64_t k);

// Natural-order index (0..31) -> index in the NVFP4 32-element pair-of-16-blocks arrangement:
// lanes 0..7 = even K of block 0, lanes 8..15 = even K of block 1,
// lanes 16..23 = odd K of block 0, lanes 24..31 = odd K of block 1.
inline int Nvfp4ReorderedPosition(int k_offset_in_32) {
  const int blk = (k_offset_in_32 >> 4) & 1;
  const int pos = k_offset_in_32 & 15;
  return ((pos & 1) << 4) + (blk << 3) + (pos >> 1);
}

// Reorders one activation row (length k, multiple of 32) into the NVFP4 AVX-512 decode order.
void ReorderNvfp4ActivationRow(const float* src, float* dst, int64_t k);

struct SwiGluParams {
  float alpha = 1.0f;
  float beta = 0.0f;
  float limit = std::numeric_limits<float>::infinity();
};

uint8_t WeightCodeAt(const uint8_t* weights, int64_t n_total, int64_t k_total, int64_t n, int64_t k);

float HalfToFloat(uint16_t h);
uint16_t FloatToHalf(float f);
float BFloat16ToFloat(uint16_t b);
uint16_t FloatToBFloat16(float f);
float DecodeFp4E2M1(uint8_t code);
float DecodeFloat8E4M3FN(uint8_t code);

void GemmRange(const uint8_t* weights, const uint16_t* scales, const uint16_t* bias, int64_t n_total,
               int64_t k, const float* a, int64_t m, float* out, int64_t out_stride, int64_t n_begin,
               int64_t n_end, bool force_scalar = false);

void Fc1SwiGluRange(const ExpertBlockLayout& layout, const uint8_t* block, const float* x_reordered,
                    int64_t m, const SwiGluParams& swiglu, float* inter_reordered, int64_t n_begin,
                    int64_t n_end, bool force_scalar = false, bool fp16_boundaries = false);

void Fc2Range(const ExpertBlockLayout& layout, const uint8_t* block, const float* inter_reordered,
              int64_t m, float* out, int64_t n_begin, int64_t n_end, bool force_scalar = false,
              bool apply_bias = true);

bool ExpertForward(const ExpertBlockLayout& layout, const uint8_t* block, const float* x, int64_t m,
                   const SwiGluParams& swiglu, float* out, bool force_scalar = false, bool apply_fc2_bias = true,
                   bool fp16_boundaries = false);

// NVFP4 row-major CPU kernels (supports both BF16 and FP16 scale/boundary rounding).
void Nvfp4GemmRange(const uint8_t* weights, const uint8_t* block_scales, float global_scale,
                    const uint16_t* bias, bool bias_is_bf16, int64_t n_total, int64_t k,
                    const float* a_reordered, int64_t m, float* out, int64_t out_stride,
                    int64_t n_begin, int64_t n_end, bool force_scalar = false,
                    bool round_scale_bf16 = true);

void Nvfp4Fc1SwiGluRange(int64_t hidden, int64_t inter, const Nvfp4ExpertView& view,
                         const float* x_reordered, int64_t m, const SwiGluParams& swiglu,
                         float* inter_reordered, int64_t n_begin, int64_t n_end,
                         bool force_scalar = false, bool bf16_boundaries = true,
                         bool fp16_boundaries = false);

void Nvfp4Fc2Range(int64_t hidden, int64_t inter, const Nvfp4ExpertView& view,
                   const float* inter_reordered, int64_t m, float* out,
                   int64_t n_begin, int64_t n_end, bool force_scalar = false,
                   bool apply_bias = false, bool round_scale_bf16 = true);

bool Nvfp4ExpertForward(int64_t hidden, int64_t inter, const Nvfp4ExpertView& view,
                        const float* x, int64_t m, const SwiGluParams& swiglu, float* out,
                        bool force_scalar = false, bool apply_fc2_bias = false,
                        bool bf16_boundaries = true, bool fp16_boundaries = false);

bool CpuHasAvx512();

}  // namespace qmoe_shared
}  // namespace contrib
}  // namespace onnxruntime

extern "C" {
int qmoe_shared_expert_forward(const uint8_t* block, int64_t hidden, int64_t inter, const float* x,
                               int64_t m, float* out, float alpha, float beta, float limit,
                               int force_scalar);
int qmoe_shared_expert_contribution(const uint8_t* block, int64_t hidden, int64_t inter, const float* x,
                                    int64_t m, float* out, float alpha, float beta, float limit,
                                    int force_scalar);
int qmoe_shared_expert_contribution_fp16(const uint8_t* block, int64_t hidden, int64_t inter, const float* x,
                                         int64_t m, float* out, float alpha, float beta, float limit,
                                         int force_scalar);
int64_t qmoe_shared_block_bytes(int64_t hidden, int64_t inter);
int qmoe_shared_cpu_has_avx512(void);
int qmoe_shared_accepts_prepack_contract(int64_t weights_prepacked, int64_t block_size);
int qmoe_shared_weight_code_at(const uint8_t* weights, int64_t n_total, int64_t k_total, int64_t n,
                               int64_t k);
int qmoe_nvfp4_expert_contribution_bf16(const uint8_t* fc1_w, const uint8_t* fc1_s, float fc1_g,
                                        const uint8_t* fc2_w, const uint8_t* fc2_s, float fc2_g,
                                        int64_t hidden, int64_t inter, const float* x, int64_t m,
                                        float* out, float alpha, float beta, float limit,
                                        int force_scalar);
int64_t qmoe_nvfp4_block_bytes(int64_t hidden, int64_t inter);
}

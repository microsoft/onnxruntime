#include "contrib_ops/cuda/llm/moe_gemm/deep_gemm_sm90.h"

#include <cmath>

#include <cuda.h>
#include <deep_gemm/impls/sm90_fp8_gemm_1d2d.cuh>

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::llm::kernels::deep_gemm_sm90 {
namespace {

constexpr int kNumSms = 132;
constexpr int kBlockM = 64;
constexpr int kBlockK = kQuantBlockSize;
// Measured winners for these two shapes; a 1x1 cluster beat both 2x1 and 1x2 everywhere.
constexpr int kFc1BlockN = 64;
constexpr int kFc2BlockN = 128;

// TMA flattens [expert, row], so each expert stride must end on a store-tile boundary and a
// store tile taller than one expert's padded slot would clobber the next expert's rows.
static_assert(kPaddedTokensPerExpert % kBlockM == 0);
static_assert(kBlockM <= kPaddedTokensPerExpert);
// The pack and SwiGLU kernels bound their work by the per-expert row count, so a count past the
// padded stride would spill into the next expert's slot.
static_assert(kMaxTokensPerExpert <= kPaddedTokensPerExpert);
// InterleavedSwiGLUKernel loads each gate/linear pair as one __nv_bfloat162.
static_assert(kFc1OutputSize == 2 * kInterSize);
// One CUDA block quantizes exactly one 128-element K chunk, which is also the granularity the
// GEMM's per-128-channel scaling assumes.
static_assert(kHiddenSize % kQuantBlockSize == 0);
static_assert(kInterSize % kQuantBlockSize == 0);
// SFA is MN-major [G, K / 128, kPaddedTokensPerExpert]; DeepGEMM aligns the MN extent to 4 and
// the TMA global stride (kPaddedTokensPerExpert floats) must be 16-byte aligned.
static_assert(kPaddedTokensPerExpert % 4 == 0);
static_assert((kPaddedTokensPerExpert * sizeof(float)) % 16 == 0);
// MGroupedMasked schedules whole BLOCK_N tiles per expert.
static_assert(kFc1OutputSize % kFc1BlockN == 0);
static_assert(kHiddenSize % kFc2BlockN == 0);

constexpr uint32_t CeilDiv(uint32_t a, uint32_t b) { return (a + b - 1) / b; }
constexpr uint32_t AlignUp(uint32_t a, uint32_t b) { return CeilDiv(a, b) * b; }

// heuristics/utils.hpp::get_swizzle_mode
constexpr uint32_t SwizzleMode(uint32_t block_size, uint32_t elem_size) {
  const uint32_t bytes = block_size * elem_size;
  if (bytes % 128 == 0) return 128;
  if (bytes % 64 == 0) return 64;
  if (bytes % 32 == 0) return 32;
  return 16;
}

// Mirrors SM90ArchSpec::get_pipeline_config for KernelType::Kernel1D2D with fp8 in / bf16 out.
template <uint32_t BlockN, uint32_t K>
struct Fp8Config {
  static constexpr uint32_t kSwizzleA = SwizzleMode(kBlockK, 1);
  static constexpr uint32_t kSwizzleB = SwizzleMode(kBlockK, 1);
  static constexpr uint32_t kSwizzleD = SwizzleMode(BlockN, static_cast<uint32_t>(sizeof(__nv_bfloat16)));
  static constexpr uint32_t kTmaThreads = 128;
  static constexpr uint32_t kMathThreads = kBlockM <= 64 ? 128u : 256u;
  static constexpr uint32_t kThreads = kTmaThreads + kMathThreads;

  // Shared memory a single H100 kernel may claim.
  static constexpr uint32_t kSmemCapacity = 232448;
  static constexpr uint32_t kSmemCd =
      AlignUp(static_cast<uint32_t>(kBlockM) * BlockN * static_cast<uint32_t>(sizeof(__nv_bfloat16)), 1024);
  static constexpr uint32_t kSmemBarriers = 16 * 8 * 2;
  static constexpr uint32_t kUniformSfb = (kBlockK % BlockN == 0) ? 1u : 2u;
  static constexpr uint32_t kSmemSfb = AlignUp(CeilDiv(K, kBlockK) * 4 * kUniformSfb, 8);
  static constexpr uint32_t kSmemExtra = kSmemCd + kSmemBarriers + kSmemSfb;
  static constexpr uint32_t kSmemPerStage =
      static_cast<uint32_t>(kBlockM) * kBlockK + BlockN * kBlockK +
      AlignUp(static_cast<uint32_t>(kBlockM) * static_cast<uint32_t>(sizeof(float)), 128);
  static constexpr uint32_t kStages =
      ((kSmemCapacity - kSmemExtra) / kSmemPerStage) < 16u
          ? ((kSmemCapacity - kSmemExtra) / kSmemPerStage)
          : 16u;
  static constexpr uint32_t kSmemBytes = kSmemExtra + kStages * kSmemPerStage;

  static_assert(kStages > 0, "DeepGEMM fp8 pipeline does not fit in shared memory");
};

// e4m3's largest finite magnitude. Activations carry full mantissas, so they take the
// conventional amax/448 scale rather than the power of two the weight prepack uses.
constexpr float kE4m3Max = 448.0f;

__device__ __forceinline__ float ActivationScale(float amax) {
  return amax > 0.0f ? amax * (1.0f / kE4m3Max) : 1.0f;
}

// Max |value| across one 128-thread block.
__device__ __forceinline__ float BlockAbsMax(float value) {
  __shared__ float partial[4];
  value = fabsf(value);
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_xor_sync(0xffffffffu, value, offset));
  }
  if ((threadIdx.x & 31) == 0) {
    partial[threadIdx.x >> 5] = value;
  }
  __syncthreads();
  return fmaxf(fmaxf(partial[0], partial[1]), fmaxf(partial[2], partial[3]));
}

// One block per (expert, row, 128-element K chunk). Only kMaxTokensPerExpert rows are launched:
// every token picks distinct experts, so an expert can hold at most num_rows tokens and the
// caller has already required num_rows <= kMaxTokensPerExpert. Rows past the count are left
// stale rather than zero-filled: the GEMMs are row-independent, masked_m stops the scheduler,
// and UnpackOutputKernel copies back only the first `count` rows.
template <int K>
__global__ void PackInputKernel(const __nv_bfloat16* __restrict__ input,
                                const int64_t* __restrict__ offsets,
                                __nv_fp8_e4m3* __restrict__ output,
                                float* __restrict__ scales,
                                int* __restrict__ masked_m) {
  const int expert = blockIdx.z;
  const int64_t first = offsets[expert];
  const int count = static_cast<int>(offsets[expert + 1] - first);
  if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
    masked_m[expert] = count;
  }
  const int row = blockIdx.y;
  if (row >= count) {
    return;
  }
  const int col = blockIdx.x * kQuantBlockSize + static_cast<int>(threadIdx.x);
  const float value = __bfloat162float(input[(first + row) * K + col]);
  const float scale = ActivationScale(BlockAbsMax(value));
  output[(static_cast<int64_t>(expert) * kPaddedTokensPerExpert + row) * K + col] =
      static_cast<__nv_fp8_e4m3>(value / scale);
  if (threadIdx.x == 0) {
    scales[(static_cast<int64_t>(expert) * (K / kQuantBlockSize) + blockIdx.x) * kPaddedTokensPerExpert + row] =
        scale;
  }
}

// FC1's output stays bf16 (the GEMM's C/D dtype); this kernel is what re-quantizes it for FC2.
__global__ void InterleavedSwiGLUKernel(const __nv_bfloat16* __restrict__ input,
                                        __nv_fp8_e4m3* __restrict__ output,
                                        float* __restrict__ scales,
                                        const int* __restrict__ masked_m,
                                        float alpha, float beta, float limit) {
  const int expert = blockIdx.z;
  const int row = blockIdx.y;
  if (row >= masked_m[expert]) {
    return;
  }
  const int col = blockIdx.x * kQuantBlockSize + static_cast<int>(threadIdx.x);
  const int64_t packed_row = static_cast<int64_t>(expert) * kPaddedTokensPerExpert + row;
  // gate and linear are adjacent in the interleaved FC1 output, so they load as one 4-byte pair.
  const __nv_bfloat162 pair =
      *reinterpret_cast<const __nv_bfloat162*>(input + packed_row * kFc1OutputSize + 2 * col);
  float gate = __bfloat162float(pair.x);
  float linear = __bfloat162float(pair.y);
  if (isfinite(limit)) {
    gate = fminf(gate, limit);
    linear = fminf(fmaxf(linear, -limit), limit);
  }
  linear += beta;
  const float value = gate * (1.0f / (1.0f + expf(-alpha * gate))) * linear;

  const float scale = ActivationScale(BlockAbsMax(value));
  output[packed_row * kInterSize + col] = static_cast<__nv_fp8_e4m3>(value / scale);
  if (threadIdx.x == 0) {
    scales[(static_cast<int64_t>(expert) * (kInterSize / kQuantBlockSize) + blockIdx.x) * kPaddedTokensPerExpert + row] =
        scale;
  }
}

template <int N>
__global__ void UnpackOutputKernel(const __nv_bfloat16* input, const int64_t* offsets,
                                   __nv_bfloat16* output) {
  const int expert = blockIdx.y;
  const int count = static_cast<int>(offsets[expert + 1] - offsets[expert]);
  const int total = count * N;
  for (int index = blockIdx.x * blockDim.x + threadIdx.x;
       index < total; index += blockDim.x * gridDim.x) {
    const int row = index / N;
    const int col = index % N;
    output[(offsets[expert] + row) * N + col] =
        input[(expert * kPaddedTokensPerExpert + row) * N + col];
  }
}

CUtensorMap MakeTensorMap(void* pointer, CUtensorMapDataType dtype, uint32_t elem_size,
                          uint32_t gmem_inner, uint32_t gmem_outer, uint32_t box_inner,
                          uint32_t box_outer, uint32_t gmem_outer_stride_elems, uint32_t swizzle) {
  // A swizzled box always spans exactly one swizzle unit along the inner dimension.
  if (swizzle != 0) {
    box_inner = swizzle / elem_size;
  }
  CUtensorMap map{};
  const cuuint64_t dims[2] = {gmem_inner, gmem_outer};
  const cuuint64_t strides[1] = {static_cast<cuuint64_t>(gmem_outer_stride_elems) * elem_size};
  const cuuint32_t box[2] = {box_inner, box_outer};
  const cuuint32_t elem_strides[2] = {1, 1};
  CUtensorMapSwizzle swizzle_mode = CU_TENSOR_MAP_SWIZZLE_NONE;
  if (swizzle == 128) {
    swizzle_mode = CU_TENSOR_MAP_SWIZZLE_128B;
  } else if (swizzle == 64) {
    swizzle_mode = CU_TENSOR_MAP_SWIZZLE_64B;
  } else if (swizzle == 32) {
    swizzle_mode = CU_TENSOR_MAP_SWIZZLE_32B;
  }
  CUresult result = cuTensorMapEncodeTiled(
      &map, dtype, 2, pointer, dims, strides, box, elem_strides,
      CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle_mode,
      CU_TENSOR_MAP_L2_PROMOTION_L2_256B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
  ORT_ENFORCE(result == CUDA_SUCCESS, "DeepGEMM cuTensorMapEncodeTiled failed: ", static_cast<int>(result));
  return map;
}

// Mirrors DeepGEMM's sm90_m_grouped_fp8_gemm_masked_1d2d host launcher: SFB is a plain pointer,
// A/B/D/SFA move through TMA. SHAPE_M stays dynamic (0) because the scheduler derives each
// expert's row count from masked_m -- an expert with masked_m == 0 contributes no tile, so the
// number of experts costs scheduling only, not GEMM work.
template <int NumExperts, int N, int K, int BlockN>
void LaunchGemm(const __nv_fp8_e4m3* a, const float* sfa, const __nv_fp8_e4m3* b, const float* sfb,
                __nv_bfloat16* d, int* masked_m, cudaStream_t stream) {
  using Config = Fp8Config<static_cast<uint32_t>(BlockN), static_cast<uint32_t>(K)>;
  auto kernel = &deep_gemm::sm90_fp8_gemm_1d2d_impl<
      cute::UMMA::Major::K, 0, N, K, NumExperts,
      kBlockM, BlockN, kBlockK,
      Config::kSwizzleA, Config::kSwizzleB, Config::kSwizzleD,
      Config::kStages, Config::kTmaThreads, Config::kMathThreads,
      /*kNumTMAMulticast=*/1, /*kIsTMAMulticastOnA=*/false,
      kNumSms, deep_gemm::GemmType::MGroupedMasked, cutlass::bfloat16_t,
      deep_gemm::epilogue::transform::EpilogueIdentity>;

  // A: K-major [G * kPaddedTokensPerExpert, K], tile (BLOCK_M, BLOCK_K).
  auto map_a = MakeTensorMap(const_cast<__nv_fp8_e4m3*>(a), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                             K, NumExperts * kPaddedTokensPerExpert, kBlockK, kBlockM, K,
                             Config::kSwizzleA);
  // B: K-major [G * N, K], tile (BLOCK_N, BLOCK_K).
  auto map_b = MakeTensorMap(const_cast<__nv_fp8_e4m3*>(b), CU_TENSOR_MAP_DATA_TYPE_UINT8, 1,
                             K, NumExperts * N, kBlockK, BlockN, K, Config::kSwizzleB);
  // D: [G * kPaddedTokensPerExpert, N], tile (BLOCK_M, BLOCK_N).
  auto map_d = MakeTensorMap(d, CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, 2, N,
                             NumExperts * kPaddedTokensPerExpert, BlockN, kBlockM, N,
                             Config::kSwizzleD);
  // SFA: MN-major [G * K / 128, kPaddedTokensPerExpert], one scale row per 128-channel K chunk.
  auto map_sfa = MakeTensorMap(const_cast<float*>(sfa), CU_TENSOR_MAP_DATA_TYPE_FLOAT32, 4,
                               kPaddedTokensPerExpert,
                               CeilDiv(static_cast<uint32_t>(K), kBlockK) * NumExperts,
                               kBlockM, 1, kPaddedTokensPerExpert, 0);

  CUDA_CALL_THROW(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
                                       Config::kSmemBytes));
  cudaLaunchAttribute attributes[2]{};
  attributes[0].id = cudaLaunchAttributeClusterDimension;
  attributes[0].val.clusterDim = {1, 1, 1};
  attributes[1].id = cudaLaunchAttributeProgrammaticStreamSerialization;
  attributes[1].val.programmaticStreamSerializationAllowed = 1;
  cudaLaunchConfig_t config{};
  config.gridDim = dim3(kNumSms, 1, 1);
  config.blockDim = dim3(Config::kThreads, 1, 1);
  config.dynamicSmemBytes = Config::kSmemBytes;
  config.stream = stream;
  config.attrs = attributes;
  config.numAttrs = 2;
  CUDA_CALL_THROW(cudaLaunchKernelEx(&config, kernel, const_cast<float*>(sfb), masked_m,
                                     static_cast<uint32_t>(kPaddedTokensPerExpert),
                                     static_cast<uint32_t>(N), static_cast<uint32_t>(K),
                                     map_a, map_b, map_d, map_sfa));
}

constexpr size_t kBufferAlign = 256;
constexpr size_t AlignBytes(size_t bytes) {
  return (bytes + kBufferAlign - 1) / kBufferAlign * kBufferAlign;
}

// Every buffer is expert-major over kPaddedTokensPerExpert rows, so the whole layout scales
// linearly with the expert count.
struct WorkspaceLayout {
  size_t fc1_a;
  size_t fc1_sfa;
  size_t fc1_out;
  size_t fc2_a;
  size_t fc2_sfa;
  size_t fc2_out;
  size_t masked;

  explicit constexpr WorkspaceLayout(int num_experts)
      : fc1_a(AlignBytes(static_cast<size_t>(num_experts) * kPaddedTokensPerExpert * kHiddenSize *
                         sizeof(__nv_fp8_e4m3))),
        fc1_sfa(AlignBytes(static_cast<size_t>(num_experts) * (kHiddenSize / kQuantBlockSize) *
                           kPaddedTokensPerExpert * sizeof(float))),
        fc1_out(AlignBytes(static_cast<size_t>(num_experts) * kPaddedTokensPerExpert *
                           kFc1OutputSize * sizeof(__nv_bfloat16))),
        fc2_a(AlignBytes(static_cast<size_t>(num_experts) * kPaddedTokensPerExpert * kInterSize *
                         sizeof(__nv_fp8_e4m3))),
        fc2_sfa(AlignBytes(static_cast<size_t>(num_experts) * (kInterSize / kQuantBlockSize) *
                           kPaddedTokensPerExpert * sizeof(float))),
        fc2_out(AlignBytes(static_cast<size_t>(num_experts) * kPaddedTokensPerExpert * kHiddenSize *
                           sizeof(__nv_bfloat16))),
        masked(AlignBytes(static_cast<size_t>(num_experts) * sizeof(int))) {}

  constexpr size_t Total() const {
    return fc1_a + fc1_sfa + fc1_out + fc2_a + fc2_sfa + fc2_out + masked;
  }
};

template <int NumExperts>
void RunImpl(const __nv_bfloat16* compact_input, const int64_t* offsets,
             const __nv_fp8_e4m3* fc1_weights, const float* fc1_weight_scales,
             const __nv_fp8_e4m3* fc2_weights, const float* fc2_weight_scales,
             __nv_bfloat16* compact_output, float alpha, float beta, float limit,
             void* workspace, cudaStream_t stream) {
  constexpr WorkspaceLayout layout(NumExperts);
  auto* cursor = static_cast<uint8_t*>(workspace);
  auto* fc1_a = reinterpret_cast<__nv_fp8_e4m3*>(cursor);
  cursor += layout.fc1_a;
  auto* fc1_sfa = reinterpret_cast<float*>(cursor);
  cursor += layout.fc1_sfa;
  auto* fc1_output = reinterpret_cast<__nv_bfloat16*>(cursor);
  cursor += layout.fc1_out;
  auto* fc2_a = reinterpret_cast<__nv_fp8_e4m3*>(cursor);
  cursor += layout.fc2_a;
  auto* fc2_sfa = reinterpret_cast<float*>(cursor);
  cursor += layout.fc2_sfa;
  auto* fc2_output = reinterpret_cast<__nv_bfloat16*>(cursor);
  cursor += layout.fc2_out;
  auto* masked_m = reinterpret_cast<int*>(cursor);

  PackInputKernel<kHiddenSize>
      <<<dim3(kHiddenSize / kQuantBlockSize, kMaxTokensPerExpert, NumExperts),
         kQuantBlockSize, 0, stream>>>(compact_input, offsets, fc1_a, fc1_sfa, masked_m);
  LaunchGemm<NumExperts, kFc1OutputSize, kHiddenSize, kFc1BlockN>(
      fc1_a, fc1_sfa, fc1_weights, fc1_weight_scales, fc1_output, masked_m, stream);
  InterleavedSwiGLUKernel<<<dim3(kInterSize / kQuantBlockSize, kMaxTokensPerExpert, NumExperts),
                            kQuantBlockSize, 0, stream>>>(fc1_output, fc2_a, fc2_sfa, masked_m, alpha, beta, limit);
  LaunchGemm<NumExperts, kHiddenSize, kInterSize, kFc2BlockN>(
      fc2_a, fc2_sfa, fc2_weights, fc2_weight_scales, fc2_output, masked_m, stream);
  UnpackOutputKernel<kHiddenSize><<<dim3(32, NumExperts), 256, 0, stream>>>(
      fc2_output, offsets, compact_output);
}

}  // namespace

size_t GetWorkspaceSize(int num_experts) {
  ORT_ENFORCE(NumExpertsSupported(num_experts), "DeepGEMM: unsupported expert count ", num_experts);
  return WorkspaceLayout(num_experts).Total();
}

void Run(const __nv_bfloat16* compact_input, const int64_t* offsets,
         const __nv_fp8_e4m3* fc1_weights, const float* fc1_weight_scales,
         const __nv_fp8_e4m3* fc2_weights, const float* fc2_weight_scales,
         __nv_bfloat16* compact_output, int num_experts, float alpha, float beta, float limit,
         void* workspace, cudaStream_t stream) {
  ORT_ENFORCE(num_experts == kNumExpertsWorld8, "DeepGEMM: unsupported expert count ", num_experts);
  RunImpl<kNumExpertsWorld8>(compact_input, offsets, fc1_weights, fc1_weight_scales, fc2_weights,
                             fc2_weight_scales, compact_output, alpha, beta, limit, workspace, stream);
}

}  // namespace onnxruntime::llm::kernels::deep_gemm_sm90
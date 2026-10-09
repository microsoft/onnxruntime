// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Test can be run like the following:
//  ./onnxruntime_provider_test --gtest_filter=CUDA_EP_Unittest.*
#if USE_FPA_INTB_GEMM
#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cutlass/numeric_types.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_profiler.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"
#include "contrib_ops/cuda/llm/gemm_profiler.h"
#include "contrib_ops/cuda/quantization/matmul_nbits.cuh"
#include "contrib_ops/cuda/quantization/dequantize_blockwise.cuh"
#include "core/providers/cuda/shared_inc/fpgeneric.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <tuple>
#include <vector>

namespace wo = onnxruntime::llm::kernels::fpA_intB_gemv;
namespace wo_profile = onnxruntime::llm::kernels::weight_only;
using onnxruntime::llm::cutlass_extensions::CutlassGemmConfig;

namespace {
constexpr bool kPipelineMode = true;  // CI pipeline?
constexpr int kMinSupportedSm = 75;

std::vector<int> get_m_list() {
#if USE_COMPACT_FPA_INTB_GEMM
  return {1, 15, 16, 32, 128, 512};
#else
  if (kPipelineMode) {
    return {1, 14};
  } else {
    return {1, 4, 8, 14, 256, 512, 1024, 2048};
  }
#endif
}

std::vector<std::pair<int, int>> get_n_k_list(wo::KernelType kernel_type) {
#if USE_COMPACT_FPA_INTB_GEMM
  std::vector<std::pair<int, int>> shapes{{2880, 4096}, {5120, 2880}};
  if (kernel_type == wo::KernelType::FP16Int8Groupwise) {
    shapes.emplace_back(201088, 2880);
  }
  return shapes;
#else
  ORT_UNUSED_PARAMETER(kernel_type);
  if (kPipelineMode) {
    return {{5120, 3072}};
  } else {
    // N and K of phi4 mini.
    return {{5120, 3072}, {8192, 3072}, {3072, 8192}, {200064, 3072}};
  }
#endif
}

struct CudaBuffer {
  void* _data = nullptr;
  size_t _bytes;

  CudaBuffer(size_t size_in_bytes) : _bytes(size_in_bytes) {
    CUDA_CALL_THROW(cudaMalloc(&_data, _bytes));
  }

  template <typename T = void>
  T* data() {
    return reinterpret_cast<T*>(_data);
  }

  void to_cpu(void* dst) {
    CUDA_CALL_THROW(cudaMemcpy(dst, _data, _bytes, cudaMemcpyDeviceToHost));
  }

  void from_cpu(void* src) {
    CUDA_CALL_THROW(cudaMemcpy(_data, src, _bytes, cudaMemcpyHostToDevice));
  }

  ~CudaBuffer() {
    EXPECT_TRUE(CUDA_CALL(cudaFree(_data)).IsOK());
  }
};

TEST(CudaBufferTest, RoundTrip) {
  float expected[] = {1.f, -2.f, 3.f};
  float actual[3] = {};
  CudaBuffer buffer(sizeof(expected));
  buffer.from_cpu(expected);
  buffer.to_cpu(actual);
  for (size_t i = 0; i < 3; ++i) {
    EXPECT_FLOAT_EQ(actual[i], expected[i]);
  }
}

TEST(CudaBufferTest, AllocationFailureIsReported) {
  EXPECT_THROW(CudaBuffer{std::numeric_limits<size_t>::max()}, onnxruntime::OnnxRuntimeException);
}

template <typename T>
float compare(void* a, void* b, size_t size, float scale) {
  auto pa = reinterpret_cast<T*>(a);
  auto pb = reinterpret_cast<T*>(b);
  float max_diff = 0.f;
  float total_diff = 0.f;
  float max_val = 0.f;
  int diff_count = 0;
  float threshold = 1e-7f;
  for (size_t n = 0; n < size; ++n) {
    float va = static_cast<float>(pa[n]);
    float vb = static_cast<float>(pb[n]);
    max_val = std::max(max_val, vb);
    float diff = std::abs(va - vb);
    if (diff > threshold) {
      max_diff = std::max(max_diff, diff);
      total_diff += diff;
      ++diff_count;
    }
  }

  float diff_threshold = max_val * scale;
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    // fp16 precision is about 3.3 decimal digits, and bf16 is about 2.0–2.3 decimal digits, so we use 10x threshold.
    diff_threshold *= 15.f;
  } else {
    diff_threshold *= 1.5f;
  }

  bool passed = max_diff <= diff_threshold;
  if (!passed) {
    printf("max diff %f (threshold %f), avg diff %f, diff count %d/%zu\n",
           max_diff, diff_threshold, total_diff / diff_count, diff_count, size);
  }

  return max_diff <= diff_threshold;
}

template <typename T1, typename T2>
void random_fill(std::vector<T1>& vec, T2 min_value, T2 max_value) {
  std::mt19937 gen(rand());
  std::uniform_real_distribution<float> dis(static_cast<float>(min_value), static_cast<float>(max_value));
  for (auto& v : vec) {
    v = static_cast<T1>(dis(gen));
  }
}

std::vector<CutlassGemmConfig> filter_gemm_configs(const std::vector<CutlassGemmConfig>& configs, int k) {
  std::vector<CutlassGemmConfig> rets;
  for (auto config : configs) {
    if (config.stages >= 5) {
      continue;
    }

    if (config.split_k_style != onnxruntime::llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K) {
      int k_size = (k + config.split_k_factor - 1) / config.split_k_factor;
      if (k_size % 64) {
        continue;
      }
    }
    rets.push_back(config);
  }
  return rets;
}

template <wo::KernelType KT>
struct cutlassTypeMapper {
};

#define CUTLASS_TYPE_MAPPER_REGISTRY(                                                           \
    CudaKernelType, CudaAType, CutlassWType, WElemBits, CutlassQuantOp)                         \
  template <>                                                                                   \
  struct cutlassTypeMapper<CudaKernelType> {                                                    \
    using AType = CudaAType;                                                                    \
    using WType = CutlassWType;                                                                 \
    static constexpr cutlass::WeightOnlyQuantOp QuantOp = CutlassQuantOp;                       \
    static constexpr int WSizeInBits = WElemBits;                                               \
    static std::string ATypeStr() { return std::is_same_v<CudaAType, half> ? "Fp16" : "BF16"; } \
    static std::string WTypeStr() {                                                             \
      return WSizeInBits == 2 ? "Int2" : (WSizeInBits == 4 ? "Int4" : "Int8");                  \
    }                                                                                           \
  };

#if USE_COMPACT_FPA_INTB_GEMM
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int8Groupwise, half, uint8_t, 8,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int4Groupwise, half, cutlass::uint4b_t, 4,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int8Groupwise, __nv_bfloat16, uint8_t, 8,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int4Groupwise, __nv_bfloat16, cutlass::uint4b_t, 4,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int2Groupwise, half, cutlass::uint2b_t, 2,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int2Groupwise, __nv_bfloat16, cutlass::uint2b_t, 2,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY);
#else
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int8Groupwise, half, uint8_t, 8,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int8Groupwise, __nv_bfloat16, uint8_t, 8,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int4Groupwise, half, cutlass::uint4b_t, 4,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int4Groupwise, __nv_bfloat16, cutlass::uint4b_t, 4,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int2Groupwise, half, cutlass::uint2b_t, 2,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int2Groupwise, __nv_bfloat16, cutlass::uint2b_t, 2,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
#endif

template <typename Func>
float measure_kernel_time(Func kernel_launcher, int warmup, int repeats, cudaStream_t s) {
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  for (int i = 0; i < warmup; ++i) {
    kernel_launcher();
  }
  cudaEventRecord(begin, s);
  for (int i = 0; i < repeats; ++i) {
    kernel_launcher();
  }
  cudaEventRecord(end, s);
  cudaEventSynchronize(end);
  float time;
  cudaEventElapsedTime(&time, begin, end);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  return time / repeats;
}

template <wo::KernelType KT, cutlass::WeightOnlyQuantOp QuantOp = cutlassTypeMapper<KT>::QuantOp,
          typename Runner, typename Config>
void run_cutlass_kernel([[maybe_unused]] void* scaled_act, Runner& runner, wo::Params& params, Config& config,
                        char* ws, size_t ws_size, cudaStream_t stream) {
  void* act = params.act;
  if (params.act_scale) {
    ORT_THROW("act_scale is not supported in this test fixture.");
  }
  if (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
    runner.gemm(act, params.weight, params.scales, params.zeros, params.bias, params.out, params.m, params.n,
                params.k, params.groupsize, config, ws, ws_size, stream);
  } else if (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY) {
    runner.gemm(act, params.weight, params.scales, nullptr, params.bias, params.out, params.m, params.n,
                params.k, params.groupsize, config, ws, ws_size, stream);
  }
}

struct BenchmarkResult {
  std::string a_type;
  std::string b_type;
  int m;
  int n;
  int k;
  int block_size;
  float cuda_time_us;
  float cutlass_time_us;
  float nbits_time_us;
  float naive_time_us;
  float speedup_cuda_vs_cutlass;
  float speedup_cuda_vs_nbits;
  float speedup_best_vs_naive;

  float best_time_us() const {
    float best = cutlass_time_us;
    if (cuda_time_us > 0.f) {
      best = std::min(best, cuda_time_us);
    }
    if (nbits_time_us > 0.f) {
      best = std::min(best, nbits_time_us);
    }
    return best;
  }
};

void PrintBenchmarkSummary(std::vector<BenchmarkResult>& benchmark_results) {
  std::cout << "\nBenchmark of FpA_IntB_GEMV, FpA_IntB_GEMM, MatMulNBits, Naive (DQ + GEMM) kernels (latency in microseconds):\n";
  constexpr size_t kLength = 139;
  std::cout << std::string(kLength, '-') << std::endl;
  std::cout << std::left << std::setw(6) << "A"
            << std::setw(6) << "W"
            << std::setw(6) << "m"
            << std::setw(8) << "n"
            << std::setw(7) << "k"
            << std::setw(12) << "block_size"
            << std::setw(12) << "gemv (us)"
            << std::setw(12) << "gemm (us)"
            << std::setw(12) << "nbits (us)"
            << std::setw(12) << "best (us)"
            << std::setw(12) << "naive (us)"
            << std::setw(12) << "gemm/gemv"
            << std::setw(12) << "nbits/gemv"
            << std::setw(12) << "best/naive"
            << std::endl;
  std::cout << std::string(kLength, '-') << std::endl;

  std::cout << std::fixed << std::setprecision(3);

  for (const auto& result : benchmark_results) {
    std::cout << std::left << std::setw(6) << result.a_type
              << std::setw(6) << result.b_type
              << std::setw(6) << result.m
              << std::setw(8) << result.n
              << std::setw(7) << result.k
              << std::setw(12) << result.block_size
              << std::setw(12) << result.cuda_time_us
              << std::setw(12) << result.cutlass_time_us
              << std::setw(12) << result.nbits_time_us
              << std::setw(12) << result.best_time_us()
              << std::setw(12) << result.naive_time_us
              << std::setw(12) << result.speedup_cuda_vs_cutlass
              << std::setw(12) << result.speedup_cuda_vs_nbits
              << std::setw(12) << result.speedup_best_vs_naive
              << std::endl;
  }
  std::cout << std::string(kLength, '-') << std::endl;
}

template <wo::KernelType KT, bool has_bias = false, bool has_act_scale = false, bool filter_configs = false,
          cutlass::WeightOnlyQuantOp QuantOp = cutlassTypeMapper<KT>::QuantOp>
class KernelTestFixture : public ::testing::Test {
 protected:
  int m_, n_, k_, block_size_;
  int warmup_ = 10;
  int repeats_ = 30;
  cudaDeviceProp device_prop_;
  std::shared_ptr<CudaBuffer> d_act_, d_act_scale_, d_weight_, d_scales_, d_zeros_, d_bias_, d_out_;
  std::vector<typename cutlassTypeMapper<KT>::AType> h_act_, h_act_scale_, h_scales_, h_zeros_, h_bias_, h_out1_, h_out2_;
  std::vector<uint8_t> h_weight_;
  std::vector<BenchmarkResult> benchmark_results_;
  cudaStream_t s_ = nullptr;
  cublasHandle_t cublas_handle_ = nullptr;

  static constexpr int WSizeInBits = cutlassTypeMapper<KT>::WSizeInBits;
  static constexpr bool kIsInt2 = (WSizeInBits == 2);
  static constexpr bool kUseSm80Layout = kIsInt2 || QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;

  static constexpr int GetKernelArch([[maybe_unused]] int device_arch) {
#if USE_COMPACT_FPA_INTB_GEMM
    return 80;
#else
    return kUseSm80Layout ? 80 : device_arch;
#endif
  }

  void SetUp() override {
    int device;
    CUDA_CALL_THROW(cudaGetDevice(&device));
    CUDA_CALL_THROW(cudaGetDeviceProperties(&device_prop_, device));
    std::srand(20240123);
    CUDA_CALL_THROW(cudaStreamCreate(&s_));
    CUBLAS_CALL_THROW(cublasCreate(&cublas_handle_));
    CUBLAS_CALL_THROW(cublasSetStream(cublas_handle_, s_));
  }

  void TearDown() override {
    PrintBenchmarkSummary(benchmark_results_);
    if (s_ != nullptr) {
      EXPECT_TRUE(CUDA_CALL(cudaStreamDestroy(s_)).IsOK());
    }
    if (cublas_handle_ != nullptr) {
      EXPECT_TRUE(CUBLAS_CALL(cublasDestroy(cublas_handle_)).IsOK());
    }
  }

  void InitBuffers(int m, int n, int k, int block_size) {
    m_ = m;
    n_ = n;
    k_ = k;
    block_size_ = block_size;

    if (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
      ORT_ENFORCE(block_size_ == 64 || block_size_ == 128);
      ORT_ENFORCE(k_ % block_size_ == 0);
    } else if (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY) {
      ORT_ENFORCE(block_size_ == (kIsInt2 ? 64 : 32));
      ORT_ENFORCE(k_ % block_size_ == 0);
    }

    using AType = typename cutlassTypeMapper<KT>::AType;

    constexpr size_t ATypeBytes = sizeof(AType);
    const size_t m_x_k = static_cast<size_t>(m_) * static_cast<size_t>(k_);
    const size_t n_x_k = static_cast<size_t>(n_) * static_cast<size_t>(k_);
    const size_t m_x_n = static_cast<size_t>(m_) * static_cast<size_t>(n_);
    d_act_ = std::make_shared<CudaBuffer>(m_x_k * ATypeBytes);
    d_act_scale_ = std::make_shared<CudaBuffer>(static_cast<size_t>(k_) * ATypeBytes);
    d_weight_ = std::make_shared<CudaBuffer>(n_x_k * WSizeInBits / static_cast<size_t>(8));
    d_scales_ = std::make_shared<CudaBuffer>(n_x_k / static_cast<size_t>(block_size_) * ATypeBytes);
    d_zeros_ = std::make_shared<CudaBuffer>(n_x_k / static_cast<size_t>(block_size_) * ATypeBytes);
    d_bias_ = std::make_shared<CudaBuffer>(static_cast<size_t>(n_) * ATypeBytes);
    d_out_ = std::make_shared<CudaBuffer>(m_x_n * ATypeBytes);

    h_act_.resize(m_x_k);
    h_act_scale_.resize(static_cast<size_t>(k_));
    h_weight_.resize(n_x_k);
    h_scales_.resize(n_x_k / static_cast<size_t>(block_size_));
    h_zeros_.resize(n_x_k / static_cast<size_t>(block_size_));
    h_bias_.resize(static_cast<size_t>(n_));
    h_out1_.resize(m_x_n);
    h_out2_.resize(m_x_n);

    random_fill(h_act_, -1.f, 1.f);
    random_fill(h_act_scale_, -1.f, 1.f);
    random_fill(h_scales_, -1.f, 1.f);
    random_fill(h_zeros_, -1.f, 1.f);
    random_fill(h_bias_, -1.f, 1.f);

    for (uint8_t& v : h_weight_) {
      v = rand() % 256;
    }

    d_act_->from_cpu(h_act_.data());
    d_act_scale_->from_cpu(h_act_scale_.data());
    d_weight_->from_cpu(h_weight_.data());
    d_scales_->from_cpu(h_scales_.data());
    d_zeros_->from_cpu(h_zeros_.data());
    d_bias_->from_cpu(h_bias_.data());
  }

  bool BenchmarkAndVerifyKernel(bool use_zero_points = QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS,
                                int decode_variant = 0) {
    std::cout << "m=" << m_ << ", n=" << n_ << ", k=" << k_ << ", block_size=" << block_size_ << std::endl;

    void* p_act_scale = nullptr;
    void* p_zeros = nullptr;
    void* p_bias = nullptr;

    if (block_size_ != 0) {
      p_zeros = use_zero_points ? d_zeros_->data() : nullptr;
      if constexpr (has_bias) {
        p_bias = d_bias_->data();
      }
      if constexpr (has_act_scale) {
        p_act_scale = d_act_scale_->data();
      }
    }

    wo::Params params(d_act_->data(), p_act_scale, d_weight_->data(), d_scales_->data(), p_zeros, p_bias,
                      d_out_->data(), 1.f, m_, n_, k_, block_size_, KT);
    params.decode_variant = decode_variant;

    //------------------------
    // Run FpA_IntB_Gemv CUDA kernel
    float cuda_time_ms = 0.f;
    if (m_ < 16) {
      cuda_time_ms = measure_kernel_time(
          [&]() {
            const int device_arch = onnxruntime::llm::common::getSMVersion();
            const int kernel_arch = GetKernelArch(device_arch);
            ORT_ENFORCE(wo::is_supported(device_arch, kernel_arch, params.type));
            wo::kernel_launcher(kernel_arch, params, s_);
          },
          warmup_, repeats_, s_);
      d_out_->to_cpu(h_out1_.data());
    }

    // ------------------------
    // Run FpA_IntB_Gemm CUTLASS kernel
    using AType = typename cutlassTypeMapper<KT>::AType;
    using WType = typename cutlassTypeMapper<KT>::WType;
    using onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner;
    auto runner = std::make_shared<CutlassFpAIntBGemmRunner<AType, WType, QuantOp>>();
#if USE_COMPACT_FPA_INTB_GEMM
    int const arch = onnxruntime::llm::common::getSMVersion();
    runner->setArch(arch < 80 ? arch : (arch == 89 ? 89 : 80));
#else
    const int device_arch = onnxruntime::llm::common::getSMVersion();
    const int kernel_arch = GetKernelArch(device_arch);
    runner->setArch(!kIsInt2 && device_arch < 80 ? device_arch : kernel_arch);
    runner->setUseSm90Native(kernel_arch == 90);
#endif
    auto& gemm_runner = *runner;
    const size_t ws_bytes = gemm_runner.getWorkspaceSize(m_, n_, k_);
    CudaBuffer ws_buffer(ws_bytes);
    char* ws_ptr = reinterpret_cast<char*>(ws_buffer.data());

    auto configs = gemm_runner.getConfigs();

    if constexpr (filter_configs) {
      configs = filter_gemm_configs(configs, k_);
    }

    float fast_time_ms = std::numeric_limits<float>::max();
    CutlassGemmConfig best_config = configs[0];

    for (auto& config : configs) {
      float time = std::numeric_limits<float>::max();
      try {
        time = measure_kernel_time(
            [&]() {
              run_cutlass_kernel<KT, QuantOp>(d_act_->data(), gemm_runner, params, config, ws_ptr, ws_bytes, s_);
            },
            2, 5, s_);
      } catch (std::exception const& e) {
        std::ostringstream msg;
        msg << "Failed to profile m=" << params.m << ", n=" << params.n << ", k=" << params.k << "for configuration:\n";
        msg << config.toString();
        msg << "\nException:" << e.what() << "\n";
        std::cout << msg.str();
        cudaGetLastError();  // Reset the last cudaError to cudaSuccess.
        continue;
      }
      if (time < fast_time_ms) {
        fast_time_ms = time;
        best_config = config;
      }
    }

    float cutlass_time_ms = measure_kernel_time(
        [&]() {
          run_cutlass_kernel<KT, QuantOp>(d_act_->data(), gemm_runner, params, best_config, ws_ptr, ws_bytes, s_);
        },
        warmup_, repeats_, s_);
    d_out_->to_cpu(h_out2_.data());

    // ------------------------
    // Compare FpA_IntB_Gemv and FpA_IntB_Gemm outputs.
    bool pass = true;
    if (m_ < 16) {
      float quant_scale = 1.f / (1 << (WSizeInBits - 1));
      const size_t m_x_n = static_cast<size_t>(m_) * static_cast<size_t>(n_);
      pass = compare<AType>(h_out1_.data(), h_out2_.data(), m_x_n, quant_scale);
    }

    // ------------------------
    // Run MatMulNBits kernel.
    // Note that it runs on random data, so the output is not compared.
    float nbits_time_ms = 0.f;
    float naive_time_ms = 0.f;
    if constexpr (KT == wo::KernelType::FP16Int8Groupwise || KT == wo::KernelType::FP16Int4Groupwise ||
                  KT == wo::KernelType::FP16Int2Groupwise) {
      const size_t n_x_k = static_cast<size_t>(n_) * static_cast<size_t>(k_);
      const size_t zero_point_bytes_per_column =
          (static_cast<size_t>(k_ / block_size_) * WSizeInBits + 7) / 8;
      std::vector<uint8_t> h_uint8_zeros(static_cast<size_t>(n_) * zero_point_bytes_per_column);
      for (uint8_t& v : h_uint8_zeros) {
        v = rand() % 256;
      }

      CudaBuffer d_uint8_zeros(h_uint8_zeros.size());
      d_uint8_zeros.from_cpu(h_uint8_zeros.data());

      if (m_ == 1) {
        nbits_time_ms = measure_kernel_time(
            [&]() {
              onnxruntime::contrib::cuda::TryMatMulNBits(WSizeInBits,
                                                         reinterpret_cast<AType*>(d_out_->data()),
                                                         reinterpret_cast<const AType*>(d_act_->data()),
                                                         reinterpret_cast<const uint8_t*>(d_weight_->data()),
                                                         reinterpret_cast<const AType*>(d_scales_->data()),
                                                         static_cast<const uint8_t*>(d_uint8_zeros.data()),
                                                         static_cast<const AType*>(nullptr),
                                                         m_, n_, k_, block_size_, device_prop_.sharedMemPerBlock, s_,
                                                         device_prop_.major * 10 + device_prop_.minor);
            },
            warmup_, repeats_, s_);
      }

      CudaBuffer d_dequantized_weight(n_x_k * sizeof(AType));

      naive_time_ms = measure_kernel_time(
          [&]() {
            auto status = onnxruntime::contrib::cuda::DequantizeNBits<AType, uint8_t>(
                WSizeInBits,
                reinterpret_cast<AType*>(d_dequantized_weight.data()),
                reinterpret_cast<const uint8_t*>(d_weight_->data()),
                reinterpret_cast<const AType*>(d_scales_->data()),
                reinterpret_cast<const uint8_t*>(d_uint8_zeros.data()),
                nullptr,
                k_,
                n_,
                block_size_,
                s_);

            ORT_THROW_IF_ERROR(status);

            const AType alpha = AType(1.f);
            const AType zero = AType(0.f);
            constexpr bool use_tf32 = false;
            CUBLAS_CALL_THROW(cublasGemmHelper(
                cublas_handle_,
                CUBLAS_OP_T,
                CUBLAS_OP_N,
                n_,
                m_,
                k_,
                &alpha,
                reinterpret_cast<const AType*>(d_dequantized_weight.data()),
                k_,
                reinterpret_cast<const AType*>(d_act_->data()),
                k_,
                &zero,
                reinterpret_cast<AType*>(d_out_->data()),
                n_,
                device_prop_,
                use_tf32));
          },
          warmup_, repeats_, s_);
    }

    // Store benchmark results
    BenchmarkResult result;
    result.a_type = cutlassTypeMapper<KT>::ATypeStr();
    result.b_type = cutlassTypeMapper<KT>::WTypeStr();
    result.m = m_;
    result.n = n_;
    result.k = k_;
    result.block_size = block_size_;
    result.cuda_time_us = cuda_time_ms * 1000.0f;
    result.cutlass_time_us = cutlass_time_ms * 1000.0f;
    result.nbits_time_us = nbits_time_ms * 1000.0f;
    result.naive_time_us = naive_time_ms * 1000.0f;
    result.speedup_cuda_vs_cutlass = cuda_time_ms > 0.f ? cutlass_time_ms / cuda_time_ms : 0.f;
    result.speedup_cuda_vs_nbits = cuda_time_ms > 0.f ? nbits_time_ms / cuda_time_ms : 0.f;
    result.speedup_best_vs_naive = result.naive_time_us / result.best_time_us();
    benchmark_results_.push_back(result);

    return pass;
  }
};

}  // namespace

#if USE_COMPACT_FPA_INTB_GEMM
using Fp16Int8GroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int8Groupwise, false, false, true>;
using Fp16Int4GroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int4Groupwise, false, false, true>;
using Bf16Int8GroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int8Groupwise, false, false, true>;
using Bf16Int4GroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int4Groupwise, false, false, true>;
using Fp16Int2GroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int2Groupwise, false, false, true>;
using Bf16Int2GroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int2Groupwise, false, false, true>;
#else
using Fp16Int8GroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int8Groupwise>;
using Fp16Int4GroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int4Groupwise>;
using Bf16Int8GroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int8Groupwise>;
using Bf16Int4GroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int4Groupwise>;
using Fp16Int2GroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int2Groupwise>;
using Bf16Int2GroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int2Groupwise>;
#endif

using Fp16Int4SymmetricGroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int4Groupwise, false, false, true,
                                                         cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>;
using Bf16Int4SymmetricGroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int4Groupwise, false, false, true,
                                                         cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>;

TEST(FpAIntBGemvTest, SupportUsesDeviceAndKernelArchitectures) {
  EXPECT_FALSE(wo::is_supported(74, 80, wo::KernelType::FP16Int4Groupwise));
  EXPECT_TRUE(wo::is_supported(75, 80, wo::KernelType::FP16Int4Groupwise));
  EXPECT_FALSE(wo::is_supported(75, 80, wo::KernelType::BF16Int4Groupwise));
  EXPECT_FALSE(wo::is_supported(90, 80, wo::KernelType::FP16Int4PerChannel));

#if USE_COMPACT_FPA_INTB_GEMM
  EXPECT_TRUE(wo::is_supported(90, 80, wo::KernelType::FP16Int4Groupwise));
  EXPECT_TRUE(wo::is_supported(90, 80, wo::KernelType::BF16Int4Groupwise));
  EXPECT_TRUE(wo::is_supported(80, 80, wo::KernelType::BF16Int2Groupwise));
  EXPECT_TRUE(wo::is_supported(90, 80, wo::KernelType::BF16Int2Groupwise));
  EXPECT_FALSE(wo::is_supported(90, 90, wo::KernelType::FP16Int4Groupwise));
  EXPECT_FALSE(wo::is_supported(90, 90, wo::KernelType::BF16Int4Groupwise));
#else
  EXPECT_TRUE(wo::is_supported(80, 80, wo::KernelType::BF16Int4Groupwise));
  EXPECT_TRUE(wo::is_supported(90, 80, wo::KernelType::BF16Int4Groupwise));
  EXPECT_FALSE(wo::is_supported(80, 90, wo::KernelType::FP16Int4Groupwise));
  // 2-bit shares the SM80 layout but has no native Hopper instantiation.
  EXPECT_TRUE(wo::is_supported(80, 80, wo::KernelType::FP16Int2Groupwise));
  EXPECT_TRUE(wo::is_supported(90, 80, wo::KernelType::BF16Int2Groupwise));
  EXPECT_FALSE(wo::is_supported(90, 90, wo::KernelType::FP16Int2Groupwise));
#ifdef EXCLUDE_SM_90
  EXPECT_FALSE(wo::is_supported(90, 90, wo::KernelType::FP16Int4Groupwise));
#else
  EXPECT_TRUE(wo::is_supported(90, 90, wo::KernelType::FP16Int4Groupwise));
#endif
#endif
}

// Keep profile choices separate across devices, layouts, dtypes, quantization, and fused features.
TEST(FpAIntBGemvTest, TacticCacheSeparatesDeviceAndQuantization) {
  using TacticCache = std::unordered_map<wo_profile::GemmIdCore, int, wo_profile::GemmIdCoreHash>;
  const wo_profile::GemmIdCore base(384, 1536, onnxruntime::llm::nvinfer::DataType::kHALF,
                                    80, false, 0, 0, 4, 32, false, false);
  std::vector<wo_profile::GemmIdCore> ids{base};
  for (int field = 0; field < 7; ++field) {
    auto id = base;
    switch (field) {
      case 0:
        id.device_id = 1;
        break;
      case 1:
        id.sm = 90;
        break;
      case 2:
        id.dtype = onnxruntime::llm::nvinfer::DataType::kBF16;
        break;
      case 3:
        id.quant_bits = 8;
        break;
      case 4:
        id.group_size = 64;
        break;
      case 5:
        id.has_bias = true;
        break;
      case 6:
        id.has_zeros = true;
        break;
    }
    EXPECT_FALSE(id == base);
    ids.push_back(id);
  }
  TacticCache cache;
  for (size_t index = 0; index < ids.size(); ++index) cache.emplace(ids[index], static_cast<int>(index));
  EXPECT_EQ(cache.size(), ids.size());
  for (size_t index = 0; index < ids.size(); ++index) EXPECT_EQ(cache.at(ids[index]), static_cast<int>(index));
}

// Allocate a bounded rotating weight/scale set beyond L2, retaining the original scratch fallback.
TEST(Int4DecodeTileTest, StreamingScratchSize) {
  constexpr size_t cache_bytes = 64 * 1024;
  const auto warm = wo_profile::ComputeWeightOnlyGemmProfilerBufferSizes(1, 16, 64, 4, 32, 0);
  const auto streaming = wo_profile::ComputeWeightOnlyGemmProfilerBufferSizes(1, 16, 64, 4, 32, 0, cache_bytes);
  ASSERT_TRUE(warm.has_value());
  ASSERT_TRUE(streaming.has_value());
  EXPECT_GT((*streaming)[1] + (*streaming)[2], 2 * cache_bytes);
  EXPECT_LE((*streaming)[1] + (*streaming)[2], 2 * cache_bytes + (*warm)[1] + (*warm)[2]);
  EXPECT_EQ((*streaming)[1] / (*warm)[1], (*streaming)[2] / (*warm)[2]);
  EXPECT_EQ((*streaming)[3], (*warm)[3]);
  const auto scratch = wo_profile::ComputeWeightOnlyGemmProfilerScratchSize(1, 16, 64, 4, 32, 0, cache_bytes);
  ASSERT_TRUE(scratch.has_value());
  EXPECT_GE(*scratch, (*streaming)[1] + (*streaming)[2]);
  EXPECT_EQ(wo_profile::ComputeWeightOnlyGemmProfilerBufferSizes(1, 4096, 4096, 4, 32, 0, cache_bytes),
            wo_profile::ComputeWeightOnlyGemmProfilerBufferSizes(1, 4096, 4096, 4, 32, 0));
  EXPECT_FALSE(wo_profile::ComputeWeightOnlyGemmProfilerScratchSize(
                   1, 16, 64, 4, 32, 0, std::numeric_limits<size_t>::max())
                   .has_value());
}

// Enumerate legal geometries and reject production misalignment, incompatible layout, and overflowing grids.
TEST(Int4DecodeTileTest, GeometryAndAlignment) {
  for (const auto& [variant, tile, threads] : std::vector<std::tuple<int, int, int>>{
           {2, 2, 128}, {3, 2, 256}, {4, 4, 128}, {5, 4, 256}, {6, 8, 128}, {7, 8, 256}}) {
    EXPECT_EQ(wo::GetInt4DecodeGeometry(variant), std::make_pair(tile, threads));
    EXPECT_TRUE(wo::IsInt4DecodeGeometryLegal(variant, 1536, 12352, 4));
    EXPECT_FALSE(wo::IsInt4DecodeGeometryLegal(variant, 32, 256, 4));
    EXPECT_FALSE(wo::IsInt4DecodeGeometryLegal(variant, 1536, 32, 4));
    EXPECT_FALSE(wo::IsInt4DecodeGeometryLegal(variant, 524296, 256, 4));
    EXPECT_FALSE(wo::IsInt4DecodeGeometryLegal(variant, 1536, 96, 4));
    EXPECT_FALSE(wo::IsInt4DecodeGeometryLegal(variant, 1536, 1536, 2));
    EXPECT_TRUE(wo::IsInt4DecodeGeometryLegal(variant, tile * 4 * 65535 - (tile * 4 * 65535) % 64, 256, 4));
    EXPECT_FALSE(wo::IsInt4DecodeGeometryLegal(variant, tile * 4 * 65536, 256, 4));
  }
  for (int variant : {0, 1, 8}) EXPECT_EQ(wo::GetInt4DecodeGeometry(variant), std::make_pair(0, 0));
}

// Select by explicit packing capability rather than an architecture identifier; retain default and CUTLASS tactics.
TEST(Int4DecodeTileTest, ProfilerEligibilityAndFallback) {
  struct Profiler : wo_profile::WeightOnlyGroupwiseQuantGemmPluginProfiler {
    explicit Profiler(WeightOnlyGemmRunnerPtr runner) { mRunner = std::move(runner); }
    using WeightOnlyGroupwiseQuantGemmPluginProfiler::checkTactic;
    using WeightOnlyGroupwiseQuantGemmPluginProfiler::getTactics;
  };
  using Runner = onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
      half, cutlass::uint4b_t, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>;
  auto runner = std::make_shared<Runner>();
  runner->setArch(80);
  Profiler profiler(runner);
  profiler.setQuant(4, false, false);
  profiler.setGroupSize(32);
  profiler.setDecodeInterleave(4);
  for (auto type : {wo::KernelType::FP16Int4Groupwise, wo::KernelType::BF16Int4Groupwise}) {
    for (int arch : {75, 80, 86, 89, 90, 100, 120}) {
      profiler.setCudaKernelType(type, arch);
      std::set<int> variants;
      for (const auto& tactic : profiler.getTactics(1, 384, 1536)) {
        if (tactic.enableCudaKernel) variants.insert(tactic.cudaKernelVariant);
      }
      EXPECT_EQ(variants, (std::set<int>{0, 2, 3, 4, 5, 6, 7}));
      for (int variant = 2; variant <= 7; ++variant) {
        CutlassGemmConfig tactic;
        tactic.enableCudaKernel = true;
        tactic.cudaKernelVariant = variant;
        EXPECT_TRUE(profiler.checkTactic(1, 384, 12352, tactic));
        EXPECT_FALSE(profiler.checkTactic(2, 384, 12352, tactic));
        for (int interleave : {0, 1, 2, 8}) {
          profiler.setDecodeInterleave(interleave);
          EXPECT_FALSE(profiler.checkTactic(1, 384, 12352, tactic));
        }
        profiler.setDecodeInterleave(4);
      }
    }
    CutlassGemmConfig tactic;
    tactic.enableCudaKernel = true;
    tactic.cudaKernelVariant = 2;
    EXPECT_FALSE(profiler.checkTactic(1, 131072, 256, tactic));
    profiler.setGroupSize(64);
    EXPECT_FALSE(profiler.checkTactic(1, 384, 1536, tactic));
    profiler.setGroupSize(32);
    for (const auto& [bits, bias, zeros] : std::vector<std::tuple<int, bool, bool>>{
             {8, false, false}, {2, false, false}, {4, true, false}, {4, false, true}}) {
      profiler.setQuant(bits, bias, zeros);
      EXPECT_FALSE(profiler.checkTactic(1, 384, 1536, tactic));
    }
    profiler.setQuant(4, false, false);
    tactic.cudaKernelVariant = 8;
    EXPECT_FALSE(profiler.checkTactic(1, 384, 1536, tactic));
    std::set<int> wide_variants;
    for (const auto& candidate : profiler.getTactics(1, 131072, 256)) {
      if (candidate.enableCudaKernel) wide_variants.insert(candidate.cudaKernelVariant);
    }
    EXPECT_EQ(wide_variants, (std::set<int>{0, 4, 5, 6, 7}));
  }

  // Upstream opt-in modes must not suppress M=1 candidates or admit them at M=5..8.
  profiler.setCudaKernelType(wo::KernelType::FP16Int4Groupwise, 80);
  for (bool wave_aware : {false, true}) {
    profiler.setWaveAwareGemv(wave_aware);
    for (int mode : {0, 1, 2}) {
      profiler.setPairedGemvMode(mode);
      std::set<int> decode_variants;
      bool has_cutlass = false;
      for (const auto& tactic : profiler.getTactics(1, 384, 1536)) {
        if (tactic.enableCudaKernel)
          decode_variants.insert(tactic.cudaKernelVariant);
        else
          has_cutlass = true;
      }
      EXPECT_EQ(decode_variants, (std::set<int>{0, 2, 3, 4, 5, 6, 7}));
      EXPECT_TRUE(has_cutlass);
      for (int rows : {5, 6, 7, 8}) {
        const auto tactics = profiler.getTactics(rows, 384, 1536);
        for (const auto& tactic : tactics) {
          if (tactic.enableCudaKernel) {
            EXPECT_LE(tactic.cudaKernelVariant, 1);
            EXPECT_TRUE(profiler.checkTactic(rows, 384, 1536, tactic));
          }
        }
        if (mode == 2) {
          ASSERT_EQ(tactics.size(), 1u);
          EXPECT_EQ(tactics[0].cudaKernelVariant, 1);
        }
      }
    }
  }
}

// Preserve upstream wave-aware tile selection and its M=8-only scope.
TEST(FpAIntBGemvTest, WaveAwareDispatchUsesSyntheticSmCount) {
  constexpr int kRtx5090SmCount = 170;
  constexpr int kInterleave = 4;
  constexpr int kDefaultCtaN = 4;

  for (int n : {512, 10240}) {
    EXPECT_EQ(wo::PickGemvCtaN(true, 8, n, kInterleave, kDefaultCtaN, kRtx5090SmCount), 2);
    EXPECT_EQ(wo::PickGemvCtaN(false, 8, n, kInterleave, kDefaultCtaN, kRtx5090SmCount), kDefaultCtaN);
    EXPECT_EQ(wo::PickGemvCtaN(true, 7, n, kInterleave, kDefaultCtaN, kRtx5090SmCount), kDefaultCtaN);
    EXPECT_EQ(wo::PickGemvCtaN(true, 9, n, kInterleave, kDefaultCtaN, kRtx5090SmCount), kDefaultCtaN);
  }
  EXPECT_EQ(wo::PickGemvCtaN(true, 8, 512, kInterleave, kDefaultCtaN, 0), kDefaultCtaN);
}

// Keep upstream wave-aware and default tactic caches distinct after extending the key.
TEST(FpAIntBGemvTest, TacticCacheSeparatesWaveAwareMode) {
  using TacticCache = std::unordered_map<wo_profile::GemmIdCore, int, wo_profile::GemmIdCoreHash>;
  wo_profile::GemmIdCore const default_id(10240, 4096, onnxruntime::llm::nvinfer::DataType::kHALF, 80, false);
  wo_profile::GemmIdCore const wave_aware_id(10240, 4096, onnxruntime::llm::nvinfer::DataType::kHALF, 80, true);
  TacticCache cache{{default_id, 4}, {wave_aware_id, 2}};

  ASSERT_EQ(cache.size(), 2u);
  EXPECT_EQ(cache.at(default_id), 4);
  EXPECT_EQ(cache.at(wave_aware_id), 2);
}

// Preserve every paired-mode/wave-aware combination in the shared cache.
TEST(FpAIntBGemvTest, TacticCacheSeparatesPairedAndWaveAwareModes) {
  using Profiler = wo_profile::WeightOnlyGroupwiseQuantGemmPluginProfiler;
  auto cache = std::make_shared<Profiler::MNKProfileMap>();
  Profiler profiler;
  profiler.setSelectionTactics(cache);

  for (bool wave_aware : {false, true}) {
    for (int mode : {0, 1, 2}) {
      wo_profile::GemmIdCore const id(512, 1024, onnxruntime::llm::nvinfer::DataType::kHALF,
                                      80, wave_aware, mode);
      cache->createMProfileMap(id);
      CutlassGemmConfig tactic;
      tactic.enableCudaKernel = true;
      tactic.cudaKernelVariant = mode == 0 ? 0 : 1;
      tactic.stages = mode + (wave_aware ? 3 : 0);
      (*cache->getMProfileMap(id))[8] = tactic;
    }
  }

  ASSERT_EQ(cache->profileMap.size(), 6u);
  for (bool wave_aware : {false, true}) {
    for (int mode : {0, 1, 2}) {
      wo_profile::GemmIdCore const id(512, 1024, onnxruntime::llm::nvinfer::DataType::kHALF,
                                      80, wave_aware, mode);
      for (int m : {5, 6, 7, 8}) {
        auto const tactic = profiler.getBestConfig(m, id);
        ASSERT_TRUE(tactic.has_value());
        EXPECT_EQ(tactic->cudaKernelVariant, mode == 0 ? 0 : 1);
        EXPECT_EQ(tactic->stages, mode + (wave_aware ? 3 : 0));
      }
    }
  }
}

// Preserve upstream paired-K enumeration and force-mode behavior within M=5..8.
TEST(FpAIntBGemvTest, PairedTacticsFollowMRangeAndMode) {
  if (onnxruntime::llm::common::getSMVersion() < kMinSupportedSm) {
    GTEST_SKIP() << "fp16 int4 groupwise GEMV requires SM " << kMinSupportedSm << " or later";
  }

  class TestProfiler : public wo_profile::WeightOnlyGroupwiseQuantGemmPluginProfiler {
   public:
    explicit TestProfiler(WeightOnlyGemmRunnerPtr const& runner) {
      mRunner = runner;
    }
    using wo_profile::WeightOnlyGroupwiseQuantGemmPluginProfiler::checkTactic;
    using wo_profile::WeightOnlyGroupwiseQuantGemmPluginProfiler::getTactics;
  };

  using Runner = onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner<
      half, cutlass::uint4b_t, cutlassTypeMapper<wo::KernelType::FP16Int4Groupwise>::QuantOp>;
  auto runner = std::make_shared<Runner>();
  runner->setArch(80);
  TestProfiler profiler(runner);
  for (int mode : {0, 1, 2}) {
    profiler.setPairedGemvMode(mode);
    for (int m : {4, 5, 6, 7, 8, 9}) {
      auto const tactics = profiler.getTactics(m, 512, 1024);
      bool const eligible = mode != 0 && m >= 5 && m <= 8;
      EXPECT_EQ(std::count_if(tactics.begin(), tactics.end(),
                              [](auto const& tactic) { return tactic.cudaKernelVariant == 1; }),
                eligible ? 1 : 0);
      if (mode == 2 && eligible) {
        ASSERT_EQ(tactics.size(), 1u);
        EXPECT_TRUE(tactics[0].enableCudaKernel);
        EXPECT_TRUE(profiler.checkTactic(m, 512, 1024, tactics[0]));
      }
    }
  }
}

TEST_F(Fp16Int8GroupwiseTest, Fp16_Int8_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < kMinSupportedSm) {
    GTEST_SKIP() << "fp16 int8 groupwise GEMM kernel requires SM " << kMinSupportedSm << " or later";
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list(wo::KernelType::FP16Int8Groupwise)) {
      InitBuffers(m, n, k,
#if USE_COMPACT_FPA_INTB_GEMM
                  32
#else
                  64
#endif
      );
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

TEST_F(Fp16Int4GroupwiseTest, Fp16_Int4_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < kMinSupportedSm) {
    GTEST_SKIP() << "fp16 int4 groupwise GEMM kernel requires SM " << kMinSupportedSm << " or later";
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list(wo::KernelType::FP16Int4Groupwise)) {
      InitBuffers(m, n, k,
#if USE_COMPACT_FPA_INTB_GEMM
                  32
#else
                  64
#endif
      );
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

// Check FP16 default bounds and every legal profiled geometry against a symmetric CUTLASS reference.
TEST_F(Fp16Int4SymmetricGroupwiseTest, Int4Group32SymmetricM1Decode) {
  EXPECT_EQ(GetKernelArch(90), 80);
  if (onnxruntime::llm::common::getSMVersion() < kMinSupportedSm) {
    GTEST_SKIP() << "FP16 INT4 decode requires SM " << kMinSupportedSm << " or later";
  }
  for (const auto& [columns, depth] : std::vector<std::pair<int, int>>{
           {32, 256}, {64, 64}, {128, 256}, {256, 768}, {2880, 4096}, {8192, 256}, {8192, 2560}, {8192, 4096}, {8256, 256}, {128, 4160}}) {
    SCOPED_TRACE(testing::Message() << "N=" << columns << " K=" << depth);
    InitBuffers(1, columns, depth, 32);
    EXPECT_TRUE(BenchmarkAndVerifyKernel());
    for (int variant = 2; variant <= 7; ++variant) {
      if (wo::IsInt4DecodeGeometryLegal(variant, columns, depth, 4)) {
        EXPECT_TRUE(BenchmarkAndVerifyKernel(false, variant));
      }
    }
  }
}

// Check BF16 default bounds and every legal profiled geometry against the symmetric reference.
TEST_F(Bf16Int4SymmetricGroupwiseTest, Int4Group32SymmetricM1Decode) {
  EXPECT_EQ(GetKernelArch(90), 80);
  if (onnxruntime::llm::common::getSMVersion() < 80) {
    GTEST_SKIP() << "BF16 INT4 decode requires SM 80 or later";
  }
  for (const auto& [columns, depth] : std::vector<std::pair<int, int>>{
           {32, 256}, {64, 64}, {128, 256}, {256, 768}, {2880, 4096}, {8192, 256}, {8192, 2560}, {8192, 4096}, {8256, 256}, {128, 4160}}) {
    SCOPED_TRACE(testing::Message() << "N=" << columns << " K=" << depth);
    InitBuffers(1, columns, depth, 32);
    EXPECT_TRUE(BenchmarkAndVerifyKernel());
    for (int variant = 2; variant <= 7; ++variant) {
      if (wo::IsInt4DecodeGeometryLegal(variant, columns, depth, 4)) {
        EXPECT_TRUE(BenchmarkAndVerifyKernel(false, variant));
      }
    }
  }
}

// Check all FP16 decode geometries on short-K, long-K, and wide-N projections with a CUTLASS reference.
TEST_F(Fp16Int4SymmetricGroupwiseTest, Int4DecodeProfiledGeometries) {
  if (onnxruntime::llm::common::getSMVersion() < kMinSupportedSm) {
    GTEST_SKIP() << "FP16 decode requires SM " << kMinSupportedSm << " or later";
  }
  for (int variant = 2; variant <= 7; ++variant) {
    for (const auto& [columns, depth] : std::vector<std::pair<int, int>>{
             {128, 256}, {1536, 12288}, {12288, 1536}}) {
      SCOPED_TRACE(testing::Message() << "variant=" << variant << " N=" << columns << " K=" << depth);
      InitBuffers(1, columns, depth, 32);
      EXPECT_TRUE(BenchmarkAndVerifyKernel(false, variant));
    }
  }
}

// Check the same BF16 geometries on representative projections with a scale-only CUTLASS reference.
TEST_F(Bf16Int4SymmetricGroupwiseTest, Int4DecodeProfiledGeometries) {
  if (onnxruntime::llm::common::getSMVersion() < 80) {
    GTEST_SKIP() << "BF16 decode requires SM80 or later";
  }
  for (int variant = 2; variant <= 7; ++variant) {
    for (const auto& [columns, depth] : std::vector<std::pair<int, int>>{
             {128, 256}, {1536, 12288}, {12288, 1536}}) {
      SCOPED_TRACE(testing::Message() << "variant=" << variant << " N=" << columns << " K=" << depth);
      InitBuffers(1, columns, depth, 32);
      EXPECT_TRUE(BenchmarkAndVerifyKernel(false, variant));
    }
  }
}

// Preserve native Hopper routing for asymmetric fixtures while compact builds retain the SM80 layout.
TEST_F(Fp16Int4GroupwiseTest, Int4GroupwiseHopperRouting) {
#if USE_COMPACT_FPA_INTB_GEMM
  EXPECT_EQ(GetKernelArch(90), 80);
#else
  EXPECT_EQ(GetKernelArch(90), 90);
#endif
}

// Check default dispatch and every legal geometry around the two-column grid.y limit.
TEST_F(Fp16Int4SymmetricGroupwiseTest, Int4Group32SymmetricM1GridLimit) {
  if (onnxruntime::llm::common::getSMVersion() < kMinSupportedSm) {
    GTEST_SKIP() << "FP16 INT4 decode requires SM " << kMinSupportedSm << " or later";
  }
  for (int columns : {524224, 524288, 524352}) {
    SCOPED_TRACE(testing::Message() << "N=" << columns);
    InitBuffers(1, columns, 256, 32);
    EXPECT_TRUE(BenchmarkAndVerifyKernel());
    for (int variant = 2; variant <= 7; ++variant) {
      if (wo::IsInt4DecodeGeometryLegal(variant, columns, 256, 4)) {
        EXPECT_TRUE(BenchmarkAndVerifyKernel(false, variant));
      }
    }
  }
}

#if USE_COMPACT_FPA_INTB_GEMM
TEST_F(Bf16Int8GroupwiseTest, BF16_Int8_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 80) {
    GTEST_SKIP() << "bf16 int8 groupwise GEMM kernel requires SM 80 or later";
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list(wo::KernelType::BF16Int8Groupwise)) {
      InitBuffers(m, n, k, 32);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

TEST_F(Bf16Int4GroupwiseTest, BF16_Int4_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 80) {
    GTEST_SKIP() << "bf16 int4 groupwise GEMM kernel requires SM 80 or later";
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list(wo::KernelType::BF16Int4Groupwise)) {
      InitBuffers(m, n, k, 32);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

TEST_F(Fp16Int2GroupwiseTest, Fp16_Int2_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < kMinSupportedSm) {
    GTEST_SKIP() << "fp16 int2 groupwise GEMM kernel requires SM " << kMinSupportedSm << " or later";
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list(wo::KernelType::FP16Int2Groupwise)) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

TEST_F(Bf16Int2GroupwiseTest, BF16_Int2_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 80) {
    GTEST_SKIP() << "bf16 int2 groupwise GEMM kernel requires SM 80 or later";
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list(wo::KernelType::BF16Int2Groupwise)) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}
#endif

#if !USE_COMPACT_FPA_INTB_GEMM
TEST_F(Bf16Int8GroupwiseTest, BF16_Int8_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 80) {
    std::cout << "Skip bf16 int8 groupwise GEMM kernel test for SM < 80" << std::endl;
    return;
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list(wo::KernelType::BF16Int8Groupwise)) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

TEST_F(Bf16Int4GroupwiseTest, BF16_Int4_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 80) {
    std::cout << "Skip bf16 int4 groupwise GEMM kernel test for SM < 80" << std::endl;
    return;
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list(wo::KernelType::BF16Int4Groupwise)) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

// The 2-bit layout interleaves 8 columns per cache line, so a block owns CtaN * 8 = 32 columns.
TEST_F(Fp16Int2GroupwiseTest, Fp16_Int2_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < kMinSupportedSm) {
    GTEST_SKIP() << "fp16 int2 groupwise GEMM kernel requires SM " << kMinSupportedSm << " or later";
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list(wo::KernelType::FP16Int2Groupwise)) {
      InitBuffers(m, n, k, 128);  // 2-bit needs block_size >= 64
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

TEST_F(Bf16Int2GroupwiseTest, BF16_Int2_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 80) {
    std::cout << "Skip bf16 int2 groupwise GEMM kernel test for SM < 80" << std::endl;
    return;
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list(wo::KernelType::BF16Int2Groupwise)) {
      InitBuffers(m, n, k, 128);  // 2-bit needs block_size >= 64
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}
#endif
#endif

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Test can be run like the following:
//  ./onnxruntime_test_all --gtest_filter=CUDA_EP_Unittest.*

#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cutlass/numeric_types.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"
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
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace wo = onnxruntime::llm::kernels::fpA_intB_gemv;
using onnxruntime::llm::cutlass_extensions::CutlassGemmConfig;

namespace {
constexpr bool kPipelineMode = false;  // CI pipeline?

std::vector<int> get_m_list() {
  if (kPipelineMode) {
    return {1, 14};
  } else {
    return {1, 4, 8, 14, 256, 512, 1024, 2048};
  }
}

std::vector<std::pair<int, int>> get_n_k_list() {
  if (kPipelineMode) {
    return {{5120, 3072}};
  } else {
    // N and K from phi4 mini
    return {{5120, 3072}, {8192, 3072}, {3072, 8192}, {200064, 3072}};
  }
}

struct CudaBuffer {
  void* _data;
  int _bytes;

  CudaBuffer(int size_in_bytes) : _bytes(size_in_bytes) {
    cudaMalloc(&_data, _bytes);
  }

  template <typename T = void>
  T* data() {
    return reinterpret_cast<T*>(_data);
  }

  void to_cpu(void* dst) {
    cudaMemcpy(dst, _data, _bytes, cudaMemcpyDeviceToHost);
  }

  void from_cpu(void* src) {
    cudaMemcpy(_data, src, _bytes, cudaMemcpyHostToDevice);
  }

  ~CudaBuffer() {
    cudaFree(_data);
  }
};

template <typename T>
float compare(void* a, void* b, int size, float scale) {
  auto pa = reinterpret_cast<T*>(a);
  auto pb = reinterpret_cast<T*>(b);
  float max_diff = 0.f;
  float total_diff = 0.f;
  float max_val = 0.f;
  int diff_count = 0;
  float threshold = 1e-7;
  for (int n = 0; n < size; ++n) {
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
    printf("max diff %f (threshold %f), avg diff %f, diff count %d/%d\n",
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

#define CUTLASS_TYPE_MAPPER_REGISTRY(                                                                \
    CudaKernelType, KernelName, CudaAType, CutlassWType, WElemBits, CutlassQuantOp)                  \
  template <>                                                                                        \
  struct cutlassTypeMapper<CudaKernelType> {                                                         \
    using AType = CudaAType;                                                                         \
    using WType = CutlassWType;                                                                      \
    static constexpr cutlass::WeightOnlyQuantOp QuantOp = CutlassQuantOp;                            \
    static constexpr int WSizeInBits = WElemBits;                                                    \
    static std::string ATypeStr() { return std::is_same_v<CudaAType, half> ? "Fp16" : "BF16"; }      \
    static std::string WTypeStr() { return WSizeInBits == 4 ? "Int4" : "Int8"; }                     \
    static std::string str(int m, int n, int k, int group_size) {                                    \
      std::stringstream ss;                                                                          \
      ss << KernelName << " m=" << m << ", n=" << n << ", k=" << k << ", group_size=" << group_size; \
      return ss.str();                                                                               \
    }                                                                                                \
  };

CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int8Groupwise, "FP16Int8Groupwise", half, uint8_t, 8,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int8Groupwise, "BF16Int8Groupwise", __nv_bfloat16, uint8_t, 8,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::FP16Int4Groupwise, "FP16Int4Groupwise", half, cutlass::uint4b_t, 4,
                             cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);
CUTLASS_TYPE_MAPPER_REGISTRY(wo::KernelType::BF16Int4Groupwise, "BF16Int4Groupwise", __nv_bfloat16, cutlass::uint4b_t,
                             4, cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS);

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

template <wo::KernelType KT, typename Runner, typename Config>
void exec_cutlass_kernel(
    [[maybe_unused]] void* scaled_act, Runner& runner, wo::Params& params, Config& config, char* ws, size_t ws_size, cudaStream_t stream) {
  static constexpr cutlass::WeightOnlyQuantOp QuantOp = cutlassTypeMapper<KT>::QuantOp;
  void* act = params.act;
  if (params.act_scale) {
    ORT_THROW("act_scale is not supported in this test fixture.");
  }
  if (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
    runner.gemm(act, params.weight, params.scales, params.zeros, params.bias, params.out, params.m, params.n,
                params.k, params.groupsize, config, ws, ws_size, stream);
  }
}

struct BenchmarkResult {
  std::string a_type;
  std::string b_type;
  int m, n, k, group_size;
  float cuda_time_us;
  float cutlass_time_us;
  float nbits_time_us;
  float speedup_cuda_vs_cutlass;
  float speedup_cuda_vs_nbits;
  float speedup_cutlass_vs_nbits;
};

void PrintBenchmarkSummary(std::vector<BenchmarkResult>& benchmark_results) {
  std::cout << "\n--- Benchmark of FpA_IntB_GEMV, FpA_IntB_GEMM and MatMulNBits kernels (latency in microseconds) ---\n";
  std::cout << std::left << std::setw(6) << "A"
            << std::setw(6) << "W"
            << std::setw(6) << "m"
            << std::setw(8) << "n"
            << std::setw(7) << "k"
            << std::setw(12) << "group_size"
            << std::setw(12) << "gemv (us)"
            << std::setw(12) << "gemm (us)"
            << std::setw(12) << "nbits (us)"
            << std::setw(12) << "gemm/gemv"
            << std::setw(12) << "nbits/gemv"
            << std::setw(12) << "nbits/gemm"
            << std::endl;
  std::cout << std::string(115, '-') << std::endl;

  std::cout << std::fixed << std::setprecision(3);

  for (const auto& result : benchmark_results) {
    std::cout << std::left << std::setw(6) << result.a_type
              << std::setw(6) << result.b_type
              << std::setw(6) << result.m
              << std::setw(8) << result.n
              << std::setw(7) << result.k
              << std::setw(12) << result.group_size
              << std::setw(12) << result.cuda_time_us
              << std::setw(12) << result.cutlass_time_us
              << std::setw(12) << result.nbits_time_us
              << std::setw(12) << result.speedup_cuda_vs_cutlass
              << std::setw(12) << result.speedup_cuda_vs_nbits
              << std::setw(12) << result.speedup_cutlass_vs_nbits
              << std::endl;
  }
  std::cout << std::string(115, '-') << std::endl;
}

template <wo::KernelType KT, bool has_bias = false, bool has_act_scale = false, bool filter_configs = false>
class KernelTestFixture : public ::testing::Test {
 protected:
  int m_, n_, k_, group_size_;
  int warmup_ = 10;
  int repeats_ = 30;
  cudaDeviceProp device_prop_;
  std::shared_ptr<CudaBuffer> d_act_, d_act_scale_, d_weight_, d_scales_, d_zeros_, d_bias_, d_out_;
  std::vector<typename cutlassTypeMapper<KT>::AType> h_act_, h_act_scale_, h_scales_, h_zeros_, h_bias_, h_out1_, h_out2_;
  std::vector<uint8_t> h_weight_;
  std::vector<BenchmarkResult> benchmark_results_;
  cudaStream_t s_;
  cublasHandle_t cublas_handle_;

  static constexpr int WSizeInBits = cutlassTypeMapper<KT>::WSizeInBits;

  void SetUp() override {
    int device;
    CUDA_CALL_THROW(cudaGetDevice(&device));
    CUDA_CALL_THROW(cudaGetDeviceProperties(&device_prop_, device));
    std::srand(20240123);
    cudaStreamCreate(&s_);
    CUBLAS_CALL_THROW(cublasCreate(&cublas_handle_));
    CUBLAS_CALL_THROW(cublasSetStream(cublas_handle_, s_));
  }

  void TearDown() override {
    PrintBenchmarkSummary(benchmark_results_);
    cudaStreamDestroy(s_);
    cublasDestroy(cublas_handle_);
  }

  void InitBuffers(int m, int n, int k, int group_size) {
    m_ = m;
    n_ = n;
    k_ = k;
    group_size_ = group_size;

    if (cutlassTypeMapper<KT>::QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
      ORT_ENFORCE(group_size_ == 64 || group_size_ == 128);
      ORT_ENFORCE(k_ % group_size_ == 0);
    }

    using AType = typename cutlassTypeMapper<KT>::AType;

    d_act_ = std::make_shared<CudaBuffer>(m_ * k_ * sizeof(AType));
    d_act_scale_ = std::make_shared<CudaBuffer>(k_ * sizeof(AType));
    d_weight_ = std::make_shared<CudaBuffer>(k_ * n_ * WSizeInBits / 8);
    d_scales_ = std::make_shared<CudaBuffer>(n_ * k_ / group_size_ * sizeof(AType));
    d_zeros_ = std::make_shared<CudaBuffer>(n_ * k_ / group_size_ * sizeof(AType));
    d_bias_ = std::make_shared<CudaBuffer>(n_ * sizeof(AType));
    d_out_ = std::make_shared<CudaBuffer>(m_ * n_ * sizeof(AType));

    h_act_.resize(m_ * k_);
    h_act_scale_.resize(k_);
    h_weight_.resize(k_ * n_);
    h_scales_.resize(n_ * k_ / group_size_);
    h_zeros_.resize(n_ * k_ / group_size_);
    h_bias_.resize(n_);
    h_out1_.resize(m_ * n_);
    h_out2_.resize(m_ * n_);

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

  bool BenchmarkAndVerifyKernel() {
    printf("%s\n", cutlassTypeMapper<KT>::str(m_, n_, k_, group_size_).c_str());

    void* p_act_scale = nullptr;
    void* p_zeros = nullptr;
    void* p_bias = nullptr;

    if (group_size_ != 0) {
      p_zeros = d_zeros_->data();
      if constexpr (has_bias) {
        p_bias = d_bias_->data();
      }
      if constexpr (has_act_scale) {
        p_act_scale = d_act_scale_->data();
      }
    }

    wo::Params params(d_act_->data(), p_act_scale, d_weight_->data(), d_scales_->data(), p_zeros, p_bias,
                      d_out_->data(), 1.f, m_, n_, k_, group_size_, KT);

    //------------------------
    // Run FpA_IntB_Gemv CUDA kernel
    float cuda_time_ms = 0.f;
    if (m_ < 16) {
      cuda_time_ms = measure_kernel_time(
          [&]() {
            int arch = onnxruntime::llm::common::getSMVersion();
            ORT_ENFORCE(wo::is_supported(arch, params.type));
            wo::kernel_launcher(arch, params, s_);
          },
          warmup_, repeats_, s_);
      d_out_->to_cpu(h_out1_.data());
    }

    // ------------------------
    // Run FpA_IntB_Gemm CUTLASS kernel
    using AType = typename cutlassTypeMapper<KT>::AType;
    using WType = typename cutlassTypeMapper<KT>::WType;
    using onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner;
    auto runner = std::make_shared<CutlassFpAIntBGemmRunner<AType, WType, cutlassTypeMapper<KT>::QuantOp>>();
    auto& gemm_runner = *runner;
    int ws_bytes = gemm_runner.getWorkspaceSize(m_, n_, k_);
    CudaBuffer ws_buffer(ws_bytes);
    char* ws_ptr = reinterpret_cast<char*>(ws_buffer.data());

    auto configs = gemm_runner.getConfigs();

    if constexpr (filter_configs) {
      configs = filter_gemm_configs(configs, k_);
    }

    float fast_time_ms = std::numeric_limits<float>::max();
    CutlassGemmConfig best_config = configs[0];

    int cfg_i = 0;
    for (auto& config : configs) {
      float time = std::numeric_limits<float>::max();
      try {
        time = measure_kernel_time(
            [&]() {
              exec_cutlass_kernel<KT>(d_act_->data(), gemm_runner, params, config, ws_ptr, ws_bytes, s_);
            },
            2, 5, s_);
      } catch (std::exception const& e) {
        std::ostringstream msg;
        msg << "Cannot profile configuration " << cfg_i;
        if constexpr (std::is_same_v<decltype(config), CutlassGemmConfig>) {
          msg << ": " << config.toString();
        }
        msg << "\n (for"
            << " m=" << params.m << ", n=" << params.n << ", k=" << params.k << ")"
            << ", reason: \"" << e.what() << "\". Skipped\n";
        std::cout << msg.str();
        cudaGetLastError();  // Reset the last cudaError to cudaSuccess.
        continue;
      }
      if (time < fast_time_ms) {
        fast_time_ms = time;
        best_config = config;
      }
      cfg_i++;
    }

    float cutlass_time_ms = measure_kernel_time(
        [&]() {
          exec_cutlass_kernel<KT>(d_act_->data(), gemm_runner, params, best_config, ws_ptr, ws_bytes, s_);
        },
        warmup_, repeats_, s_);
    d_out_->to_cpu(h_out2_.data());

    // ------------------------
    // Compare FpA_IntB_Gemv and FpA_IntB_Gemm outputs.
    bool pass = true;
    if (m_ < 16) {
      float quant_scale = 1.f / (1 << (WSizeInBits - 1));
      pass = compare<AType>(h_out1_.data(), h_out2_.data(), m_ * n_, quant_scale);
    }

    // ------------------------
    // Run MatMulNBits kernel.
    // Note that it runs on random data, so the output is not compared.
    float nbits_time_ms = 0.f;
    if constexpr (KT == wo::KernelType::FP16Int8Groupwise || KT == wo::KernelType::FP16Int4Groupwise) {
      std::vector<uint8_t> h_uint8_zeros(n_ * k_ / group_size_);
      for (uint8_t& v : h_uint8_zeros) {
        v = rand() % 256;
      }

      ORT_ENFORCE(k_ / group_size_ * WSizeInBits % 8 == 0);
      CudaBuffer d_uint8_zeros(n_ * k_ / group_size_ * WSizeInBits / 8);
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
                                                         m_, n_, k_, group_size_, device_prop_.sharedMemPerBlock, s_);
            },
            warmup_, repeats_, s_);
      } else {
        CudaBuffer d_dequantized_weight(SafeInt<int>(n_) * k_ * sizeof(AType));

        nbits_time_ms = measure_kernel_time(
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
                  group_size_,
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
    }

    // Store benchmark results
    BenchmarkResult result;
    result.a_type = cutlassTypeMapper<KT>::ATypeStr();
    result.b_type = cutlassTypeMapper<KT>::WTypeStr();
    result.m = m_;
    result.n = n_;
    result.k = k_;
    result.group_size = group_size_;
    result.cuda_time_us = cuda_time_ms * 1000.0f;
    result.cutlass_time_us = cutlass_time_ms * 1000.0f;
    result.nbits_time_us = nbits_time_ms * 1000.0f;
    result.speedup_cuda_vs_cutlass = cuda_time_ms > 0.f ? cutlass_time_ms / cuda_time_ms : 0.f;
    result.speedup_cuda_vs_nbits = cuda_time_ms > 0.f ? nbits_time_ms / cuda_time_ms : 0.f;
    result.speedup_cutlass_vs_nbits = cutlass_time_ms > 0.f ? nbits_time_ms / cutlass_time_ms : 0.f;

    benchmark_results_.push_back(result);

    return pass;
  }
};

}  // namespace

using Fp16Int8GroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int8Groupwise>;
using Fp16Int4GroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int4Groupwise>;
using Bf16Int8GroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int8Groupwise>;
using Bf16Int4GroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int4Groupwise>;

TEST_F(Fp16Int8GroupwiseTest, Fp16_Int8_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 75) {
    std::cout << "Skip fp16 int8 groupwise GEMM kernel for SM < 75" << std::endl;
    return;
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list()) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
      InitBuffers(m, n, k, 128);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

TEST_F(Fp16Int4GroupwiseTest, Fp16_Int4_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 75) {
    std::cout << "Skip fp16 int4 groupwise GEMM kernel for SM < 75" << std::endl;
    return;
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list()) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
      InitBuffers(m, n, k, 128);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

TEST_F(Bf16Int8GroupwiseTest, BF16_Int8_Gemm_CudaKernel) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 80) {
    std::cout << "Skip bf16 int8 groupwise GEMM kernel test for SM < 80" << std::endl;
    return;
  }

  for (auto m : get_m_list()) {
    for (const auto& [n, k] : get_n_k_list()) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
      InitBuffers(m, n, k, 128);
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
    for (const auto& [n, k] : get_n_k_list()) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
      InitBuffers(m, n, k, 128);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

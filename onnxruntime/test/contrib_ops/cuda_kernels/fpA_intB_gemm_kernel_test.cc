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
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <functional>
#include <iomanip>
#include <iostream>
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
  return {1, 8, 15, 16, 32, 128, 512};
#else
  if (kPipelineMode) {
    return {1, 8, 14};
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
  void* _data;
  size_t _bytes;

  CudaBuffer(size_t size_in_bytes) : _bytes(size_in_bytes) {
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
  cudaStream_t s_;
  cublasHandle_t cublas_handle_;

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
    cudaStreamCreate(&s_);
    CUBLAS_CALL_THROW(cublasCreate(&cublas_handle_));
    CUBLAS_CALL_THROW(cublasSetStream(cublas_handle_, s_));
  }

  void TearDown() override {
    PrintBenchmarkSummary(benchmark_results_);
    cudaStreamDestroy(s_);
    cublasDestroy(cublas_handle_);
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

  // Reports the device that runs the test to the GEMV dispatcher, as MatMulNBits and the tactic profiler do.
  void ReportDevice(wo::Params& params) const {
    params.device_sm = device_prop_.major * 10 + device_prop_.minor;
    params.device_sm_count = device_prop_.multiProcessorCount;
  }

  // Fills what the dispatcher reads from Params: the reported device and the opt-in wave-aware mode.
  using ParamsHook = std::function<void(wo::Params&)>;

  // Launches the zero-point-free GEMV once with debug logging and returns the CtaN column tile it reports.
  int LaunchCtaN(const ParamsHook& configure) {
    static_assert(QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY);
    wo::Params params(d_act_->data(), nullptr, d_weight_->data(), d_scales_->data(), nullptr, nullptr,
                      d_out_->data(), 1.f, m_, n_, k_, block_size_, KT);
    params.debug = true;
    configure(params);
    const int kernel_arch = GetKernelArch(onnxruntime::llm::common::getSMVersion());
    testing::internal::CaptureStdout();
    try {
      wo::kernel_launcher(kernel_arch, params, s_);
      cudaStreamSynchronize(s_);
    } catch (...) {
      // End the capture so that later tests can capture stdout as well.
      testing::internal::GetCapturedStdout();
      throw;
    }
    const std::string log = testing::internal::GetCapturedStdout();
    const size_t pos = log.find("CtaN=");
    return pos == std::string::npos ? 0 : std::atoi(log.c_str() + pos + 5);
  }

  // Launches with this device reported. Without a reported device the dispatcher must use its device-independent
  // cutoffs.
  int GemvLaunchCtaN(bool report_device = true) {
    return LaunchCtaN([&](wo::Params& params) {
      if (report_device) {
        ReportDevice(params);
      }
    });
  }

  // Checks that one M = 8 launch is bitwise identical to two M = 4 launches over the same weights. The tile
  // shape only decides which block computes which output, not the K reduction order of that output. `configure`
  // defaults to reporting this device.
  bool M8MatchesTwoM4Launches(const ParamsHook& configure = nullptr) {
    static_assert(QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY);
    using AType = typename cutlassTypeMapper<KT>::AType;
    ORT_ENFORCE(m_ == 8);
    const int kernel_arch = GetKernelArch(onnxruntime::llm::common::getSMVersion());
    auto launch = [&](int rows, int first_row) {
      wo::Params params(d_act_->data<AType>() + static_cast<size_t>(first_row) * k_, nullptr, d_weight_->data(),
                        d_scales_->data(), nullptr, nullptr,
                        d_out_->data<AType>() + static_cast<size_t>(first_row) * n_, 1.f, rows, n_, k_,
                        block_size_, KT);
      if (configure) {
        configure(params);
      } else {
        ReportDevice(params);
      }
      wo::kernel_launcher(kernel_arch, params, s_);
    };
    // Start from a NaN pattern so that an output that is never written cannot match by accident.
    auto run = [&](const auto& launches, std::vector<AType>& host_out) {
      CUDA_CALL_THROW(cudaMemsetAsync(d_out_->data(), 0xFF, d_out_->_bytes, s_));
      launches();
      CUDA_CALL_THROW(cudaStreamSynchronize(s_));
      d_out_->to_cpu(host_out.data());
    };
    auto one_m8_launch = [&]() { launch(8, 0); };
    auto two_m4_launches = [&]() {
      launch(4, 0);
      launch(4, 4);
    };
    run(one_m8_launch, h_out1_);
    run(two_m4_launches, h_out2_);
    const bool finite = std::all_of(h_out1_.begin(), h_out1_.end(),
                                    [](AType v) { return std::isfinite(static_cast<float>(v)); });
    return finite && std::memcmp(h_out1_.data(), h_out2_.data(), h_out1_.size() * sizeof(AType)) == 0;
  }

  // The device the A100 M = 8 windows are qualified on: A100 (SM 8.0, 108 SMs). A30 is also SM 8.0 but has 56 SMs.
  bool UsesA100M8Tiles() const {
    return device_prop_.major == 8 && device_prop_.minor == 0 && device_prop_.multiProcessorCount == 108;
  }

  // The device the H200 M = 8 rule is qualified on: SM 9.0 with 132 SMs. H100 PCIe (114 SMs) and H20 (78 SMs)
  // are also SM 9.0 and keep the device-independent cutoffs.
  bool UsesH200M8Tiles() const {
    return device_prop_.major == 9 && device_prop_.minor == 0 && device_prop_.multiProcessorCount == 132;
  }

  // Picks the {M, N, expected CtaN} table of the device that runs the test: A100, H200, or any other device.
  const std::vector<std::tuple<int, int, int>>& M8TileCasesForDevice(
      const std::vector<std::tuple<int, int, int>>& a100_cases,
      const std::vector<std::tuple<int, int, int>>& h200_cases,
      const std::vector<std::tuple<int, int, int>>& other_cases) const {
    return UsesA100M8Tiles() ? a100_cases : (UsesH200M8Tiles() ? h200_cases : other_cases);
  }

  // Tile-only check that a caller that does not report its device gets the device-independent cutoffs, also on an A100.
  void CheckUnreportedDeviceTiles(const std::vector<std::tuple<int, int, int>>& cases,
                                  const std::vector<int>& k_values = {1024, 4096}) {
    for (const int k : k_values) {
      for (const auto& [m, n, expected_cta_n] : cases) {
        SCOPED_TRACE(testing::Message() << "unreported device M=" << m << " N=" << n << " K=" << k);
        InitBuffers(m, n, k, kIsInt2 ? 64 : 32);
        EXPECT_EQ(GemvLaunchCtaN(false), expected_cta_n);
      }
    }
  }

  // For each K and {M, N, expected CtaN} case, checks the column tile the dispatcher launches, the result against
  // the CUTLASS reference, and (for M = 8) that it equals two M = 4 launches bit for bit. The fixture is scale-only,
  // so the weights always use the SM80-interleaved layout (also on Hopper), the only layout with a narrow tile.
  void CheckGemvTiles(const std::vector<std::tuple<int, int, int>>& cases,
                      const std::vector<int>& k_values = {1024, 4096}) {
    static_assert(GetKernelArch(90) == 80, "The narrow M = 8 tile exists only in the SM80-interleaved layout");
    for (const int k : k_values) {
      for (const auto& [m, n, expected_cta_n] : cases) {
        SCOPED_TRACE(testing::Message() << "M=" << m << " N=" << n << " K=" << k);
        InitBuffers(m, n, k, kIsInt2 ? 64 : 32);
        EXPECT_EQ(GemvLaunchCtaN(), expected_cta_n);
        EXPECT_TRUE(BenchmarkAndVerifyKernel());
        if (m == 8) {
          EXPECT_TRUE(M8MatchesTwoM4Launches());
        }
      }
    }
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
    ReportDevice(params);

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
using Fp16Int8SymmetricGroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int8Groupwise, false, false, true,
                                                         cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY>;
using Bf16Int8SymmetricGroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int8Groupwise, false, false, true,
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
                                    80, 4, 32, false, true);
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

TEST(FpAIntBGemvTest, A100M8DispatchUsesMeasuredWindows) {
  for (int group_size : {32, 64, 128}) {
    for (int k : {1024, 2048, 3072, 4096, 8192}) {
      for (const auto& [n, expected] : std::vector<std::pair<int, int>>{
               {1024, 4}, {1712, 4}, {1728, 4}, {1744, k <= 3072 ? 2 : 4}, {2560, k <= 3072 ? 2 : 4}, {2576, 4}, {2880, 4}, {3456, 4}, {3472, 2}, {4096, 2}, {4320, 2}, {4336, 4}, {4352, 4}, {5120, 4}, {5136, 4}}) {
        SCOPED_TRACE(testing::Message() << "INT4 N=" << n << " K=" << k << " group size " << group_size);
        EXPECT_EQ(wo::PickM8GemvCtaN(wo::M8GemvDevice::kA100, 8, n, k, group_size, 4, false, 4), expected);
      }
      for (const auto& [n, expected] : std::vector<std::pair<int, int>>{
               {640, 4}, {864, 4}, {872, 2}, {1024, 2}, {1280, 2}, {1288, 2}, {1296, 2}, {1304, 2}, {1312, 4}, {1536, 4}, {2560, 4}}) {
        SCOPED_TRACE(testing::Message() << "INT8 N=" << n << " K=" << k << " group size " << group_size);
        EXPECT_EQ(wo::PickM8GemvCtaN(wo::M8GemvDevice::kA100, 8, n, k, group_size, 2, false, 4), expected);
      }
    }
  }
}

// The A100 windows were timed without a bias, which keeps the default tile there.
TEST(FpAIntBGemvTest, A100M8DispatchKeepsDefaultTileWithBias) {
  for (int k : {1024, 3072, 4096}) {
    for (int n : {2560, 3472, 4096, 4320}) {
      SCOPED_TRACE(testing::Message() << "INT4 N=" << n << " K=" << k);
      EXPECT_EQ(wo::PickM8GemvCtaN(wo::M8GemvDevice::kA100, 8, n, k, 32, 4, true, 4), 4);
    }
    for (int n : {872, 1024, 1280, 1304}) {
      SCOPED_TRACE(testing::Message() << "INT8 N=" << n << " K=" << k);
      EXPECT_EQ(wo::PickM8GemvCtaN(wo::M8GemvDevice::kA100, 8, n, k, 32, 2, true, 4), 4);
    }
  }
}

// H200 INT4: the narrow tile fills one wave of 132 SMs * 5 blocks * 8 columns = 5280 columns and fades with the
// bytes of weights plus scales. INT8 never takes the narrow tile.
TEST(FpAIntBGemvTest, H200M8DispatchUsesMeasuredRule) {
  constexpr auto kH200 = wo::M8GemvDevice::kH200;
  const std::vector<std::tuple<int, int, int>> int4_cases{
      // {N, K, expected CtaN} at group size 32 (0.5625 bytes per weight) without a bias.
      // N <= 4224 takes the narrow tile while weights plus scales stay within 28 MB.
      {8, 1024, 2},
      {1024, 1024, 2},
      {1024, 31744, 2},
      {2048, 16384, 2},
      {3072, 16192, 2},
      {3072, 16224, 4},
      {4096, 12128, 2},
      {4096, 12160, 4},
      {4224, 11776, 2},
      {4224, 11808, 4},
      // 4224 < N <= 5280 takes it while they stay within 42 MB: q / o projection 5120 x 5120 does, the 5120 x 17408
      // down projection does not.
      {4232, 1024, 2},
      {4232, 17632, 2},
      {4232, 17664, 4},
      {5120, 5120, 2},
      {5120, 13824, 2},
      {5120, 14560, 2},
      {5120, 14592, 4},
      {5120, 17408, 4},
      {5280, 1024, 2},
      {5280, 14112, 2},
      {5280, 14144, 4},
      // A second wave of narrow blocks starts above N = 5280.
      {5288, 1024, 4},
      {5296, 1024, 4},
      {5312, 1024, 4},
      {6144, 1024, 4},
      {17408, 5120, 4},
      // N * K beyond the 32-bit range must not wrap around into the narrow window.
      {4096, 1 << 20, 4},
  };
  for (const auto& [n, k, expected] : int4_cases) {
    SCOPED_TRACE(testing::Message() << "INT4 N=" << n << " K=" << k);
    EXPECT_EQ(wo::PickM8GemvCtaN(kH200, 8, n, k, 32, 4, false, 4), expected);
  }
  for (int k : {1024, 3072, 4096, 8192, 17408}) {
    for (int n : {4, 640, 1024, 1280, 1288, 2560, 3584, 5120}) {
      for (bool has_bias : {false, true}) {
        SCOPED_TRACE(testing::Message() << "INT8 N=" << n << " K=" << k << " bias " << has_bias);
        EXPECT_EQ(wo::PickM8GemvCtaN(kH200, 8, n, k, 32, 2, has_bias, 4), 4);
      }
    }
  }
}

// The caps are in bytes of INT4 weights plus 16-bit scales, so the K at which they bind grows with the group size
// (N = 5120 with a scale-only epilogue: K = 14560 / 15424 / 15872 for group sizes 32 / 64 / 128).
TEST(FpAIntBGemvTest, H200M8DispatchCapsFollowWeightAndScaleBytes) {
  constexpr auto kH200 = wo::M8GemvDevice::kH200;
  constexpr int64_t kTieMaxBytes = 28'000'000;
  constexpr int64_t kOneWaveMaxBytes = 42'000'000;
  constexpr int64_t kBiasOneWaveMaxBytes = 34'000'000;
  // The largest K that is a multiple of the group size and keeps the weights plus scales within max_bytes.
  auto max_k = [](int n, int group_size, int64_t max_bytes) {
    int k = group_size;
    while (wo::Int4WeightAndScaleBytes(n, k + group_size, group_size) <= max_bytes) {
      k += group_size;
    }
    return k;
  };
  for (int group_size : {32, 64, 128}) {
    for (const auto& [n, has_bias, max_bytes] : std::vector<std::tuple<int, bool, int64_t>>{
             {1024, false, kTieMaxBytes},
             {3072, false, kTieMaxBytes},
             {4096, false, kTieMaxBytes},
             {4224, false, kTieMaxBytes},
             {4232, false, kOneWaveMaxBytes},
             {4608, false, kOneWaveMaxBytes},
             {5120, false, kOneWaveMaxBytes},
             {5280, false, kOneWaveMaxBytes},
             {4232, true, kBiasOneWaveMaxBytes},
             {4608, true, kBiasOneWaveMaxBytes},
             {5120, true, kBiasOneWaveMaxBytes},
             {5280, true, kBiasOneWaveMaxBytes}}) {
      const int k = max_k(n, group_size, max_bytes);
      SCOPED_TRACE(testing::Message() << "N=" << n << " K=" << k << " group size " << group_size << " bias " << has_bias);
      EXPECT_EQ(wo::PickM8GemvCtaN(kH200, 8, n, k, group_size, 4, has_bias, 4), 2);
      EXPECT_EQ(wo::PickM8GemvCtaN(kH200, 8, n, k + group_size, group_size, 4, has_bias, 4), 4);
    }
  }
}

// With a bias the half-width INT4 kernel is 6 - 32% slower than the default one, except for the wave gain of
// 4224 < N <= 5280, which the caps keep below the point where the K-dependent penalty overtakes it.
TEST(FpAIntBGemvTest, H200M8DispatchWithBiasUsesWaveWindowOnly) {
  constexpr auto kH200 = wo::M8GemvDevice::kH200;
  const std::vector<std::tuple<int, int, int>> int4_cases{
      // {N, K, expected CtaN} at group size 32 with a bias.
      {8, 1024, 4},
      {1024, 1024, 4},
      {2048, 4096, 4},
      {4096, 4096, 4},
      {4224, 1024, 4},
      {4224, 5120, 4},
      {4232, 1024, 2},
      {4232, 14272, 2},
      {4232, 14304, 4},
      {5120, 5120, 2},
      {5120, 11776, 2},
      {5120, 11808, 4},
      {5120, 12288, 4},
      {5120, 17408, 4},
      {5280, 5120, 2},
      {5280, 11424, 2},
      {5280, 11456, 4},
      {5288, 1024, 4},
      {6144, 1024, 4},
  };
  for (const auto& [n, k, expected] : int4_cases) {
    SCOPED_TRACE(testing::Message() << "INT4 N=" << n << " K=" << k);
    EXPECT_EQ(wo::PickM8GemvCtaN(kH200, 8, n, k, 32, 4, true, 4), expected);
  }
}

TEST(FpAIntBGemvTest, M8DispatchKeepsDefaultTileOutsideMeasuredShapes) {
  for (const auto device : {wo::M8GemvDevice::kUnmeasured, wo::M8GemvDevice::kA100, wo::M8GemvDevice::kH200}) {
    for (bool has_bias : {false, true}) {
      for (int m : {7, 9}) {
        EXPECT_EQ(wo::PickM8GemvCtaN(device, m, 4096, 4096, 32, 4, has_bias, 4), 4);
        EXPECT_EQ(wo::PickM8GemvCtaN(device, m, 1280, 4096, 32, 2, has_bias, 4), 4);
      }
      EXPECT_EQ(wo::PickM8GemvCtaN(device, 8, 4097, 4096, 32, 4, has_bias, 4), 4);
      EXPECT_EQ(wo::PickM8GemvCtaN(device, 8, 1281, 4096, 32, 2, has_bias, 4), 4);
      EXPECT_EQ(wo::PickM8GemvCtaN(device, 8, 1280, 4096, 32, 1, has_bias, 4), 4);
      EXPECT_EQ(wo::PickM8GemvCtaN(device, 8, 1280, 4096, 32, 2, has_bias, 2), 2);
    }
  }
}

TEST(FpAIntBGemvTest, A100DeviceRequiresMatchingSmAndSmCount) {
  EXPECT_TRUE(wo::IsA100Device(80, 108));
  EXPECT_FALSE(wo::IsA100Device(80, 56));  // A30: same SM 8.0, fewer SMs
  EXPECT_FALSE(wo::IsA100Device(86, 108));
  EXPECT_FALSE(wo::IsA100Device(90, 132));
  EXPECT_FALSE(wo::IsA100Device(0, 0));  // a caller that does not report its device
  EXPECT_FALSE(wo::IsA100Device(80, 0));
  EXPECT_FALSE(wo::IsA100Device(0, 108));
}

TEST(FpAIntBGemvTest, H200DeviceRequiresMatchingSmAndSmCount) {
  EXPECT_TRUE(wo::IsH200Device(90, 132));
  EXPECT_FALSE(wo::IsH200Device(90, 114));  // H100 PCIe: same SM 9.0, fewer SMs
  EXPECT_FALSE(wo::IsH200Device(90, 78));   // H20
  EXPECT_FALSE(wo::IsH200Device(80, 132));
  EXPECT_FALSE(wo::IsH200Device(100, 132));
  EXPECT_FALSE(wo::IsH200Device(80, 108));
  EXPECT_FALSE(wo::IsH200Device(0, 0));  // a caller that does not report its device
  EXPECT_FALSE(wo::IsH200Device(90, 0));
  EXPECT_FALSE(wo::IsH200Device(0, 132));

  EXPECT_EQ(wo::GetM8GemvDevice(80, 108), wo::M8GemvDevice::kA100);
  EXPECT_EQ(wo::GetM8GemvDevice(90, 132), wo::M8GemvDevice::kH200);
  EXPECT_EQ(wo::GetM8GemvDevice(80, 56), wo::M8GemvDevice::kUnmeasured);
  EXPECT_EQ(wo::GetM8GemvDevice(90, 114), wo::M8GemvDevice::kUnmeasured);
  EXPECT_EQ(wo::GetM8GemvDevice(0, 0), wo::M8GemvDevice::kUnmeasured);
}

TEST(FpAIntBGemvTest, OtherDevicesKeepM8DispatchCutoffs) {
  for (int k : {1024, 4096, 8192}) {
    for (int group_size : {32, 64, 128}) {
      for (bool has_bias : {false, true}) {
        SCOPED_TRACE(testing::Message() << "K=" << k << " group size " << group_size << " bias " << has_bias);
        for (const auto& [n, expected] : std::vector<std::pair<int, int>>{
                 {2880, 2}, {5120, 2}, {5128, 2}, {5136, 4}, {6144, 4}}) {
          EXPECT_EQ(wo::PickM8GemvCtaN(wo::M8GemvDevice::kUnmeasured, 8, n, k, group_size, 4, has_bias, 4), expected);
        }
        for (const auto& [n, expected] : std::vector<std::pair<int, int>>{
                 {640, 2}, {1280, 2}, {1284, 2}, {1288, 4}, {1536, 4}, {2560, 4}}) {
          EXPECT_EQ(wo::PickM8GemvCtaN(wo::M8GemvDevice::kUnmeasured, 8, n, k, group_size, 2, has_bias, 4), expected);
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

// The wave-aware SM count is the one the caller reports, and only an sm_12x device has one.
TEST(FpAIntBGemvTest, WaveAwareSmCountComesFromReportedDevice) {
  EXPECT_EQ(wo::GemvPickSmCount(120, 170), 170);  // RTX 5090
  EXPECT_EQ(wo::GemvPickSmCount(121, 48), 48);
  EXPECT_EQ(wo::GemvPickSmCount(80, 108), 0);   // A100
  EXPECT_EQ(wo::GemvPickSmCount(90, 132), 0);   // H100
  EXPECT_EQ(wo::GemvPickSmCount(100, 148), 0);  // sm_100 is not sm_12x
  EXPECT_EQ(wo::GemvPickSmCount(0, 0), 0);      // a caller that does not report its device
  EXPECT_EQ(wo::GemvPickSmCount(120, 0), 0);
  EXPECT_EQ(wo::GemvPickSmCount(0, 170), 0);
}

// Keep upstream wave-aware and default tactic caches distinct after extending the key.
TEST(FpAIntBGemvTest, TacticCacheSeparatesWaveAwareMode) {
  using TacticCache = std::unordered_map<wo_profile::GemmIdCore, int, wo_profile::GemmIdCoreHash>;
  wo_profile::GemmIdCore const default_id(10240, 4096, onnxruntime::llm::nvinfer::DataType::kHALF,
                                          80, 4, 128, false, true, false, "", false);
  wo_profile::GemmIdCore const wave_aware_id(10240, 4096, onnxruntime::llm::nvinfer::DataType::kHALF,
                                             80, 4, 128, false, true, false, "", true);
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
                                      80, 4, 128, false, true, false, "", wave_aware, mode);
      EXPECT_EQ(id.wave_aware, wave_aware);
      EXPECT_EQ(id.tag, mode);
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
                                      80, 4, 128, false, true, false, "", wave_aware, mode);
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

namespace {
// {M, N, expected CtaN} cases for the scale-only SM80-interleaved layout. Devices without a measured rule
// retain the default-grid cutoffs of 320 blocks for INT4 and 160 for INT8. A100 cases cover
// its measured windows and K-dependent lower INT4 window. H200 cases cover the 5280-column single-wave edge
// and the weight-plus-scale byte caps of INT4 (INT8 always keeps CtaN 4). Neighbouring M = 7 / 9 keep CtaN 4.
const std::vector<std::tuple<int, int, int>> kInt4M8TileCases{
    {8, 2880, 2}, {8, 5120, 2}, {7, 5120, 4}, {9, 5120, 4}, {8, 5136, 4}, {8, 6144, 4}};
const std::vector<std::tuple<int, int, int>> kInt8M8TileCases{
    {8, 640, 2}, {8, 1280, 2}, {7, 1280, 4}, {9, 1280, 4}, {8, 1288, 4}, {8, 1536, 4}, {8, 2560, 4}};
const std::vector<std::tuple<int, int, int>> kInt4A100M8TileCases{
    {8, 2880, 4}, {8, 3456, 4}, {8, 3472, 2}, {8, 4320, 2}, {8, 4336, 4}, {8, 5120, 4}, {7, 4320, 4}, {9, 4320, 4}, {8, 5136, 4}, {8, 6144, 4}};
const std::vector<std::tuple<int, int, int>> kInt8A100M8TileCases{
    {8, 640, 4}, {8, 864, 4}, {8, 872, 2}, {8, 1280, 2}, {8, 1288, 2}, {8, 1304, 2}, {8, 1312, 4}, {7, 1280, 4}, {9, 1280, 4}, {8, 1536, 4}, {8, 2560, 4}};
const std::vector<std::tuple<int, int, int>> kInt4A100M8SmallKTileCases{
    {8, 1728, 4}, {8, 1744, 2}, {8, 2560, 2}, {8, 2576, 4}};
const std::vector<std::tuple<int, int, int>> kInt4A100M8LargeKTileCases{
    {8, 1744, 4}, {8, 2560, 4}};
// The INT4 cases use multiples of 16 columns: the default tile and the two M = 4 launches of the bit-identity check
// both need N % (4 * kInterleave) == 0.
const std::vector<std::tuple<int, int, int>> kInt4H200M8TileCases{
    {8, 2880, 2}, {8, 4224, 2}, {8, 4240, 2}, {8, 5120, 2}, {8, 5280, 2}, {7, 5280, 4}, {9, 5280, 4}, {8, 5296, 4}, {8, 5312, 4}, {8, 6144, 4}};
const std::vector<std::tuple<int, int, int>> kInt8H200M8TileCases{
    {8, 640, 4}, {8, 1280, 4}, {7, 1280, 4}, {9, 1280, 4}, {8, 1288, 4}, {8, 2560, 4}};
// K = 14336 streams 0.5625 bytes per weight at group size 32: N = 3072 (24.8 MB) is within the 28 MB cap of
// N <= 4224 and N = 4096 (33.0 MB) past it; N = 5120 (41.3 MB) is within the 42 MB cap above it and N = 5280
// (42.6 MB) past it.
const std::vector<std::tuple<int, int, int>> kInt4H200M8MidKTileCases{
    {8, 2048, 2}, {8, 3072, 2}, {8, 4096, 4}, {8, 5120, 2}, {8, 5280, 4}, {8, 5296, 4}};
// K = 16384: N = 3072 (28.3 MB), 4224 (38.9 MB) and 5120 (47.2 MB) are past their caps.
const std::vector<std::tuple<int, int, int>> kInt4H200M8LargeKTileCases{
    {8, 2048, 2}, {8, 3072, 4}, {8, 4224, 4}, {8, 5120, 4}};
}  // namespace

// Check the FP16 INT4 narrow tile, its grid cutoff and the M = 7 / 9 fallback against the CUTLASS reference.
TEST_F(Fp16Int4SymmetricGroupwiseTest, Int4Group32SymmetricM8NarrowTile) {
  if (onnxruntime::llm::common::getSMVersion() < kMinSupportedSm) {
    GTEST_SKIP() << "FP16 INT4 GEMV requires SM " << kMinSupportedSm << " or later";
  }
  CheckGemvTiles(M8TileCasesForDevice(kInt4A100M8TileCases, kInt4H200M8TileCases, kInt4M8TileCases));
  if (UsesA100M8Tiles()) {
    CheckGemvTiles(kInt4A100M8SmallKTileCases, {1024, 2048, 3072});
    CheckGemvTiles(kInt4A100M8LargeKTileCases, {4096, 8192});
  }
  if (UsesH200M8Tiles()) {
    CheckGemvTiles(kInt4H200M8MidKTileCases, {14336});
    CheckGemvTiles(kInt4H200M8LargeKTileCases, {16384});
  }
  CheckUnreportedDeviceTiles(kInt4M8TileCases);
}

// Check the BF16 INT4 narrow tile, its grid cutoff and the M = 7 / 9 fallback against the CUTLASS reference.
TEST_F(Bf16Int4SymmetricGroupwiseTest, Int4Group32SymmetricM8NarrowTile) {
  if (onnxruntime::llm::common::getSMVersion() < 80) {
    GTEST_SKIP() << "BF16 INT4 GEMV requires SM 80 or later";
  }
  CheckGemvTiles(M8TileCasesForDevice(kInt4A100M8TileCases, kInt4H200M8TileCases, kInt4M8TileCases));
  if (UsesA100M8Tiles()) {
    CheckGemvTiles(kInt4A100M8SmallKTileCases, {1024, 2048, 3072});
    CheckGemvTiles(kInt4A100M8LargeKTileCases, {4096, 8192});
  }
  if (UsesH200M8Tiles()) {
    CheckGemvTiles(kInt4H200M8MidKTileCases, {14336});
    CheckGemvTiles(kInt4H200M8LargeKTileCases, {16384});
  }
  CheckUnreportedDeviceTiles(kInt4M8TileCases);
}

// Check the FP16 INT8 narrow tile (N = 1280 is 160 default blocks), the first aligned N above the cutoff and the
// M = 7 / 9 fallback against the CUTLASS reference.
TEST_F(Fp16Int8SymmetricGroupwiseTest, Int8Group32SymmetricM8NarrowTile) {
  if (onnxruntime::llm::common::getSMVersion() < kMinSupportedSm) {
    GTEST_SKIP() << "FP16 INT8 GEMV requires SM " << kMinSupportedSm << " or later";
  }
  CheckGemvTiles(M8TileCasesForDevice(kInt8A100M8TileCases, kInt8H200M8TileCases, kInt8M8TileCases));
  CheckUnreportedDeviceTiles(kInt8M8TileCases);
}

// Check the BF16 INT8 narrow tile (N = 1280 is 160 default blocks), the first aligned N above the cutoff and the
// M = 7 / 9 fallback against the CUTLASS reference.
TEST_F(Bf16Int8SymmetricGroupwiseTest, Int8Group32SymmetricM8NarrowTile) {
  if (onnxruntime::llm::common::getSMVersion() < 80) {
    GTEST_SKIP() << "BF16 INT8 GEMV requires SM 80 or later";
  }
  CheckGemvTiles(M8TileCasesForDevice(kInt8A100M8TileCases, kInt8H200M8TileCases, kInt8M8TileCases));
  CheckUnreportedDeviceTiles(kInt8M8TileCases);
}

// The opt-in wave-aware tile follows the device the caller reports in Params, so the RTX 5090 policy (sm_12x, 170 SMs)
// is checked on any GPU: the kernels run on this device and only the reported device differs. N = 10240 is 640
// default blocks on that device's 510 slots, so wave-aware picks the narrow tile where the grid.y cutoff alone
// keeps the default tile.
TEST_F(Fp16Int4SymmetricGroupwiseTest, Int4Group32WaveAwareTileFollowsReportedDevice) {
  if (onnxruntime::llm::common::getSMVersion() < kMinSupportedSm) {
    GTEST_SKIP() << "FP16 INT4 GEMV requires SM " << kMinSupportedSm << " or later";
  }
  constexpr int kRtx5090Sm = 120;
  constexpr int kRtx5090SmCount = 170;
  auto report = [](int sm, int sm_count, bool wave_aware) {
    return [=](wo::Params& params) {
      params.device_sm = sm;
      params.device_sm_count = sm_count;
      params.wave_aware = wave_aware;
    };
  };
  for (const auto& [m, n, expected_cta_n] : std::vector<std::tuple<int, int, int>>{
           {8, 5120, 2}, {8, 6144, 4}, {8, 10240, 2}, {7, 10240, 4}, {9, 10240, 4}}) {
    SCOPED_TRACE(testing::Message() << "M=" << m << " N=" << n);
    InitBuffers(m, n, 4096, 32);
    EXPECT_EQ(LaunchCtaN(report(kRtx5090Sm, kRtx5090SmCount, true)), expected_cta_n);
  }

  InitBuffers(8, 10240, 4096, 32);
  EXPECT_EQ(LaunchCtaN(report(kRtx5090Sm, kRtx5090SmCount, false)), 4);  // opt-in off
  EXPECT_EQ(LaunchCtaN(report(0, 0, true)), 4);                          // device not reported
  EXPECT_EQ(LaunchCtaN(report(kRtx5090Sm, 0, true)), 4);                 // SM count not reported
  EXPECT_EQ(LaunchCtaN(report(80, 108, true)), 4);                       // A100 is not sm_12x
  EXPECT_TRUE(M8MatchesTwoM4Launches(report(kRtx5090Sm, kRtx5090SmCount, true)));
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

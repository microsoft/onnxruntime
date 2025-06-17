#include <cuda_profiler_api.h>
#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include "cutlass/numeric_types.h"
#include "contrib_ops/cuda/llm/common/cuda_runtime_utils.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm/fpA_intB_gemm.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemv/fpA_intB_gemv.h"
#include "contrib_ops/cuda/quantization/matmul_nbits.cuh"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <ctime>
#include <functional>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace wo = onnxruntime::llm::kernels::fpA_intB_gemv;
using onnxruntime::llm::cutlass_extensions::CutlassGemmConfig;

struct CudaBuffer {
  void* _data;
  int _size;

  CudaBuffer(int size_in_bytes) : _size(size_in_bytes) {
    cudaMalloc(&_data, _size);
  }

  template <typename T = void>
  T* data() {
    return reinterpret_cast<T*>(_data);
  }

  void copy_to(void* dst) {
    cudaMemcpy(dst, _data, _size, cudaMemcpyDeviceToHost);
  }

  void copy_from(void* src) {
    cudaMemcpy(_data, src, _size, cudaMemcpyHostToDevice);
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
#if defined(ENABLE_BF16)
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    // bfloat16 has fewer mantissa digits so the cumulative error will be larger.
    diff_threshold *= 3.f;
  } else
#endif
  {
    diff_threshold *= 1.5f;
  }

  printf("max diff %f (diff threshold %f), avg diff %f, diff cnt %d/%d\n",
         max_diff, diff_threshold, total_diff / diff_count, diff_count, size);
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

template <typename T>
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

// Common profiling function
template <typename Func>
float run_kernel_and_measure_time(Func kernel_launcher, int warmup, int repeats) {
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  for (int i = 0; i < warmup; ++i) {
    kernel_launcher(s);
  }
  cudaEventRecord(begin, s);
  for (int i = 0; i < repeats; ++i) {
    kernel_launcher(s);
  }
  cudaEventRecord(end, s);
  cudaEventSynchronize(end);
  float time;
  cudaEventElapsedTime(&time, begin, end);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaStreamDestroy(s);
  return time / repeats;
}

template <wo::KernelType KT, typename Runner, typename Config>
void exec_cutlass_kernel(
    void* scaled_act, Runner& runner, wo::Params& params, Config& config, char* ws, size_t ws_size, cudaStream_t stream) {
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

template <wo::KernelType KT, bool has_bias = false, bool has_act_scale = false, bool filter_configs = false>
class KernelTestFixture : public ::testing::Test {
 protected:
  int m_, n_, k_, group_size_;
  int warmup_ = 10;
  int repeats_ = 30;
  cudaDeviceProp prop_;
  std::shared_ptr<CudaBuffer> d_act_, d_act_scale_, d_weight_, d_scales_, d_zeros_, d_bias_, d_out_;
  std::vector<typename cutlassTypeMapper<KT>::AType> h_act_, h_act_scale_, h_scales_, h_zeros_, h_bias_, h_out1_, h_out2_;
  std::vector<uint8_t> h_weight_;

  void SetUp() override {
    int device;
    CUDA_CALL_THROW(cudaGetDevice(&device));
    CUDA_CALL_THROW(cudaGetDeviceProperties(&prop_, device));
    std::srand(20240123);
  }

  void InitBuffers(int m, int n, int k, int group_size) {
    m_ = m;
    n_ = n;
    k_ = k;
    group_size_ = group_size;

    ORT_ENFORCE(m_ <= 16);
    if (cutlassTypeMapper<KT>::QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
      ORT_ENFORCE(group_size_ == 64 || group_size_ == 128);
      ORT_ENFORCE(k_ % group_size_ == 0);
    }

    using AType = typename cutlassTypeMapper<KT>::AType;
    static constexpr int ASizeInBits = sizeof(AType) * 8;
    static constexpr int WSizeInBits = cutlassTypeMapper<KT>::WSizeInBits;

    d_act_ = std::make_shared<CudaBuffer>(m_ * k_ * ASizeInBits / 8);
    d_act_scale_ = std::make_shared<CudaBuffer>(k_ * ASizeInBits / 8);
    d_weight_ = std::make_shared<CudaBuffer>(k_ * n_ * WSizeInBits / 8);
    d_scales_ = std::make_shared<CudaBuffer>(n_ * k_ / group_size_ * ASizeInBits / 8);
    d_zeros_ = std::make_shared<CudaBuffer>(n_ * k_ / group_size_ * ASizeInBits / 8);
    d_bias_ = std::make_shared<CudaBuffer>(n_ * ASizeInBits / 8);
    d_out_ = std::make_shared<CudaBuffer>(m_ * n_ * ASizeInBits / 8);

    h_act_.resize(m_ * k_);
    h_act_scale_.resize(k_);
    h_weight_.resize(k_ * n_);
    h_scales_.resize(n_ * k_);
    h_zeros_.resize(n_ * k_);
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

    d_act_->copy_from(h_act_.data());
    d_act_scale_->copy_from(h_act_scale_.data());
    d_weight_->copy_from(h_weight_.data());
    d_scales_->copy_from(h_scales_.data());
    d_zeros_->copy_from(h_zeros_.data());
    d_bias_->copy_from(h_bias_.data());
  }

  bool BenchmarkAndVerifyKernel() {
    printf("Kernel %s\n", cutlassTypeMapper<KT>::str(m_, n_, k_, group_size_).c_str());

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

    // Run FpA_IntB_Gemv CUDA kernel
    float time1 = run_kernel_and_measure_time(
        [&](cudaStream_t s) {
          int arch = onnxruntime::llm::common::getSMVersion();
          ORT_ENFORCE(wo::is_supported(arch, params.type));
          wo::kernel_launcher(arch, params, s);
        },
        warmup_, repeats_);
    d_out_->copy_to(h_out1_.data());

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
    std::cout << "Total " << configs.size() << " configurations" << std::endl;

    if constexpr (filter_configs) {
      configs = filter_gemm_configs<KT>(configs, k_);
      std::cout << "After filtering, " << configs.size() << " configurations left" << std::endl;
    }

    float fast_time = std::numeric_limits<float>::max();
    CutlassGemmConfig best_config = configs[0];  // Initialize with first config

    int cfg_i = 0;
    for (auto& config : configs) {
      float time = std::numeric_limits<float>::max();
      try {
        time = run_kernel_and_measure_time(
            [&](cudaStream_t s) {
              exec_cutlass_kernel<KT>(d_act_->data(), gemm_runner, params, config, ws_ptr, ws_bytes, s);
            },
            2, 5);
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
      if (time < fast_time) {
        fast_time = time;
        best_config = config;
      }
      cfg_i++;
    }

    float time2 = run_kernel_and_measure_time(
        [&](cudaStream_t s) {
          exec_cutlass_kernel<KT>(d_act_->data(), gemm_runner, params, best_config, ws_ptr, ws_bytes, s);
        },
        warmup_, repeats_);
    d_out_->copy_to(h_out2_.data());

    float quant_scale = 1.f / (1 << (cutlassTypeMapper<KT>::WSizeInBits - 1));
    bool pass = compare<AType>(h_out1_.data(), h_out2_.data(), m_ * n_, quant_scale);
    printf("cuda kernel cost time %.3f us, cutlass kernel cost time %.3f us, FpA_IntB_Gemv speedup %.2f\n\n",
           time1 * 1000, time2 * 1000, time2 / time1);

    // Run MatMulNBits cuda kernel (only for m=1).
    if constexpr (KT == wo::KernelType::FP16Int8Groupwise || KT == wo::KernelType::FP16Int4Groupwise) {
      if (m_ == 1) {
        std::vector<uint8_t> h_uint8_zeros(n_ * k_);
        for (uint8_t& v : h_uint8_zeros) {
          v = rand() % 256;
        }

        ORT_ENFORCE(k_ / group_size_ * cutlassTypeMapper<KT>::WSizeInBits % 8 == 0);
        CudaBuffer d_uint8_zeros(n_ * k_ / group_size_ * cutlassTypeMapper<KT>::WSizeInBits / 8);
        d_uint8_zeros.copy_from(h_uint8_zeros.data());

        float time3 = run_kernel_and_measure_time(
            [&](cudaStream_t s) {
              onnxruntime::contrib::cuda::TryMatMulNBits(cutlassTypeMapper<KT>::WSizeInBits,
                                                         reinterpret_cast<AType*>(d_out_->data()),
                                                         reinterpret_cast<const AType*>(d_act_->data()),
                                                         reinterpret_cast<const uint8_t*>(d_weight_->data()),
                                                         reinterpret_cast<const AType*>(d_scales_->data()),
                                                         static_cast<const uint8_t*>(d_uint8_zeros.data()),
                                                         m_, n_, k_, group_size_, prop_.sharedMemPerBlock, s);
            },
            warmup_, repeats_);

        printf("matmul %d bits kernel cost time %.3f us, FpA_IntB_Gemv speedup %.2f\n\n",
               cutlassTypeMapper<KT>::WSizeInBits, time3 * 1000, time3 / time1);
      }
    }
    return pass;
  }
};

// Define test cases using the fixture
using Fp16Int8GroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int8Groupwise>;
using Fp16Int4GroupwiseTest = KernelTestFixture<wo::KernelType::FP16Int4Groupwise>;
using Bf16Int8GroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int8Groupwise>;
using Bf16Int4GroupwiseTest = KernelTestFixture<wo::KernelType::BF16Int4Groupwise>;

TEST_F(Fp16Int8GroupwiseTest, FpA_IntB_Gemm_Fp16_Int8_Groupwise) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 75) {
    std::cout << "Skip fp16 int8 groupwise GEMM kernel for SM < 75" << std::endl;
    return;
  }

  std::vector<int> ms{1, 2, 4, 8, 10, 14};
  std::vector<std::pair<int, int>> n_k_list = {
      {5120, 3072},
      {8192, 3072},
      {3072, 8192},
      {200064, 3072}};

  for (auto m : ms) {
    for (const auto& [n, k] : n_k_list) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
      InitBuffers(m, n, k, 128);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

TEST_F(Fp16Int4GroupwiseTest, FpA_IntB_Gemm_Fp16_Int4_Groupwise) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 75) {
    std::cout << "Skip fp16 int4 groupwise GEMM kernel for SM < 75" << std::endl;
    return;
  }

  std::vector<int> ms{1, 2, 4, 8, 10, 14};
  std::vector<std::pair<int, int>> n_k_list = {
      {5120, 3072},
      {8192, 3072},
      {3072, 8192},
      {200064, 3072}};

  for (auto m : ms) {
    for (const auto& [n, k] : n_k_list) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
      InitBuffers(m, n, k, 128);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

TEST_F(Bf16Int8GroupwiseTest, FpA_IntB_Gemm_BF16_Int8_Groupwise) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 80) {
    std::cout << "Skip bf16 int8 groupwise GEMM kernel test for SM < 80" << std::endl;
    return;
  }

  std::vector<int> ms{1, 2, 4, 8, 10, 14};
  std::vector<std::pair<int, int>> n_k_list = {
      {5120, 3072},
      {8192, 3072},
      {3072, 8192},
      {200064, 3072}};

  for (auto m : ms) {
    for (const auto& [n, k] : n_k_list) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
      InitBuffers(m, n, k, 128);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

TEST_F(Bf16Int4GroupwiseTest, FpA_IntB_Gemm_BF16_Int4_Groupwise) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 80) {
    std::cout << "Skip bf16 int4 groupwise GEMM kernel test for SM < 80" << std::endl;
    return;
  }

  std::vector<int> ms{1, 2, 4, 8, 10, 14};
  std::vector<std::pair<int, int>> n_k_list = {
      {5120, 3072},
      {8192, 3072},
      {3072, 8192},
      {200064, 3072}};

  for (auto m : ms) {
    for (const auto& [n, k] : n_k_list) {
      InitBuffers(m, n, k, 64);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
      InitBuffers(m, n, k, 128);
      EXPECT_TRUE(BenchmarkAndVerifyKernel());
    }
  }
}

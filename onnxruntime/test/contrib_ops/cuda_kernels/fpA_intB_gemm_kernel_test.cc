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

struct CudaBuffer {
  void* _data;
  int _size;

  CudaBuffer(int size_in_bytes)
      : _size(size_in_bytes) {
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
float compare(void* _pa, void* _pb, int size, float scale) {
  auto pa = reinterpret_cast<T*>(_pa);
  auto pb = reinterpret_cast<T*>(_pb);
  float max_diff = 0.f, tot_diff = 0.f;
  float max_val = 0.f;
  int diff_cnt = 0;
  float threshold = 1e-7;
  for (int n = 0; n < size; ++n) {
    float va = static_cast<float>(pa[n]);
    float vb = static_cast<float>(pb[n]);
    max_val = std::max(max_val, vb);
    float diff = std::abs(va - vb);
    if (diff > threshold) {
      max_diff = std::max(max_diff, diff);
      tot_diff += diff;
      ++diff_cnt;
    }
  }
  float diff_thres = max_val * scale;
#if defined(ENABLE_BF16)
  if constexpr (std::is_same_v<T, __nv_bfloat16>) {
    // bfloat16 has fewer mantissa digits than float16(10 bits for fp16 but only 7 bits for bf16), so the cumulative
    // error will be larger.
    diff_thres *= 3.f;
  } else
#endif
  {
    diff_thres *= 1.5f;
  }
  printf("max diff %f (diff threshold %f), avg diff %f, diff cnt %d/%d\n", max_diff, diff_thres, tot_diff / diff_cnt,
         diff_cnt, size);
  return max_diff <= diff_thres;
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
std::vector<onnxruntime::llm::cutlass_extensions::CutlassGemmConfig> get_configs(T& runner, [[maybe_unused]] int k) {
  auto configs = runner.getConfigs();
  std::vector<onnxruntime::llm::cutlass_extensions::CutlassGemmConfig> rets;
  for (auto config : configs) {
    // if (config.stages >= 5) {
    //   continue;
    // }
    // if (config.split_k_style != onnxruntime::llm::cutlass_extensions::SplitKStyle::NO_SPLIT_K) {
    //   int k_size = (k + config.split_k_factor - 1) / config.split_k_factor;
    //   if (k_size % 64) {
    //     continue;
    //   }
    // }
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

float run_cuda_kernel(wo::Params& params, int warmup, int repeats) {
  int arch = onnxruntime::llm::common::getSMVersion();
  ORT_ENFORCE(wo::is_supported(arch, params.type));
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);
  for (int i = 0; i < warmup; ++i) {
    wo::kernel_launcher(arch, params, s);
  }
  cudaEventRecord(begin, s);
  for (int i = 0; i < repeats; ++i) {
    wo::kernel_launcher(arch, params, s);
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
    [[maybe_unused]] void* scaled_act, Runner& runner, wo::Params& params, Config& config, char* ws, size_t ws_size, cudaStream_t stream) {
  static constexpr cutlass::WeightOnlyQuantOp QuantOp = cutlassTypeMapper<KT>::QuantOp;
  void* act = params.act;
  // if (params.act_scale)
  // {
  //     onnxruntime::llm::kernels::apply_per_channel_scale_kernel_launcher<AType, AType>(
  //         reinterpret_cast<AType*>(scaled_act), reinterpret_cast<AType const*>(params.act),
  //         reinterpret_cast<AType const*>(params.act_scale), params.m, params.k, nullptr, stream);
  //     act = scaled_act;
  // }
  if (QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
    runner.gemm(act, params.weight, params.scales, params.zeros, params.bias, params.out, params.m, params.n,
                params.k, params.groupsize, config, ws, ws_size, stream);
  }
}

template <wo::KernelType KT>
float run_cutlass_kernel(wo::Params& params, int warmup, int repeats) {
  int arch = onnxruntime::llm::common::getSMVersion();
  ORT_ENFORCE(KT == params.type);
  ORT_ENFORCE(wo::is_supported(arch, params.type));
  using AType = typename cutlassTypeMapper<KT>::AType;
  using WType = typename cutlassTypeMapper<KT>::WType;
  CudaBuffer scaled_act(params.m * params.k * sizeof(AType));

  using onnxruntime::llm::kernels::cutlass_kernels::CutlassFpAIntBGemmRunner;
  auto runner = std::make_shared<CutlassFpAIntBGemmRunner<AType, WType, cutlassTypeMapper<KT>::QuantOp>>();
  auto& gemm = *runner;
  cudaStream_t s;
  cudaStreamCreate(&s);
  cudaEvent_t begin, end;
  cudaEventCreate(&begin);
  cudaEventCreate(&end);

  auto configs = get_configs(gemm, params.k);
  std::cout << "total " << configs.size() << " configurations" << std::endl;
  int ws_bytes = gemm.getWorkspaceSize(params.m, params.n, params.k);
  char* ws_ptr = nullptr;
  if (ws_bytes)
    cudaMalloc(&ws_ptr, ws_bytes);
  float fast_time = 1e8;
  auto best_config = configs[0];
  int cfg_i = 0;
  for (auto& config : configs) {
    float time = std::numeric_limits<float>::max();
    try {
      for (int i = 0; i < 2; ++i) {
        exec_cutlass_kernel<KT>(scaled_act.data(), gemm, params, config, ws_ptr, ws_bytes, s);
      }
      cudaEventRecord(begin, s);
      for (int i = 0; i < 5; ++i) {
        exec_cutlass_kernel<KT>(scaled_act.data(), gemm, params, config, ws_ptr, ws_bytes, s);
      }
      cudaEventRecord(end, s);
      cudaEventSynchronize(end);
      cudaEventElapsedTime(&time, begin, end);
    } catch (std::exception const& e) {
      std::ostringstream msg;
      msg << "Cannot profile configuration " << cfg_i;
      if constexpr (std::is_same_v<decltype(config), onnxruntime::llm::cutlass_extensions::CutlassGemmConfig>) {
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

  for (int i = 0; i < warmup; ++i) {
    exec_cutlass_kernel<KT>(scaled_act.data(), gemm, params, best_config, ws_ptr, ws_bytes, s);
  }
  cudaEventRecord(begin, s);
  for (int i = 0; i < repeats; ++i) {
    exec_cutlass_kernel<KT>(scaled_act.data(), gemm, params, best_config, ws_ptr, ws_bytes, s);
  }
  if (ws_ptr)
    cudaFree(ws_ptr);
  cudaEventRecord(end, s);
  cudaEventSynchronize(end);
  float time;
  cudaEventElapsedTime(&time, begin, end);
  cudaEventDestroy(begin);
  cudaEventDestroy(end);
  cudaStreamDestroy(s);
  return time / repeats;
}

template <typename T>
float run_matmul_n_bits_kernel(T* output,                   // Output C [1, N]
                               const T* a_data,             // Input A [1, K]
                               const uint8_t* b_data,       // Input B Quantized [N, K/bs, bs]
                               const T* scales_data,        // Scales [N, K/bs]
                               const uint8_t* zero_points,  // Zero Points [N, K/bs] (can be nullptr)
                               int m,                       // Rows of A and C (MUST be 1)
                               int n,                       // Columns of B and C
                               int k,                       // Columns of A / Rows of B
                               int group_size,              // Quantization block size for B
                               int bits,                    // Bits per element in B (4 or 8)
                               int warmup, int repeats,
                               const cudaDeviceProp prop) {
  ORT_ENFORCE(m == 1, "Only support m = 1 for matmul_n_bits kernel");  cudaStream_t s;
  cudaStreamCreate(&s);

  bool has_gemv = onnxruntime::contrib::cuda::TryMatMulNBits(bits, output, a_data, b_data, scales_data, zero_points,
                                                             m, n, k, group_size, prop.sharedMemPerBlock, s);
  if (!has_gemv) {
    std::cerr << "Failed to run matmul" << bits << "bits kernel." << std::endl;
    return 0.0f;
  } else {
    cudaEvent_t begin, end;
    cudaEventCreate(&begin);
    cudaEventCreate(&end);
    for (int i = 0; i < warmup; ++i) {
      onnxruntime::contrib::cuda::TryMatMulNBits(bits, output, a_data, b_data, scales_data, zero_points,
                                                 m, n, k, group_size, prop.sharedMemPerBlock, s);
    }
    cudaEventRecord(begin, s);
    for (int i = 0; i < repeats; ++i) {
      onnxruntime::contrib::cuda::TryMatMulNBits(bits, output, a_data, b_data, scales_data, zero_points,
                                                 m, n, k, group_size, prop.sharedMemPerBlock, s);
    }
    cudaEventRecord(end, s);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, begin, end);
    cudaEventDestroy(begin);
    cudaEventDestroy(end);
    return time / repeats;
  }
}

template <wo::KernelType KT, bool has_bias = false, bool has_act_scale = false>
bool benchmark_and_verify(int m, int n, int k, int group_size, int warmup, int repeats,
  [[maybe_unused]] const cudaDeviceProp& prop) {
  std::srand(20240123);
  ORT_ENFORCE(m <= 16);
  if (cutlassTypeMapper<KT>::QuantOp == cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_AND_ZEROS) {
    ORT_ENFORCE(group_size == 64 || group_size == 128);
    ORT_ENFORCE(k % group_size == 0);
  }
  using AType = typename cutlassTypeMapper<KT>::AType;
  static constexpr int ASizeInBits = sizeof(AType) * 8;
  static constexpr int WSizeInBits = cutlassTypeMapper<KT>::WSizeInBits;
  printf("Kernel %s\n", cutlassTypeMapper<KT>::str(m, n, k, group_size).c_str());

  CudaBuffer d_act(m * k * ASizeInBits / 8);
  CudaBuffer d_act_scale(k * ASizeInBits / 8);
  CudaBuffer d_weight(k * n * WSizeInBits / 8);
  CudaBuffer d_scales(n * k / group_size * ASizeInBits / 8);
  CudaBuffer d_zeros(n * k / group_size * ASizeInBits / 8);
  CudaBuffer d_bias(n * ASizeInBits / 8);
  CudaBuffer d_out(m * n * ASizeInBits / 8);
  std::vector<AType> h_act(m * k), h_act_scale(k);
  std::vector<uint8_t> h_weight(k * n);
  std::vector<AType> h_scales(n * k);
  std::vector<AType> h_zeros(n * k);
  std::vector<AType> h_bias(n);
  std::vector<AType> h_out1(m * n), h_out2(m * n);

  random_fill(h_act, -1.f, 1.f);
  random_fill(h_act_scale, -1.f, 1.f);
  random_fill(h_scales, -1.f, 1.f);
  random_fill(h_zeros, -1.f, 1.f);
  random_fill(h_bias, -1.f, 1.f);

  for (uint8_t& v : h_weight) {
    v = rand() % 256;
  }

  d_act.copy_from(h_act.data());
  d_act_scale.copy_from(h_act_scale.data());
  d_weight.copy_from(h_weight.data());
  d_scales.copy_from(h_scales.data());
  d_zeros.copy_from(h_zeros.data());
  d_bias.copy_from(h_bias.data());

  void* p_act_scale = nullptr;
  void* p_zeros = nullptr;
  void* p_bias = nullptr;

  if (group_size != 0) {
    p_zeros = d_zeros.data();
    if constexpr (has_bias) {
      p_bias = d_bias.data();
    }
    if constexpr (has_act_scale) {
      p_act_scale = d_act_scale.data();
    }
  }
  wo::Params params(d_act.data(), p_act_scale, d_weight.data(), d_scales.data(), p_zeros, p_bias, d_out.data(), 1.f,
                    m, n, k, group_size, KT);
  float time1, time2;
  time1 = run_cuda_kernel(params, warmup, repeats);
  d_out.copy_to(h_out1.data());
  time2 = run_cutlass_kernel<KT>(params, warmup, repeats);
  d_out.copy_to(h_out2.data());
  float quant_scale = 1.f / (1 << (WSizeInBits - 1));
  bool pass = compare<AType>(h_out1.data(), h_out2.data(), m * n, quant_scale);
  printf("cuda kernel cost time %.3f us, cutlass kernel cost time %.3f us, cuda speedup %.2f\n\n", time1 * 1000,
         time2 * 1000, time2 / time1);

  if constexpr (KT == wo::KernelType::FP16Int8Groupwise || KT == wo::KernelType::FP16Int4Groupwise) {
    if (m == 1) {
      std::vector<uint8_t> h_uint8_zeros(n * k);
      for (uint8_t& v : h_uint8_zeros) {
        v = rand() % 256;
      }

      ORT_ENFORCE(k / group_size * WSizeInBits % 8 == 0);
      CudaBuffer d_uint8_zeros(n * k / group_size * WSizeInBits / 8);
      d_uint8_zeros.copy_from(h_uint8_zeros.data());

      float time3 = run_matmul_n_bits_kernel<AType>(
          reinterpret_cast<AType*>(d_out.data()),
          reinterpret_cast<const AType*>(d_act.data()),
          reinterpret_cast<const uint8_t*>(d_weight.data()),
          reinterpret_cast<const AType*>(d_scales.data()),
          static_cast<const uint8_t*>(d_uint8_zeros.data()),
          m,
          n,
          k,
          group_size,
          WSizeInBits,
          warmup, repeats, prop);
      printf("matmul %d bits kernel cost time %.3f us, fpA speedup %.2f\n\n", WSizeInBits, time3 * 1000, time3 / time1);
    }
  }

  return pass;
}

TEST(KernelTest, FpA_IntB_Gemm_Fp16) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 75) {
    std::cout << "Skip fp16 intB GEMM kernel for SM < 75" << std::endl;
    return;
  }

  bool pass;
  constexpr int warmup = 10;
  constexpr int repeats = 30;
  std::vector<int> ms{1, 2, 4, 8, 10, 14};

  std::vector<std::pair<int, int>> n_k_list = {
      {5120, 3072},
      {8192, 3072},
      {3072, 8192},
      {200064, 3072}};

  int device;
  CUDA_CALL_THROW(cudaGetDevice(&device));

  cudaDeviceProp prop;
  CUDA_CALL_THROW(cudaGetDeviceProperties(&prop, device));


  for (auto m : ms) {
    for (const auto& [n, k] : n_k_list) {
      pass = benchmark_and_verify<wo::KernelType::FP16Int8Groupwise>(m, n, k, 64, warmup, repeats, prop);
      EXPECT_TRUE(pass);
      pass = benchmark_and_verify<wo::KernelType::FP16Int8Groupwise>(m, n, k, 128, warmup, repeats, prop);
      EXPECT_TRUE(pass);
      pass = benchmark_and_verify<wo::KernelType::FP16Int4Groupwise>(m, n, k, 64, warmup, repeats, prop);
      EXPECT_TRUE(pass);
      pass = benchmark_and_verify<wo::KernelType::FP16Int4Groupwise>(m, n, k, 128, warmup, repeats, prop);
      EXPECT_TRUE(pass);
    }
  }
}

TEST(KernelTest, FpA_IntB_Gemm_BF16) {
  int const arch = onnxruntime::llm::common::getSMVersion();
  if (arch < 80) {
    std::cout << "Skip bf16 intB GEMM  kernel test for SM < 80" << std::endl;
    return;
  }

  bool pass;
  constexpr int warmup = 10;
  constexpr int repeats = 30;
  std::vector<int> ms{1, 2, 4, 8, 10, 14};
  std::vector<std::pair<int, int>> n_k_list = {
      {5120, 3072},
      {8192, 3072},
      {3072, 8192},
      {200064, 3072}};

  int device;
  CUDA_CALL_THROW(cudaGetDevice(&device));

  cudaDeviceProp prop;
  CUDA_CALL_THROW(cudaGetDeviceProperties(&prop, device));

  for (auto m : ms) {
    for (const auto& [n, k] : n_k_list) {
      pass = benchmark_and_verify<wo::KernelType::BF16Int8Groupwise>(m, n, k, 64, warmup, repeats, prop);
      EXPECT_TRUE(pass);
      pass = benchmark_and_verify<wo::KernelType::BF16Int8Groupwise>(m, n, k, 128, warmup, repeats, prop);
      EXPECT_TRUE(pass);
      pass = benchmark_and_verify<wo::KernelType::BF16Int4Groupwise>(m, n, k, 64, warmup, repeats, prop);
      EXPECT_TRUE(pass);
      pass = benchmark_and_verify<wo::KernelType::BF16Int4Groupwise>(m, n, k, 128, warmup, repeats, prop);
      EXPECT_TRUE(pass);
    }
  }
}

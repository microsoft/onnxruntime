// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License
#include <unordered_map>
#include <mutex>

#include "dnnl_util.h"
#include "dnnl.hpp"
#include "dnnl_types.h"
#include "core/common/common.h"
#include "core/providers/shared_library/provider_api.h"

namespace onnxruntime {
namespace ort_dnnl {
namespace dnnl_util {


enum GPUInfo {
  AVALIBLITY,
  BF16SUPPORT
};

std::once_flag flag1, flag2;
// dnnl::engin::kind::gpu represents actual HW and we want to limit how much we instantiate the hardware
// This code has been designed so that it can be called multiple times.  The engine will only be created
// the first call.
// Wrapped in the `call_once` code we create a gpu engine.
//   if GPU engine is successful created set gpuRuntimeFound=true
//   Use the engine to create a bfloat16 matmul primitive if successful set gpuBF16Supported=true
bool GetGPUInfo(GPUInfo gpu_info) {
  static bool gpuRuntimeFound = false;
  static bool gpuBF16Supported = false;
#ifdef DNNL_GPU_RUNTIME
#if (DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL) || (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL)
  std::call_once(flag1, []() {
    dnnl::engine gpu_engine;
    if (dnnl_engine_get_count(dnnl_engine_kind_t::dnnl_gpu)) {
      gpu_engine = dnnl::engine(dnnl::engine::kind::gpu, 0);
    }
    if (gpu_engine) {
      gpuRuntimeFound = true;
      // attempt to make a dnnl::matmul::desc. If we are able to successfully make a bf16 matmul::desc
      // assume the GPU supports all BF16 operations.
      auto src0_md = dnnl::memory::desc({1,1}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::ab);
      auto src1_md = dnnl::memory::desc({1,1}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::ab);
      auto dst_md = dnnl::memory::desc({1,1}, dnnl::memory::data_type::bf16, dnnl::memory::format_tag::ab);
      auto matmul_d = dnnl::matmul::desc(src0_md, src1_md, dst_md);
      try {
        auto matmul_pd = dnnl::matmul::primitive_desc(matmul_d, gpu_engine);
        gpuBF16Supported = true;
      } catch(const dnnl::error& e) {
        if (e.status == dnnl_unimplemented) {
          gpuBF16Supported = false;
        }
      }
    }
    });
#endif // (DNNL_GPU_RUNTIME == DNNL_RUNTIME_OCL) || (DNNL_GPU_RUNTIME == DNNL_RUNTIME_SYCL)
#endif  // defined(DNNL_GPU_RUNTIME)
  switch (gpu_info)
  {
  case AVALIBLITY: {
    return gpuRuntimeFound;
    break;
  }
  case BF16SUPPORT: {
    return gpuBF16Supported;
    break;
  }
  default:
    return false;
    break;
  }
}

bool IsGPURuntimeAvalible(){
  return GetGPUInfo(AVALIBLITY);
}

bool IsGPUBF16Supported() {
  return GetGPUInfo(BF16SUPPORT);
}

bool IsBF16Supported() {
  static bool use_all_bf16_hardware = false;
  if(IsGPURuntimeAvalible() && IsGPUBF16Supported()){
    return true;
  }
  std::call_once(flag2, []() {
    const std::string bf16_env = onnxruntime::GetEnvironmentVar("ORT_DNNL_USE_ALL_BF16_HW");
    if (!bf16_env.empty()) {
      use_all_bf16_hardware = (std::stoi(bf16_env) == 0 ? false : true);
    }
      });

  // HasAVX512Skylake checks for AVX512BW which can run bfloat16 but
  // is slower than float32 by 3x to 4x.
  // By default the AVX512BW ISA is not used. It is still useful for validation
  // so it can be enabled by setting the environment variable ORT_DNNL_USE_ALL_BF16_HW=1
  if (use_all_bf16_hardware && CPUIDInfo::GetCPUIDInfo().HasAVX512Skylake()) {
    return true;
  }
  
  // If AVX512-BF16 or AMX-BF16 exist then bfloat16 ops are HW accelerated
  if (CPUIDInfo::GetCPUIDInfo().HasAVX512_BF16() ||
      CPUIDInfo::GetCPUIDInfo().HasAMX_BF16()) {
    return true;
  }
  return false;
}

dnnl::algorithm OrtOperatorToDnnlAlgorithm(std::string op) {
  std::unordered_map<std::string, dnnl::algorithm> operator_to_algorithm = {
          // binary algorithms
          {"Add", dnnl::algorithm::binary_add},
          {"Mul", dnnl::algorithm::binary_mul},
          {"Sub", dnnl::algorithm::binary_sub},
          {"Div", dnnl::algorithm::binary_div},
          // eltwise algorithms
          {"Abs", dnnl::algorithm::eltwise_abs},
          {"BiasGelu", dnnl::algorithm::eltwise_gelu_erf},
          {"Elu", dnnl::algorithm::eltwise_elu},  // algorithm requires alpha value
          {"Equal", dnnl::algorithm::binary_eq},
          {"Exp", dnnl::algorithm::eltwise_exp},
          {"FastGelu", dnnl::algorithm::eltwise_gelu_tanh},
          {"Gelu", dnnl::algorithm::eltwise_gelu_erf},
          {"Greater", dnnl::algorithm::binary_gt},
          {"GreaterOrEqual", dnnl::algorithm::binary_ge},
          {"LeakyRelu", dnnl::algorithm::eltwise_relu},  // algorithm requires alpha value
          {"Less", dnnl::algorithm::binary_lt},
          {"LessOrEqual", dnnl::algorithm::binary_le},
          {"Log", dnnl::algorithm::eltwise_log},
          {"Relu", dnnl::algorithm::eltwise_relu},
          {"Round", dnnl::algorithm::eltwise_round},
          // OneDNN eltwise_logistic is defined as 1/(1 + exp(-x)) which matches the definition of "Sigmoid" in ONNX
          {"Sigmoid", dnnl::algorithm::eltwise_logistic},
          // OneDNN eltwise_soft_relu is defined as ln(1 + exp(x)) which matches the definition of "Softplus" in ONNX
          {"Softplus", dnnl::algorithm::eltwise_soft_relu},
          {"Sqrt", dnnl::algorithm::eltwise_sqrt},
          {"Tanh", dnnl::algorithm::eltwise_tanh},
          // Reduction algorithms
          {"ReduceL1", dnnl::algorithm::reduction_norm_lp_power_p_sum},
          {"ReduceL2", dnnl::algorithm::reduction_norm_lp_sum},
          {"ReduceLogSum", dnnl::algorithm::reduction_sum},
          {"ReduceLogSumExp", dnnl::algorithm::reduction_sum},
          {"ReduceMax", dnnl::algorithm::reduction_max},
          {"ReduceMean", dnnl::algorithm::reduction_mean},
          {"ReduceMin", dnnl::algorithm::reduction_min},
          {"ReduceProd", dnnl::algorithm::reduction_mul},
          {"ReduceSum", dnnl::algorithm::reduction_sum},
          {"ReduceSumSquare", dnnl::algorithm::reduction_sum}
        };

  auto found = operator_to_algorithm.find(op);
  if (found == operator_to_algorithm.end()) {
    ORT_THROW("op type not supported");
  } else {
    return found->second;
  }
}

}  // namespace dnnl_util
}  // namespace ort_dnnl
}  // namespace onnxruntime

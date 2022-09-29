// Copyright(C) 2022 Intel Corporation
// Licensed under the MIT License

#include "dnnl_util.h"
#include "dnnl.hpp"
#include <unordered_map>
#include "core/common/common.h"

namespace onnxruntime {
namespace ort_dnnl {
namespace dnnl_util {

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
          {"Tanh", dnnl::algorithm::eltwise_tanh}
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

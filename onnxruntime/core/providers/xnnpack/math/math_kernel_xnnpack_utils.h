// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once
#include <optional>
#include "core/common/status.h"
#include "core/providers/xnnpack/detail/utils.h"


namespace onnxruntime {
namespace xnnpack {

namespace kernel_utils {
enum class ElementWiseOpTypeEnum : uint8_t {
  OP_INVALID,
  // unary ops
  OP_UNARY_START,
  OP_ABS,
  OP_NEG,
  OP_FLOOR,
  OP_Round,
  OP_CEIL,
  OP_SQRT,
  OP_SQUARE,
  OP_POW,
  OP_LOGISTIC,
  OP_TRUNCATE,
  // binary ops
  OP_BINARY_START,
  OP_ADD,
  OP_MUL,
  OP_DIV,
  OP_SUB,
  OP_MAX,
  OP_MIN,
  // activation ops
  OP_ACTIVATION_START,
  OP_CLAMP,       // f(x) = min(max(x, min), max)
  OP_PRELU,       // f(x) = slope * x for x < 0, f(x) = x for x >= 0
  OP_LEAKY_RELU,  // f(x) = alpha * x for x < 0, f(x) = x for x >= 0
  OP_ELU,         // f(x) = alpha * (exp(x) - 1.) for x < 0, f(x) = x for x >= 0
  OP_HARD_SWISH,  // y = x * max(0, min(1, alpha * x + beta))
                  // = x * HardSigmoid<alpha, beta>(x), where alpha = 1/6 and beta = 0.5
  OP_SIGMOID,     // y = 1 / (1 + exp(-x))
  OP_TANH,        // hyperbolic tangent of the given input tensor element-wise
};

union ActivationParam {
  struct Clip {
    float min;
    float max;
  } clip;
  struct PRelu {
    float alpha;
  } prelu;
  struct LeakyRelu {
    float alpha;
  } leaky_relu;
  struct Elu {
    float alpha;
  } elu;
  float value[2];
};

Status Createkernel(struct xnn_operator*& op,
                    const std::optional<std::pair<float, float>>& clip_min_max,
                    ElementWiseOpTypeEnum OPNameType,
                    OpComputeType op_precision_type,
                    OpQuantParam quant_param);

Status Setupkernel(xnn_operator_t op,
                   ElementWiseOpTypeEnum OPNameType,
                   OpComputeType op_precision_type,
                   size_t num_input1_dims,
                   const size_t* input1_shape,
                   size_t num_input2_dims,
                   const size_t* input2_shape,
                   const void* input1,
                   const void* input2,
                   void* output,
                   pthreadpool_t threadpool);

Status Createkernel(struct xnn_operator*& op,
                    ElementWiseOpTypeEnum OPNameType,
                    OpComputeType op_precision_type,
                    size_t channels,
                    size_t input_stride,
                    size_t output_stride);

Status Setupkernel(xnn_operator_t op,
                   ElementWiseOpTypeEnum OPNameType,
                   OpComputeType op_precision_type,
                   size_t batch_size,
                   const void* input,
                   void* output,
                   pthreadpool_t threadpool);

Status Createkernel(struct xnn_operator*& op,
                    ElementWiseOpTypeEnum OPNameType,
                    OpComputeType op_precision_type);
Status Createkernel(struct xnn_operator*& op,
                    ElementWiseOpTypeEnum OPNameType,
                    OpComputeType op_precision_type,
                    size_t channels,
                    size_t input_stride,
                    size_t output_stride,
                    const ActivationParam& activation_param,
                    OpQuantParam quant_param);

}  // namespace kernel_utils
}  // namespace xnnpack
}  // namespace onnxruntime

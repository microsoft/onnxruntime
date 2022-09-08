// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/math/math_kernel_xnnpack_utils.h"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <utility>
#include <variant>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/providers/xnnpack/math/activation.h"
#include "onnxruntime_c_api.h"
#include "xnnpack.h"

namespace onnxruntime {
namespace xnnpack {

namespace kernel_utils {
using XnnOpEnum = ElementWiseOpTypeEnum;

struct UnaryOpCreateFunc {
  std::string_view name;
  xnn_status (*float_func)(size_t, size_t, size_t, uint32_t, xnn_operator_t*);
  xnn_status (*half_func)(size_t, size_t, size_t, uint32_t, xnn_operator_t*);
};

struct UnaryOpSetupFunc {
  std::string_view name;
  xnn_status (*float_func)(xnn_operator_t, size_t, const float*, float*, pthreadpool_t);
  xnn_status (*half_func)(xnn_operator_t, size_t, const void*, void*, pthreadpool_t);
};

struct BinaryOpCreateFunc {
  std::string_view name;
  xnn_status (*float_func)(float, float, uint32_t, xnn_operator_t*);
  xnn_status (*half_func)(float, float, uint32_t, xnn_operator_t*);
  xnn_status (*qu8_func)(uint8_t, float, uint8_t, float, uint8_t, float,
                         uint8_t, uint8_t, uint32_t, xnn_operator_t*);
  xnn_status (*qs8_func)(int8_t, float, int8_t, float, int8_t, float,
                         int8_t, int8_t, uint32_t, xnn_operator_t*);
};

struct BinaryOpSetupFunc {
  std::string_view name;
  xnn_status (*float_func)(xnn_operator_t, size_t, const size_t*, size_t,
                           const size_t*, const float*, const float*, float*, pthreadpool_t);
  xnn_status (*half_func)(xnn_operator_t, size_t, const size_t*, size_t,
                          const size_t*, const void*, const void*, void*, pthreadpool_t);
  xnn_status (*qu8_func)(
      xnn_operator_t, size_t, const size_t*, size_t, const size_t*,
      const uint8_t*, const uint8_t*, uint8_t*, pthreadpool_t);
  xnn_status (*qs8_func)(
      xnn_operator_t, size_t, const size_t*, size_t, const size_t*,
      const int8_t*, const int8_t*, int8_t*, pthreadpool_t);
};

struct MinMaxOpCreateFunc {
  std::string_view name;
  xnn_status (*float_func)(uint32_t, xnn_operator_t*);
  xnn_status (*half_func)(uint32_t, xnn_operator_t*);
};

struct MinMaxOpSetupFunc {
  std::string_view name;
  xnn_status (*float_func)(xnn_operator_t, size_t, const size_t*, size_t,
                           const size_t*, const float*, const float*, float*, pthreadpool_t);
  xnn_status (*half_func)(xnn_operator_t, size_t, const size_t*, size_t,
                          const size_t*, const void*, const void*, void*, pthreadpool_t);
};

struct ActivationOpCreateFunc {
  std::string_view name;
  // float
  typedef xnn_status (*float_func_2f)(size_t, size_t, size_t, float, float, uint32_t, xnn_operator_t*);
  typedef xnn_status (*float_func_1f)(size_t, size_t, size_t, float, uint32_t, xnn_operator_t*);
  typedef xnn_status (*float_func_0f)(size_t, size_t, size_t, uint32_t, xnn_operator_t*);
  typedef xnn_status (*float_func_vf)(size_t, size_t, size_t, const float*, uint32_t, xnn_caches_t, xnn_operator_t*);

  //half
  typedef xnn_status (*half_func_vf)(size_t, size_t, size_t, const void*, uint32_t, xnn_caches_t, xnn_operator_t*);

  // uint8
  typedef xnn_status (*qu8_create_f1f)(size_t, size_t, size_t, float, uint8_t, float, uint8_t,
                                       float, uint32_t, xnn_operator_t*);
  typedef xnn_status (*qu8_create_f1_range)(size_t, size_t, size_t, uint8_t, float, uint8_t,
                                                     float, uint8_t, uint8_t, uint32_t, xnn_operator_t*);
  typedef xnn_status (*qu8_create_f0_range)(size_t, size_t, size_t, uint8_t, uint8_t, uint32_t, xnn_operator_t*);

  // int8
  typedef xnn_status (*qs8_create_f1_f_range)(size_t, size_t, size_t, float, int8_t, float, int8_t,
                                        float, int8_t, int8_t, uint32_t, xnn_operator_t*);
  typedef xnn_status (*qs8_create_f1_f)(size_t, size_t, size_t, float, int8_t, float, int8_t,
                                       float, uint32_t, xnn_operator_t*);
  typedef xnn_status (*qs8_create_f1_range)(size_t, size_t, size_t, int8_t, float, int8_t,
                                            float, int8_t, int8_t, uint32_t, xnn_operator_t*);
  typedef xnn_status (*qs8_create_f0_range)(size_t, size_t, size_t, int8_t, int8_t, uint32_t, xnn_operator_t*);

  std::variant<float_func_2f, float_func_1f, float_func_0f, float_func_vf> float_func;
  std::variant<float_func_2f, float_func_1f, float_func_0f, half_func_vf> half_func;
  std::variant<qu8_create_f1f, qu8_create_f1_range, qu8_create_f0_range> qu8_func;
  std::variant<qs8_create_f1_f_range, qs8_create_f1_f, qs8_create_f1_range, qs8_create_f0_range> qs8_func;
};

struct ActivationOpSetupFunc {
  std::string_view name;
  xnn_status (*float_func)(xnn_operator_t, size_t, const float*, float*,
                           pthreadpool_t);
  xnn_status (*half_func)(xnn_operator_t, size_t, const void*, void*,
                          pthreadpool_t);
  xnn_status (*qu8_func)(xnn_operator_t, size_t, const uint8_t*, uint8_t*,
                         pthreadpool_t);
  xnn_status (*qs8_func)(xnn_operator_t, size_t, const int8_t*, int8_t*,
                         pthreadpool_t);
};

const InlinedHashMap<ElementWiseOpTypeEnum, BinaryOpCreateFunc>
    AddSubMulDivCreateFuncMap = {
        {ElementWiseOpTypeEnum::OP_ADD,
         {"Add", xnn_create_add_nd_f32, xnn_create_add_nd_f16, xnn_create_add_nd_qu8, xnn_create_add_nd_qs8}},
        {ElementWiseOpTypeEnum::OP_SUB,
         {"Sub", xnn_create_subtract_nd_f32, xnn_create_subtract_nd_f16, xnn_create_subtract_nd_qu8, xnn_create_subtract_nd_qs8}},
        {ElementWiseOpTypeEnum::OP_MUL,
         {"Mul", xnn_create_multiply_nd_f32, xnn_create_multiply_nd_f16, xnn_create_multiply_nd_qu8, xnn_create_multiply_nd_qs8}},
        {ElementWiseOpTypeEnum::OP_DIV,
         {"Div", xnn_create_divide_nd_f32, xnn_create_divide_nd_f16,nullptr,nullptr}},
};

const InlinedHashMap<ElementWiseOpTypeEnum, BinaryOpSetupFunc>
    AddSubMulDivSetupFuncMap = {
        {ElementWiseOpTypeEnum::OP_ADD,
         {"Add", xnn_setup_add_nd_f32, xnn_setup_add_nd_f16, xnn_setup_add_nd_qu8, xnn_setup_add_nd_qs8}},
        {ElementWiseOpTypeEnum::OP_SUB,
         {"Sub", xnn_setup_subtract_nd_f32, xnn_setup_subtract_nd_f16, xnn_setup_subtract_nd_qu8, xnn_setup_subtract_nd_qs8}},
        {ElementWiseOpTypeEnum::OP_MUL,
         {"Mul", xnn_setup_multiply_nd_f32, xnn_setup_multiply_nd_f16, xnn_setup_multiply_nd_qu8, xnn_setup_multiply_nd_qs8}},
        {ElementWiseOpTypeEnum::OP_DIV,
         {"Div", xnn_setup_divide_nd_f32, xnn_setup_divide_nd_f16, nullptr,nullptr}},
};

const InlinedHashMap<ElementWiseOpTypeEnum, MinMaxOpCreateFunc>
    MinMaxCreateFuncMap = {
        {ElementWiseOpTypeEnum::OP_MAX,
         {"Max", xnn_create_maximum_nd_f32, xnn_create_maximum_nd_f16}},
        {ElementWiseOpTypeEnum::OP_MIN,
         {"Min", xnn_create_minimum_nd_f32, xnn_create_minimum_nd_f16}},
};

const InlinedHashMap<ElementWiseOpTypeEnum, MinMaxOpSetupFunc>
    MinMaxSetupFuncMap = {
        {ElementWiseOpTypeEnum::OP_MAX,
         {"Max", xnn_setup_maximum_nd_f32, xnn_setup_maximum_nd_f16}},
        {ElementWiseOpTypeEnum::OP_MIN,
         {"Min", xnn_setup_minimum_nd_f32, xnn_setup_minimum_nd_f16}},
};

const InlinedHashMap<ElementWiseOpTypeEnum, UnaryOpCreateFunc>
    UnaryCreateFuncMap = {
        {ElementWiseOpTypeEnum::OP_ABS,
         {"Abs", xnn_create_abs_nc_f32, xnn_create_abs_nc_f16}},
        {ElementWiseOpTypeEnum::OP_Round,
         {"Round", xnn_create_bankers_rounding_nc_f32, xnn_create_bankers_rounding_nc_f16}},
        {ElementWiseOpTypeEnum::OP_CEIL,
         {"Ceil", xnn_create_ceiling_nc_f32, xnn_create_ceiling_nc_f16}},
        {ElementWiseOpTypeEnum::OP_FLOOR,
         {"Floor", xnn_create_floor_nc_f32, xnn_create_floor_nc_f16}},
        {ElementWiseOpTypeEnum::OP_NEG,
         {"Neg", xnn_create_negate_nc_f32, xnn_create_negate_nc_f16}},
        {ElementWiseOpTypeEnum::OP_SQRT,
         {"Sqrt", xnn_create_square_root_nc_f32, xnn_create_square_root_nc_f16}},
        {ElementWiseOpTypeEnum::OP_TRUNCATE,
         {"Truncate", xnn_create_truncation_nc_f32, xnn_create_truncation_nc_f16}},
        {ElementWiseOpTypeEnum::OP_SQUARE,
         {"Square", xnn_create_square_nc_f32, xnn_create_square_nc_f16}},
};

const InlinedHashMap<ElementWiseOpTypeEnum, UnaryOpSetupFunc>
    UnarySetupFuncMap = {
        {ElementWiseOpTypeEnum::OP_ABS,
         {"Abs", xnn_setup_abs_nc_f32, xnn_setup_abs_nc_f16}},
        {ElementWiseOpTypeEnum::OP_Round,
         {"Round", xnn_setup_bankers_rounding_nc_f32, xnn_setup_bankers_rounding_nc_f16}},
        {ElementWiseOpTypeEnum::OP_CEIL,
         {"Ceil", xnn_setup_ceiling_nc_f32, xnn_setup_ceiling_nc_f16}},
        {ElementWiseOpTypeEnum::OP_FLOOR,
         {"Floor", xnn_setup_floor_nc_f32, xnn_setup_floor_nc_f16}},
        {ElementWiseOpTypeEnum::OP_NEG,
         {"Neg", xnn_setup_negate_nc_f32, xnn_setup_negate_nc_f16}},
        {ElementWiseOpTypeEnum::OP_SQRT,
         {"Sqrt", xnn_setup_square_root_nc_f32, xnn_setup_square_root_nc_f16}},
        {ElementWiseOpTypeEnum::OP_TRUNCATE,
         {"Truncate", xnn_setup_truncation_nc_f32, xnn_setup_truncation_nc_f16}},
        {ElementWiseOpTypeEnum::OP_SQUARE,
         {"Square", xnn_setup_square_nc_f32, xnn_setup_square_nc_f16}},
};

xnn_status float_func_0f_default(size_t, size_t, size_t, uint32_t, xnn_operator_t*) {
  ORT_NOT_IMPLEMENTED("float_func_0f is not implemented");
}
xnn_status qu8_create_f1_default(size_t, size_t, size_t, float, uint8_t, float, uint8_t,
                                 float, uint32_t, xnn_operator_t*) {
  ORT_NOT_IMPLEMENTED("qu8_create_f1_default is not implemented");
}

xnn_status qs8_create_f0_default(size_t, size_t, size_t, int8_t, int8_t, uint32_t, xnn_operator_t*) {
  ORT_NOT_IMPLEMENTED("qs8_create_f0_default is not implemented");
}

const InlinedHashMap<ElementWiseOpTypeEnum, ActivationOpCreateFunc>
    ActivationCreateFuncMap = {
        {ElementWiseOpTypeEnum::OP_CLAMP,
         {"Clip", xnn_create_clamp_nc_f32, xnn_create_clamp_nc_f16, xnn_create_clamp_nc_u8, xnn_create_clamp_nc_s8}},

        {ElementWiseOpTypeEnum::OP_PRELU,
         {"PRelu", xnn_create_prelu_nc_f32, xnn_create_prelu_nc_f16, qu8_create_f1_default, qs8_create_f0_default}},
        {ElementWiseOpTypeEnum::OP_LEAKY_RELU,
         {"LeakyRelu", xnn_create_leaky_relu_nc_f32, xnn_create_leaky_relu_nc_f16, xnn_create_leaky_relu_nc_qu8, xnn_create_leaky_relu_nc_qs8}},
        {ElementWiseOpTypeEnum::OP_ELU,
         {"Elu", xnn_create_elu_nc_f32, xnn_create_elu_nc_f16, qu8_create_f1_default, xnn_create_elu_nc_qs8}},
        {ElementWiseOpTypeEnum::OP_HARD_SWISH,
         {"HardSwish", xnn_create_hardswish_nc_f32, xnn_create_hardswish_nc_f16, qu8_create_f1_default, qs8_create_f0_default}},

        {ElementWiseOpTypeEnum::OP_SIGMOID,
         {"Sigmoid", xnn_create_sigmoid_nc_f32, xnn_create_sigmoid_nc_f16, xnn_create_sigmoid_nc_qu8, xnn_create_sigmoid_nc_qs8}},
        {ElementWiseOpTypeEnum::OP_TANH,
         {"Tanh", float_func_0f_default, float_func_0f_default, xnn_create_tanh_nc_qu8, xnn_create_tanh_nc_qs8}},
};

const InlinedHashMap<ElementWiseOpTypeEnum, ActivationOpSetupFunc>
    ActivationSetupFuncMap = {
        {ElementWiseOpTypeEnum::OP_CLAMP,
         {"Clip", xnn_setup_clamp_nc_f32, xnn_setup_clamp_nc_f16, xnn_setup_clamp_nc_u8, xnn_setup_clamp_nc_s8}},
        {ElementWiseOpTypeEnum::OP_PRELU,
         {"PRelu", xnn_setup_prelu_nc_f32, xnn_setup_prelu_nc_f16, nullptr, nullptr}},
        {ElementWiseOpTypeEnum::OP_LEAKY_RELU,
         {"LeakyRelu", xnn_setup_leaky_relu_nc_f32, xnn_setup_leaky_relu_nc_f16, xnn_setup_leaky_relu_nc_qu8, xnn_setup_leaky_relu_nc_qs8}},
        {ElementWiseOpTypeEnum::OP_ELU,
         {"Elu", xnn_setup_elu_nc_f32, xnn_setup_elu_nc_f16, nullptr, xnn_setup_elu_nc_qs8}},
        {ElementWiseOpTypeEnum::OP_HARD_SWISH,
         {"HardSwish", xnn_setup_hardswish_nc_f32, xnn_setup_hardswish_nc_f16, nullptr, nullptr}},
        {ElementWiseOpTypeEnum::OP_SIGMOID,
         {"Sigmoid", xnn_setup_sigmoid_nc_f32, xnn_setup_sigmoid_nc_f16, xnn_setup_sigmoid_nc_qu8, xnn_setup_sigmoid_nc_qs8}},
        {ElementWiseOpTypeEnum::OP_TANH,
         {"Tanh", nullptr, nullptr, xnn_setup_tanh_nc_qu8, xnn_setup_tanh_nc_qs8}},
};

Status CreateBinaryArithmetickernelBase(struct xnn_operator*& op,
                                        const std::optional<std::pair<float, float>>& clip_min_max,
                                        OpComputeType op_precision_type,
                                        OpQuantParam quant_param,
                                        const BinaryOpCreateFunc& create_func) {
  float foutput_min = clip_min_max ? clip_min_max->first : -INFINITY;
  float foutput_max = clip_min_max ? clip_min_max->second : INFINITY;
  xnn_status status = xnn_status::xnn_status_uninitialized;

  switch (op_precision_type) {
    case OpComputeType::op_compute_type_fp32:
      status = create_func.float_func(foutput_min, foutput_max, 0, &op);
      break;
    case OpComputeType::op_compute_type_fp16:
      status = create_func.half_func(foutput_min, foutput_max, 0, &op);
      break;
    case OpComputeType::op_compute_type_qu8: {
      const float output_scale = quant_param[2].first[0];
      const uint8_t output_zero_point = quant_param[2].second;
      const uint8_t output_min = xnnpack::xnn_u8s8_quantize<uint8_t>(foutput_min, output_scale, output_zero_point);
      const uint8_t output_max = xnnpack::xnn_u8s8_quantize<uint8_t>(foutput_max, output_scale, output_zero_point);
      status = create_func.qu8_func(quant_param[0].second, quant_param[0].first[0],
                           quant_param[1].second, quant_param[1].first[0],
                           quant_param[2].second, quant_param[2].first[0],
                           output_min, output_max, 0, &op);
      break;
    }
    case OpComputeType::op_compute_type_qs8: {
      const float output_scale = quant_param[2].first[0];
      const int8_t output_zero_point = quant_param[2].second;
      const int8_t output_min = xnnpack::xnn_u8s8_quantize<int8_t>(foutput_min, output_scale, output_zero_point);
      const int8_t output_max = xnnpack::xnn_u8s8_quantize<int8_t>(foutput_max, output_scale, output_zero_point);
      status = create_func.qs8_func(quant_param[0].second, quant_param[0].first[0],
                           quant_param[1].second, quant_param[1].first[0],
                           quant_param[2].second, quant_param[2].first[0],
                           output_min, output_max, 0, &op);
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op precision type for: ",
                             create_func.name, " ", OpTypeToString(op_precision_type));
  }
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to create xnnpack kernel. xnn_create_", create_func.name, "_nd_",
                           OpTypeToString(op_precision_type), " returned ", status);
  }
  return Status::OK();
}

Status SetupBinaryArithmetickernelBase(xnn_operator_t op,
                                 OpComputeType op_precision_type,
                                 size_t num_input1_dims,
                                 const size_t* input1_shape,
                                 size_t num_input2_dims,
                                 const size_t* input2_shape,
                                 const void* input1,
                                 const void* input2,
                                 void* output,
                                 pthreadpool_t threadpool,
                                 const BinaryOpSetupFunc& setup_func) {
  xnn_status status = xnn_status::xnn_status_uninitialized;

  switch (op_precision_type) {
    case OpComputeType::op_compute_type_fp32:
      status = setup_func.float_func(op, num_input1_dims, input1_shape, num_input2_dims, input2_shape,
                                     static_cast<const float*>(input1),
                                     static_cast<const float*>(input2), static_cast<float*>(output), threadpool);
      break;
    case OpComputeType::op_compute_type_fp16:
      status = setup_func.half_func(op, num_input1_dims, input1_shape, num_input2_dims, input2_shape,
                                    input1, input2, output, threadpool);
      break;
    case OpComputeType::op_compute_type_qu8: {
      status = setup_func.qu8_func(op, num_input1_dims, input1_shape, num_input2_dims, input2_shape,
                                   static_cast<const uint8_t*>(input1),
                                   static_cast<const uint8_t*>(input2), static_cast<uint8_t*>(output), threadpool);
      break;
    }
    case OpComputeType::op_compute_type_qs8: {
      status = setup_func.qs8_func(op, num_input1_dims, input1_shape, num_input2_dims, input2_shape,
                                   static_cast<const int8_t*>(input1),
                                   static_cast<const int8_t*>(input2), static_cast<int8_t*>(output), threadpool);
      break;
    }
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op precision type for : ",
                             setup_func.name, " ", OpTypeToString(op_precision_type));
  }
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to setup xnnpack kernel. xnn_setup_", setup_func.name, "_nd_",
                           OpTypeToString(op_precision_type), " returned ", status);
  }
  return Status::OK();
}

Status CreateMinMaxkernelBase(struct xnn_operator*& op,
                              OpComputeType op_precision_type,
                              const MinMaxOpCreateFunc& create_func) {
  xnn_status status = xnn_status::xnn_status_uninitialized;

  switch (op_precision_type) {
    case OpComputeType::op_compute_type_fp32:
      status = create_func.float_func(0, &op);
      break;
    case OpComputeType::op_compute_type_fp16:
      status = create_func.half_func( 0, &op);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op precision type for: ",
                             create_func.name, " ", OpTypeToString(op_precision_type));
  }
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to create xnnpack kernel. xnn_create_", create_func.name, "_nd_",
                           OpTypeToString(op_precision_type), " returned ", status);
  }
  return Status::OK();
}

Status SetupMinMaxkernelBase(struct xnn_operator*& op,
                             OpComputeType op_precision_type,
                             size_t num_input1_dims,
                             const size_t* input1_shape,
                             size_t num_input2_dims,
                             const size_t* input2_shape,
                             const void* input1,
                             const void* input2,
                             void* output,
                             pthreadpool_t threadpool,
                             const MinMaxOpSetupFunc& setup_func) {
  xnn_status status = xnn_status::xnn_status_uninitialized;

  switch (op_precision_type) {
    case OpComputeType::op_compute_type_fp32:
      status = setup_func.float_func(op, num_input1_dims, input1_shape, num_input2_dims, input2_shape,
                                     static_cast<const float*>(input1),
                                     static_cast<const float*>(input2), static_cast<float*>(output), threadpool);
      break;
    case OpComputeType::op_compute_type_fp16:
      status = setup_func.half_func(op, num_input1_dims, input1_shape, num_input2_dims, input2_shape,
                                    input1, input2, output, threadpool);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op precision type for : ",
                             setup_func.name, " ", OpTypeToString(op_precision_type));
  }
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to setup xnnpack kernel. xnn_setup_", setup_func.name, "_nd_",
                           OpTypeToString(op_precision_type), " returned ", status);
  }
  return Status::OK();
}

Status CreateUnaryArithmetickernelBase(struct xnn_operator*& op,
                                       OpComputeType op_precision_type,
                                       size_t channels,
                                       size_t input_stride,
                                       size_t output_stride,
                                       const UnaryOpCreateFunc& create_func) {
  xnn_status status = xnn_status::xnn_status_uninitialized;

  switch (op_precision_type) {
    case OpComputeType::op_compute_type_fp32:
      status = create_func.float_func(channels, input_stride, output_stride, 0, &op);
      break;
    case OpComputeType::op_compute_type_fp16:
      status = create_func.half_func(channels, input_stride, output_stride, 0, &op);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op precision type for: ",
                             create_func.name, " ", OpTypeToString(op_precision_type));
  }
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to create xnnpack kernel. xnn_create_", create_func.name, "_nd_",
                           OpTypeToString(op_precision_type), " returned ", status);
  }
  return Status::OK();
}

Status SetupUnaryArithmetickernelBase(xnn_operator_t op,
                                      OpComputeType op_precision_type,
                                      size_t batch_size,
                                      const void* input,
                                      void* output,
                                      pthreadpool_t threadpool,
                                      const UnaryOpSetupFunc& setup_func) {
  xnn_status status = xnn_status::xnn_status_uninitialized;

  switch (op_precision_type) {
    case OpComputeType::op_compute_type_fp32:
      status = setup_func.float_func(op, batch_size, static_cast<const float*>(input),
                                     static_cast<float*>(output), threadpool);
      break;
    case OpComputeType::op_compute_type_fp16:
      status = setup_func.half_func(op, batch_size, input, output, threadpool);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op precision type for : ",
                             setup_func.name, " ", OpTypeToString(op_precision_type));
  }
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to setup xnnpack kernel. xnn_setup_", setup_func.name, "_nd_",
                           OpTypeToString(op_precision_type), " returned ", status);
  }
  return Status::OK();
}

Status CreateActivationkernelBase(struct xnn_operator*& op,
                                  OpComputeType op_precision_type,
                                  size_t channels,
                                  size_t input_stride,
                                  size_t output_stride,
                                  const ActivationParam& activation_param,
                                  const OpQuantParam& quant_param,
                                  const ActivationOpCreateFunc& create_func) {
  xnn_status status = xnn_status::xnn_status_uninitialized;

  auto dispatch_task_float = [&]() -> xnn_status {
    switch (create_func.float_func.index()) {
      case 0:
        return std::get<0>(create_func.float_func)(channels, input_stride, output_stride,
                                                   activation_param.value[0], activation_param.value[1], 0, &op);
      case 1:
        return std::get<1>(create_func.float_func)(channels, input_stride, output_stride,
                                                   activation_param.value[0], 0, &op);
      case 2:
        return std::get<2>(create_func.float_func)(channels, input_stride, output_stride, 0, &op);
      default:
        return xnn_status::xnn_status_unsupported_parameter;
    }
  };
  auto dispatch_task_half = [&]() ->xnn_status {
    switch (create_func.half_func.index()) {
      case 0:
        return std::get<0>(create_func.half_func)(channels, input_stride, output_stride,
                                                   activation_param.value[0], activation_param.value[1], 0, &op);
      case 1:
        return std::get<1>(create_func.half_func)(channels, input_stride, output_stride,
                                                   activation_param.value[0], 0, &op);
      case 2:
        return std::get<2>(create_func.half_func)(channels, input_stride, output_stride, 0, &op);
      default:
        return xnn_status::xnn_status_uninitialized;
    }
  };
  auto dispatch_task_qu8 = [&]() -> xnn_status {
    switch (create_func.qu8_func.index()) {
      case 0:
        return std::get<0>(create_func.qu8_func)(channels, input_stride, output_stride, activation_param.value[0],
                                                 quant_param[0].second, quant_param[0].first[0],
                                                 quant_param[1].second, quant_param[1].first[0], 0, &op);
      case 1:
        return std::get<1>(create_func.qu8_func)(channels, input_stride, output_stride,
                                                 quant_param[0].second, quant_param[0].first[0],
                                                 quant_param[1].second, quant_param[1].first[0], 0, 255, 0, &op);
      case 2:
        return std::get<2>(create_func.qu8_func)(channels, input_stride, output_stride, 0, 255, 0, &op);
      default:
        return xnn_status::xnn_status_uninitialized;
    }
  };
  auto dispatch_task_qs8 = [&]() -> xnn_status {
    switch (create_func.qs8_func.index()) {
      case 0:
        return std::get<0>(create_func.qs8_func)(channels, input_stride, output_stride,
                                                 activation_param.value[0], quant_param[0].second,
                                                 quant_param[0].first[0], quant_param[1].second,
                                                 quant_param[1].first[0], -126, 127, 0, &op);
      case 1:
        return std::get<1>(create_func.qs8_func)(channels, input_stride, output_stride,
                                                 activation_param.value[0], quant_param[0].second,
                                                 quant_param[0].first[0], quant_param[1].second,
                                                 quant_param[1].first[0], 0, &op);
      case 2:
        return std::get<2>(create_func.qs8_func)(channels, input_stride, output_stride,
                                                 quant_param[0].second, quant_param[0].first[0],
                                                 quant_param[1].second, quant_param[1].first[0], -126, 127, 0, &op);
      case 3:
        return std::get<3>(create_func.qs8_func)(channels, input_stride, output_stride,
                                                 0, 127, 0, &op);
      default:
        return xnn_status::xnn_status_unsupported_parameter;
    }
  };
  const InlinedHashMap<OpComputeType, std::function<xnn_status()>> dispatch_tasks = {
      {OpComputeType::op_compute_type_fp32, dispatch_task_float},
      {OpComputeType::op_compute_type_fp16, dispatch_task_half},
      {OpComputeType::op_compute_type_qu8, dispatch_task_qu8},
      {OpComputeType::op_compute_type_qs8, dispatch_task_qs8},
  };

  if (dispatch_tasks.at(op_precision_type)() != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to create xnnpack kernel. xnn_create_", create_func.name, "_nd_",
                           OpTypeToString(op_precision_type), " returned ", status);
  }
  return Status::OK();
}

Status SetupActivationkernelBase(xnn_operator_t op,
                                      OpComputeType op_precision_type,
                                      size_t batch_size,
                                      const void* input,
                                      void* output,
                                      pthreadpool_t threadpool,
                                      const ActivationOpSetupFunc& setup_func) {
  xnn_status status = xnn_status::xnn_status_uninitialized;

  switch (op_precision_type) {
    case OpComputeType::op_compute_type_fp32:
      status = setup_func.float_func(op, batch_size, static_cast<const float*>(input),
                                     static_cast<float*>(output), threadpool);
      break;
    case OpComputeType::op_compute_type_fp16:
      status = setup_func.half_func(op, batch_size, input, output, threadpool);
      break;
    case OpComputeType::op_compute_type_qu8:
      status = setup_func.qu8_func(op, batch_size, static_cast<const uint8_t*>(input),
                                   static_cast<uint8_t*>(output), threadpool);
      break;
    case OpComputeType::op_compute_type_qs8:
      status = setup_func.qs8_func(op, batch_size, static_cast<const int8_t*>(input),
                                   static_cast<int8_t*>(output), threadpool);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op precision type for : ",
                             setup_func.name, " ", OpTypeToString(op_precision_type));
  }
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to setup xnnpack kernel. xnn_setup_", setup_func.name, "_nd_",
                           OpTypeToString(op_precision_type), " returned ", status);
  }
  return Status::OK();
}

Status Createkernel(struct xnn_operator*& op,
                    const std::optional<std::pair<float, float>>& clip_min_max,
                    ElementWiseOpTypeEnum OPNameType,
                    OpComputeType op_precision_type,
                    OpQuantParam quant_param) {
  if (OPNameType == ElementWiseOpTypeEnum::OP_DIV) {
    ORT_ENFORCE(op_precision_type == OpComputeType::op_compute_type_fp32 ||
                    op_precision_type == OpComputeType::op_compute_type_fp16,
                "Div only support fp32 and fp16");
  }
  return CreateBinaryArithmetickernelBase(op, clip_min_max, op_precision_type,
                                          quant_param, AddSubMulDivCreateFuncMap.at(OPNameType));
}

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
                   pthreadpool_t threadpool) {
  if (OPNameType == ElementWiseOpTypeEnum::OP_DIV) {
    ORT_ENFORCE(op_precision_type == OpComputeType::op_compute_type_fp32 ||
                    op_precision_type == OpComputeType::op_compute_type_fp16,
                "Div only support fp32 and fp16");
  }
  if (OPNameType == ElementWiseOpTypeEnum::OP_MIN || OPNameType == ElementWiseOpTypeEnum::OP_MAX) {
    return SetupMinMaxkernelBase(op, op_precision_type, num_input1_dims, input1_shape, num_input2_dims, input2_shape,
                                 input1, input2, output, threadpool, MinMaxSetupFuncMap.at(OPNameType));
  }
  return SetupBinaryArithmetickernelBase(op, op_precision_type, num_input1_dims, input1_shape, num_input2_dims, input2_shape,
                                         input1, input2, output, threadpool, AddSubMulDivSetupFuncMap.at(OPNameType));
}

Status Createkernel(struct xnn_operator*& op,
                    ElementWiseOpTypeEnum OPNameType,
                    OpComputeType op_precision_type,
                    size_t channels,
                    size_t input_stride,
                    size_t output_stride) {
  return CreateUnaryArithmetickernelBase(op, op_precision_type, channels, input_stride, output_stride,
                                         UnaryCreateFuncMap.at(OPNameType));
}

Status Setupkernel(xnn_operator_t op,
                   ElementWiseOpTypeEnum OPNameType,
                   OpComputeType op_precision_type,
                   size_t batch_size,
                   const void* input,
                   void* output,
                   pthreadpool_t threadpool) {
  if (OPNameType > ElementWiseOpTypeEnum::OP_ACTIVATION_START) {
    return SetupActivationkernelBase(op, op_precision_type, batch_size,
                                     input, output, threadpool, ActivationSetupFuncMap.at(OPNameType));
  }
  return SetupUnaryArithmetickernelBase(op, op_precision_type, batch_size,
                                        input, output, threadpool, UnarySetupFuncMap.at(OPNameType));
}

Status Createkernel(struct xnn_operator*& op,
                    ElementWiseOpTypeEnum OPNameType,
                    OpComputeType op_precision_type) {
  return CreateMinMaxkernelBase(op, op_precision_type, MinMaxCreateFuncMap.at(OPNameType));
}

Status Createkernel(struct xnn_operator*& op,
                    ElementWiseOpTypeEnum OPNameType,
                    OpComputeType op_precision_type,
                    size_t channels,
                    size_t input_stride,
                    size_t output_stride,
                    const ActivationParam& activation_param,
                    OpQuantParam quant_param) {
  return CreateActivationkernelBase(op, op_precision_type, channels, input_stride,
                                    output_stride, activation_param, quant_param,
                                    ActivationCreateFuncMap.at(OPNameType));
}

}  // namespace kernel_create_utils
}  // namespace xnnpack
}  // namespace onnxruntime

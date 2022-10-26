
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/math/activation.h"

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/providers/xnnpack/detail/utils.h"


namespace onnxruntime {
namespace xnnpack {

namespace {
using XnnOpEnum = kernel_utils::ElementWiseOpTypeEnum;
using ActivationParam = kernel_utils::ActivationParam;
const InlinedHashMap<std::string_view, XnnOpEnum> OpTypeMap = {
    {"Clip", XnnOpEnum::OP_CLAMP},
    {"PRelu", XnnOpEnum::OP_PRELU},
    {"LeakyRelu", XnnOpEnum::OP_LEAKY_RELU},
    {"Elu", XnnOpEnum::OP_ELU},
    {"HardSwish", XnnOpEnum::OP_HARD_SWISH},
    {"Sigmoid", XnnOpEnum::OP_SIGMOID},
    {"Tanh", XnnOpEnum::OP_TANH},
};

Status CreateXnnpackKernel(struct xnn_operator*& op,
                           const ActivationParam& activation_param,
                           const OpQuantParam& quant_param,
                           OpComputeType op_precision_type,
                           std::string_view op_name,
                           size_t channels) {
  Status status;
  auto op_name_type = OpTypeMap.at(op_name);
  op = nullptr;
  switch (op_name_type) {
    case XnnOpEnum::OP_CLAMP:
    case XnnOpEnum::OP_LEAKY_RELU:
    case XnnOpEnum::OP_ELU:
    case XnnOpEnum::OP_HARD_SWISH:
    case XnnOpEnum::OP_SIGMOID:
    case XnnOpEnum::OP_TANH:
      status = kernel_utils::Createkernel(op, op_name_type, op_precision_type,
                                          channels, channels, channels, activation_param, quant_param);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op type: ", op_name);
  }
  ORT_RETURN_IF_ERROR(status);
  return Status::OK();
}

bool IsQuantizedActOp(QuantizedOpType) {
  return true;
}
bool IsValidQuantActOp(const NodeUnit& , const GraphViewer& ) {
    return true;
}
}  // namespace

bool ActivationOp::IsOnnxNodeSupported(const NodeUnit& nodeunit, const GraphViewer& graph) {
  bool supported = false;
  auto qtype = GetQuantizedOpType(nodeunit);
  if (IsQuantizedActOp(qtype) && IsValidQuantActOp(nodeunit, graph) == false) {
    return false;
  }
  do {
    const auto& inputs = nodeunit.Inputs();
    const auto& x_arg = inputs[0].node_arg;
    const auto* x_type = x_arg.TypeAsProto();

    if (nodeunit.OpType() == "Tanh") {
      if (x_type == nullptr ||
           (x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8 &&
           x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8)) {
        break;
      }
      supported = true;
      break;
    }
    supported = true;
  } while (0);
  return supported;
}

ActivationOp::ActivationOp(const OpKernelInfo& info) : OpKernel(std::move(info)) {
  const auto& node{Node()};
  op_name_ = node.OpType();
  ORT_ENFORCE(
      OpTypeMap.count(op_name_) > 0,
      "This kernel doesn't take responsible for this op's implementation, OpType:", op_name_);
  op_name_type_ = OpTypeMap.at(op_name_);
  ORT_THROW_IF_ERROR(Init(info));
  const auto& input_defs = node.InputDefs();
  const NodeArg& X = *input_defs[0];
  auto input_dtype = X.TypeAsProto()->tensor_type().elem_type();
  if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    op_precision_type_ = OpComputeType::op_compute_type_fp32;
  } else if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8 ||
             input_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    quant_param_ = ParseQuantParamForOp(info, input_dtype, 1);
    op_precision_type_ = input_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8
                             ? OpComputeType::op_compute_type_qs8
                             : OpComputeType::op_compute_type_qu8;
  } else {
    auto stype = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*X.TypeAsProto()));
    ORT_THROW("unsupported elementwise op ", op_name_,
              " in XnnpackEP, we have FLOAT|UINT8|INT8, but got ", stype);
  }
  size_t channels = X.Shape()->dim(X.Shape()->dim_size() - 1).dim_value();
  struct xnn_operator* p = nullptr;
  ORT_THROW_IF_ERROR(CreateXnnpackKernel(p, activation_param_, quant_param_, op_precision_type_, op_name_, channels));
  op0_.reset(p);
}

Status ActivationOp::Init(const OpKernelInfo& info) {
  switch (op_name_type_) {
    case XnnOpEnum::OP_CLAMP: {
      constexpr auto min_val = std::numeric_limits<float>::lowest();
      constexpr auto max_val = std::numeric_limits<float>::max();
      info.GetAttrOrDefault("min", &activation_param_.clip.min, min_val);
      info.GetAttrOrDefault("max", &activation_param_.clip.max, max_val);
      ORT_ENFORCE(activation_param_.clip.min <= activation_param_.clip.max);
    } break;
    case XnnOpEnum::OP_PRELU:
      ORT_THROW("PReLU is not supported yet.");
      break;
    case XnnOpEnum::OP_LEAKY_RELU: {
      constexpr auto alpha_default = 0.01F;
      info.GetAttrOrDefault("alpha", &activation_param_.leaky_relu.alpha, alpha_default);
    } break;
    case XnnOpEnum::OP_ELU: {
      constexpr auto alpha_default = 1.0F;
      info.GetAttrOrDefault("alpha", &activation_param_.elu.alpha, alpha_default);
    } break;
    case XnnOpEnum::OP_HARD_SWISH:
      // alpha = 1/6 and beta = 0.5
      break;
    case XnnOpEnum::OP_SIGMOID:
      break;
    case XnnOpEnum::OP_TANH:
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op type: ", op_name_);
  }
  return Status::OK();
}

Status ActivationOp::Compute(OpKernelContext* context) const {
  const Tensor& X1 = *context->Input<Tensor>(0);
  // const Tensor* X2_ptr = context->Input<Tensor>(1); // only p-relu has the second input
  auto X1_shape = X1.Shape();
  size_t batch_size = X1_shape.NumDimensions() == 1 ? 1 : X1_shape[0];

  Tensor* Y = context->Output(0, X1.Shape());
  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }
  xnn_status status = xnn_status_invalid_state;
  pthreadpool_t threadpool = nullptr;

  switch (op_name_type_) {
    case XnnOpEnum::OP_CLAMP:
    case XnnOpEnum::OP_LEAKY_RELU:
    case XnnOpEnum::OP_ELU:
    case XnnOpEnum::OP_HARD_SWISH:
    case XnnOpEnum::OP_SIGMOID:
    case XnnOpEnum::OP_TANH:
      ORT_RETURN_IF_ERROR(kernel_utils::Setupkernel(
          op0_.get(), op_name_type_, op_precision_type_, batch_size, X1.DataRaw(),
          Y->MutableDataRaw(), threadpool));
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op type: ", op_name_);
  }

  status = xnn_run_operator(op0_.get(), threadpool);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

// Register the kernel
#define REGISTER_ACT_OP_FLOAT(OP, ver)                                     \
  ONNX_OPERATOR_KERNEL_EX(OP, kOnnxDomain, ver, kXnnpackExecutionProvider, \
                          KernelDefBuilder().TypeConstraint(               \
                              "T", DataTypeImpl::GetTensorType<float>()),  \
                          ActivationOp);

// dynamic schema, take uint8_t|int8_t
#define REGISTER_MATH_OP_U8S8(OP, ver)                                                \
  ONNX_OPERATOR_KERNEL_EX(OP, kDynamicDomainByCreate, ver, kXnnpackExecutionProvider, \
                          KernelDefBuilder(),                                         \
                          ActivationOp);

REGISTER_ACT_OP_FLOAT(Elu, 6);
REGISTER_ACT_OP_FLOAT(LeakyRelu, 6);
REGISTER_ACT_OP_FLOAT(HardSwish, 14);
REGISTER_ACT_OP_FLOAT(Sigmoid, 6);
REGISTER_ACT_OP_FLOAT(Tanh, 6);

}  // namespace xnnpack
}  // namespace onnxruntime

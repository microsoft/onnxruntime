// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/math/math_elementwise_ops.h"
#include <algorithm>
#include <cstdint>
#include <string_view>
#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/common/status.h"
#include "core/graph/constants.h"
#include "core/providers/xnnpack/detail/utils.h"

namespace onnxruntime {
namespace xnnpack {

namespace {
using XnnOpEnum = kernel_utils::ElementWiseOpTypeEnum;
const InlinedHashMap<std::string_view, XnnOpEnum> OpTypeMap = {
    {"Add", XnnOpEnum::OP_ADD},
    {"Mul", XnnOpEnum::OP_MUL},
    {"Div", XnnOpEnum::OP_DIV},
    {"Sub", XnnOpEnum::OP_SUB},
    {"QLinearAdd", XnnOpEnum::OP_ADD},
    {"QLinearMul", XnnOpEnum::OP_MUL},
    {"QLinearSub", XnnOpEnum::OP_SUB},

    {"Abs", XnnOpEnum::OP_ABS},
    {"Neg", XnnOpEnum::OP_NEG},
    {"Floor", XnnOpEnum::OP_FLOOR},
    {"Ceil", XnnOpEnum::OP_CEIL},
    {"Sqrt", XnnOpEnum::OP_SQRT},
    {"Round", XnnOpEnum::OP_Round},
    {"Max", XnnOpEnum::OP_MAX},
    {"Min", XnnOpEnum::OP_MIN},
};

Status CreateXnnpackKernel(struct xnn_operator*& op,
                           const std::optional<std::pair<float, float>>& clip_min_max,
                           const OpQuantParam& quant_param,
                           OpComputeType op_precision_type,
                           std::string_view op_name,
                           size_t channels) {
  Status status;
  auto op_name_type = OpTypeMap.at(op_name);
  op = nullptr;
  switch (op_name_type) {
    case XnnOpEnum::OP_ADD:
    case XnnOpEnum::OP_SUB:
    case XnnOpEnum::OP_MUL:
    case XnnOpEnum::OP_DIV:
      status = kernel_utils::Createkernel(
          op, clip_min_max, op_name_type, op_precision_type, quant_param);
      break;
    case XnnOpEnum::OP_ABS:
    case XnnOpEnum::OP_Round:
    case XnnOpEnum::OP_CEIL:
    case XnnOpEnum::OP_FLOOR:
    case XnnOpEnum::OP_NEG:
    case XnnOpEnum::OP_SQRT:
    case XnnOpEnum::OP_TRUNCATE:
    case XnnOpEnum::OP_SQUARE:
      status = kernel_utils::Createkernel(
          op, op_name_type, op_precision_type, channels, channels, channels);
      break;
    case XnnOpEnum::OP_MAX:
    case XnnOpEnum::OP_MIN:
      status = kernel_utils::Createkernel(
          op, op_name_type, op_precision_type);
      break;
    default:
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported op type: ", op_name);
  }
  ORT_RETURN_IF_ERROR(status);
  return Status::OK();
}

bool IsQuantizedBinaryMathOp(QuantizedOpType) {
  // div has no quantized op

  // unary op has only float  type
  return true;
}

bool IsValidQuantBinaryMathOp(const NodeUnit&, const GraphViewer&) {
  return true;
}

}  // namespace


bool ElementWiseOp::IsOnnxNodeSupported(const NodeUnit& nodeunit, const GraphViewer& graph) {
  bool supported = false;
  auto qtype = GetQuantizedOpType(nodeunit);
  if (IsQuantizedBinaryMathOp(qtype) && IsValidQuantBinaryMathOp(nodeunit, graph) == false) {
    return false;
  }
  return !supported;
}

ElementWiseOp::ElementWiseOp(const OpKernelInfo& info) : OpKernel(info) {
  // get values from any fusion with an activation
  std::string activation;
  if (info.GetAttr<std::string>("activation", &activation).IsOK()) {
    std::vector<float> activation_params;

    // min/max could be from Clip or Relu
    if (info.GetAttrs<float>("activation_params", activation_params).IsOK()) {
      if (activation_params.size() == 2) {
        clip_min_max_ = {activation_params[0], activation_params[1]};
      }
    }
  }
  const auto& node{Node()};
  op_name_ = node.OpType();
  ORT_ENFORCE(
      OpTypeMap.count(op_name_) > 0,
      "This kernel doesn't take responsible for this op's implementation, OpType:", op_name_);
  op_name_type_ = OpTypeMap.at(op_name_);
  bool is_binary_op = op_name_type_ > kernel_utils::ElementWiseOpTypeEnum::OP_BINARY_START;
  const auto& input_defs = node.InputDefs();
  const NodeArg& X = *input_defs[0];

  auto input_dtype = X.TypeAsProto()->tensor_type().elem_type();
  if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    op_precision_type_ = OpComputeType::op_compute_type_fp32;
  } else if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8 ||
             input_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    quant_param_ = ParseQuantParamForOp(info, input_dtype, is_binary_op ? 2 : 1);
    op_precision_type_ = input_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8
                             ? OpComputeType::op_compute_type_qs8
                             : OpComputeType::op_compute_type_qu8;
  } else {
    auto stype = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*X.TypeAsProto()));
    ORT_THROW("unsupported elementwise op ", op_name_,
              " in XnnpackEP, we have FLOAT|UINT8|INT8, but got ", stype);
  }
  size_t channels = X.Shape()->dim(X.Shape()->dim_size()-1).dim_value();
  struct xnn_operator* p = nullptr;
  ORT_THROW_IF_ERROR(CreateXnnpackKernel(p, clip_min_max_, quant_param_, op_precision_type_, op_name_, channels));
  op0_.reset(p);
}

Status ElementWiseOp::Compute(OpKernelContext* context) const {
  bool is_binary_op = op_name_type_ > kernel_utils::ElementWiseOpTypeEnum::OP_BINARY_START;
  const Tensor& X1 = *context->Input<Tensor>(0);
  const Tensor* X2_ptr = is_binary_op ? context->Input<Tensor>(1) : nullptr;
  auto X1_shape = X1.Shape();
  auto X2_shape = X2_ptr ? X2_ptr->Shape() : TensorShape{};

  auto batch_size = X1_shape[0];

  Tensor* Y = context->Output(0, X1.Shape());
  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }
  xnn_status status = xnn_status_invalid_state;
  size_t num_input1_dims = X1_shape.NumDimensions();
  InlinedVector<size_t> input1_shape(num_input1_dims);
  std::transform(X1_shape.GetDims().begin(), X1_shape.GetDims().end(), input1_shape.begin(),
                 [](int64_t dim) { return gsl::narrow_cast<size_t>(dim); });
  size_t num_input2_dims = X2_shape.NumDimensions();
  InlinedVector<size_t> input2_shape(num_input2_dims);
  std::transform(X2_shape.GetDims().begin(), X2_shape.GetDims().end(), input2_shape.begin(),
                 [](int64_t dim) { return gsl::narrow_cast<size_t>(dim); });

  pthreadpool_t threadpool = nullptr;

  switch (op_name_type_) {
    case XnnOpEnum::OP_ADD:
    case XnnOpEnum::OP_SUB:
    case XnnOpEnum::OP_MUL:
    case XnnOpEnum::OP_DIV:
    case XnnOpEnum::OP_MAX:
    case XnnOpEnum::OP_MIN:
      ORT_RETURN_IF_ERROR(kernel_utils::Setupkernel(
          op0_.get(), op_name_type_, op_precision_type_, num_input1_dims,
          input1_shape.data(), num_input2_dims, input2_shape.data(), X1.DataRaw(),
          X2_ptr->DataRaw(), Y->MutableDataRaw(), threadpool));
      break;
    case XnnOpEnum::OP_ABS:
    case XnnOpEnum::OP_Round:
    case XnnOpEnum::OP_CEIL:
    case XnnOpEnum::OP_FLOOR:
    case XnnOpEnum::OP_NEG:
    case XnnOpEnum::OP_SQRT:
    case XnnOpEnum::OP_TRUNCATE:
    case XnnOpEnum::OP_SQUARE:
      ORT_RETURN_IF_ERROR(kernel_utils::Setupkernel(
          op0_.get(), op_name_type_, op_precision_type_, batch_size,
          X1.DataRaw(), Y->MutableDataRaw(), threadpool));
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
#define REGISTER_MATH_OP_FLOAT(OP, ver)                             \
  ONNX_OPERATOR_KERNEL_EX(OP, kOnnxDomain, ver, kXnnpackExecutionProvider, \
                          KernelDefBuilder().TypeConstraint(               \
                              "T", DataTypeImpl::GetTensorType<float>()),  \
                          ElementWiseOp);

// dynamic schema, take uint8_t|int8_t
#define REGISTER_MATH_OP_U8S8(OP, ver)                                         \
  ONNX_OPERATOR_KERNEL_EX(OP, kDynamicDomainByCreate, ver, kXnnpackExecutionProvider, \
                          KernelDefBuilder(),                                         \
                              ElementWiseOp);

REGISTER_MATH_OP_FLOAT(Add, 7);
REGISTER_MATH_OP_FLOAT(Sub, 7);
REGISTER_MATH_OP_FLOAT(Mul, 7);
REGISTER_MATH_OP_FLOAT(Div, 7);

REGISTER_MATH_OP_FLOAT(Abs, 6);
REGISTER_MATH_OP_FLOAT(Round, 11);
REGISTER_MATH_OP_FLOAT(Ceil, 6);

REGISTER_MATH_OP_FLOAT(Floor, 6);
REGISTER_MATH_OP_FLOAT(Neg, 6);
REGISTER_MATH_OP_FLOAT(Sqrt, 6);
REGISTER_MATH_OP_FLOAT(Min, 8);
REGISTER_MATH_OP_FLOAT(Max, 8);


REGISTER_MATH_OP_U8S8(QLinearAdd, 1);
REGISTER_MATH_OP_U8S8(QLinearSub, 1);
REGISTER_MATH_OP_U8S8(QLinearMul, 1);
}  // namespace xnnpack
}  // namespace onnxruntime

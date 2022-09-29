// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/nn/concat_and_split.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <utility>

#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/common/status.h"
#include "core/framework/op_kernel.h"
#include "core/framework/tensorprotoutils.h"
#include "core/optimizer/initializer.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/graph/graph.h"
#include "xnnpack.h"

namespace onnxruntime {
namespace xnnpack {

namespace {
Status CreateXnnpackSubgraph(const Node& node, XnnpackSubgraph& subgraph_uptr, size_t axis,
                             InlinedVector<uint32_t>& external_tensors, const OpQuantParam& quant_param,
                             OpComputeType op_type, bool is_concat = true) {
  xnn_subgraph_t subgraph = nullptr;

  size_t input_count = is_concat ? node.InputDefs().size() : 1;
  xnn_status status = xnn_create_subgraph(/*external_value_ids=*/input_count +
                                              node.OutputDefs().size(),
                                          /*flags=*/0, &subgraph);
  subgraph_uptr.reset(subgraph);

  if (xnn_status_success != status) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "create subgraph failed with ", status);
  }

  external_tensors.resize(input_count + node.OutputDefs().size(), -1);

  auto get_shape_in_size_t = [](const ONNX_NAMESPACE::TensorShapeProto& shape_proto) {
    auto input_shape = utils::GetTensorShapeFromTensorShapeProto(shape_proto);
    InlinedVector<size_t> dim(input_shape.NumDimensions());
    std::transform(input_shape.GetDims().begin(), input_shape.GetDims().end(), dim.begin(),
                   [](int64_t i) { return i > 0 ? static_cast<size_t>(i) : 1; });
    return dim;
  };
  if (op_type == OpComputeType::op_compute_type_fp32) {
    for (size_t ind = 0; ind < input_count; ++ind) {
      auto dim = get_shape_in_size_t(*node.InputDefs()[ind]->Shape());

      status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, dim.size(), dim.data(), nullptr, ind,
                                       /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &external_tensors[ind]);
      if (xnn_status_success != status) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "define quantized tensor value failed with ", status);
      }
    }
    for (size_t ind = 0; ind < node.OutputDefs().size(); ++ind) {
      size_t output_value_ind = ind + input_count;
      auto dim = get_shape_in_size_t(*node.OutputDefs()[ind]->Shape());

      status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, dim.size(), dim.data(), nullptr,
                                       output_value_ind, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT,
                                       &external_tensors[output_value_ind]);
    }
  } else if (op_type == OpComputeType::op_compute_type_qs8 || op_type == OpComputeType::op_compute_type_qu8) {
    auto xnn_dtype = op_type == OpComputeType::op_compute_type_qs8 ? xnn_datatype_qint8 : xnn_datatype_quint8;
    for (size_t ind = 0; ind < input_count; ++ind) {
      auto dim = get_shape_in_size_t(*node.InputDefs()[ind]->Shape());
      status = xnn_define_quantized_tensor_value(
          subgraph, xnn_dtype, quant_param[0].second, quant_param[0].first[0],
          dim.size(), dim.data(), nullptr, ind,
          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_INPUT, &external_tensors[ind]);
      if (xnn_status_success != status) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "define quantized tensor value failed with ", status);
      }
    }
    for (size_t ind = 0; ind < node.OutputDefs().size(); ++ind) {
      size_t output_value_ind = ind + input_count;

      auto dim = get_shape_in_size_t(*node.OutputDefs()[ind]->Shape());
      status = xnn_define_quantized_tensor_value(
          subgraph, xnn_dtype, quant_param[0].second, quant_param[0].first[0],
          dim.size(), dim.data(), nullptr, output_value_ind,
          /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &external_tensors[output_value_ind]);
      if (xnn_status_success != status) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "define quantized tensor value failed with ", status);
      }
    }
  }
  size_t arg_count = is_concat ? input_count : node.OutputDefs().size();
  if (arg_count == 2) {
    auto define_func = is_concat ? xnn_define_concatenate2 : xnn_define_even_split2;
    status = define_func(
        subgraph, axis, external_tensors[0], external_tensors[1], external_tensors[2], /*flags=*/0);
  } else if (arg_count == 3) {
    auto define_func = is_concat ? xnn_define_concatenate3 : xnn_define_even_split3;
    status = define_func(
        subgraph, axis, external_tensors[0], external_tensors[1], external_tensors[2],
        external_tensors[3], /*flags=*/0);
  } else if (arg_count == 4) {
    auto define_func = is_concat ? xnn_define_concatenate4 : xnn_define_even_split4;
    status = define_func(
        subgraph, axis, external_tensors[0], external_tensors[1], external_tensors[2],
        external_tensors[3], external_tensors[4], /*flags=*/0);
  }

  if (xnn_status_success != status) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           is_concat ? "xnn_define_concatenate_" : "xnn_define_split_", arg_count,
                           " failed with ", status);
  }
  return Status::OK();
}

Status CreateXnnRuntime(XnnpackSubgraph& subgraph_uptr, XnnpackRuntime& runtime_ptr,
                        XnnpackWorkspace& workspace_uptr, pthreadpool_t tpool, OpComputeType op_type) {
  xnn_workspace_t workspace = nullptr;
  xnn_status xstatus = xnn_create_workspace(&workspace);
  ORT_ENFORCE(xstatus == xnn_status_success, "xnn_create_workspace for concat failed,",
              OpTypeToString(op_type), ". Status:", xstatus);
  workspace_uptr.reset(workspace);

  xnn_runtime_t runtime = nullptr;
  xstatus = xnn_create_runtime_v4(subgraph_uptr.get(), nullptr, workspace, tpool,
                                  /*flags=*/0, &runtime);
  ORT_ENFORCE(xstatus == xnn_status_success, "xnn_create_runtime_v4 for Split failed,",
              OpTypeToString(op_type), ". Status:", xstatus);
  runtime_ptr.reset(runtime);
  return Status::OK();
}

Status XnnSetupAndRun(const InlinedVector<xnn_external_value>& external, xnn_runtime_t runtime,
                      OpComputeType op_type) {
  xnn_status xstatus = xnn_setup_runtime(runtime, external.size(), external.data());

  if (xstatus != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_runtime ",
                           OpTypeToString(op_type), " returned ", xstatus);
  }
  xstatus = xnn_invoke_runtime(runtime);
  if (xstatus != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", xstatus);
  }
  return Status::OK();
}
}  // namespace

bool Concat::IsOnnxNodeSupported(const NodeUnit& node_unit,
                                 const GraphViewer&) {
  bool support = false;
  do {
    int dtype = 0;
    auto input_defs = node_unit.Inputs();
    if (input_defs.size() > 3) {
      break;
    }
    for (const auto& input_def : input_defs) {
      const auto& x_arg = input_def.node_arg;
      const auto* x_shape = x_arg.Shape();
      if (!x_shape) {
        return false;
      }
      // how to handle batch size
      auto input_shape = utils::GetTensorShapeFromTensorShapeProto(*x_shape).GetDims();
      for (size_t i = 1; i < input_shape.size(); i++) {
        if (input_shape[i] < 1) {
          return false;
        }
      }

      const auto* x_type = x_arg.TypeAsProto();
      if (x_type == nullptr ||
          (x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
           x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8 &&
           x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8)) {
        return false;
      }
      if (dtype == 0) {
        dtype = x_type->tensor_type().elem_type();
      } else if (x_type->tensor_type().elem_type() != dtype) {
        return false;
      }
    }
    support = true;
  } while (false);

  return support;
}

Concat::Concat(const OpKernelInfo& info) : XnnpackKernel{info}, ConcatBase(info) {
  const auto& node = info.node();
  auto input_defs = node.InputDefs();
  int x_dtype = 0;
  OpQuantParam quant_param;
  ORT_ENFORCE(GetType(*input_defs[0], x_dtype));
  if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    op_type_ = OpComputeType::op_compute_type_fp32;
  } else if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
             x_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    op_type_ = OpComputeType::op_compute_type_qu8;  // use to represent 8bit quantized data
    quant_param = ParseQuantParamForOp(info, x_dtype, input_defs.size());
  } else {
    auto stype = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*input_defs[0]->TypeAsProto()));
    ORT_THROW("unsupported dtype in Concat, we have FLOAT|UINT8, but got ", stype);
  }
  axis_ = HandleNegativeAxis(axis_, input_defs[0]->Shape()->dim_size());
  ORT_THROW_IF_ERROR(CreateXnnpackSubgraph(node, subgraph_, axis_, external_tensors_, quant_param, op_type_));
  ORT_THROW_IF_ERROR(CreateXnnRuntime(subgraph_, runtime_, workspace_, GetThreadPool(), op_type_));
}

// compute method of Concat
Status Concat::Compute(OpKernelContext* ctx) const {
  // Number of input tensors to concatenate
  int input_count = Node().InputArgCount().front();

  // Hold pointers to the input tensors to be used in the PrepareForCompute() step
  InlinedTensorsVector input_tensors;
  input_tensors.reserve(input_count);
  for (int i = 0; i < input_count; ++i) {
    input_tensors.push_back(ctx->Input<Tensor>(i));
  }

  // Validate inputs and prepare some metadata used during actual compute
  Prepare p;
  auto status = PrepareForCompute(ctx, input_tensors, p);
  if (!status.IsOK())
    return status;

  // Return at this point if output tensor is going to be empty
  if (p.output_num_elements == 0) {
    return Status::OK();
  }

  Tensor& Y = *p.output_tensor;

  InlinedVector<xnn_external_value> external(external_tensors_.size());

  for (int i = 0; i < input_count; ++i) {
    external[i] = {external_tensors_[i], const_cast<void*>(input_tensors[i]->DataRaw())};
  }
  external[input_count] = {external_tensors_[input_count], Y.MutableDataRaw()};
  ORT_RETURN_IF_ERROR(XnnSetupAndRun(external, runtime_.get(), op_type_));

  return Status::OK();
}

bool Split::IsOnnxNodeSupported(const NodeUnit& node_unit,
                                const GraphViewer&) {
  bool support = false;
  do {
    if (node_unit.Outputs().size() > 3) {
      break;
    }
    auto input_def = node_unit.Inputs()[0];
    const auto& x_arg = input_def.node_arg;
    for (const auto& output_def : node_unit.Outputs()) {
      const auto* o_shape = output_def.node_arg.Shape();
      if (!o_shape) {
        return false;
      }
      auto input_shape = utils::GetTensorShapeFromTensorShapeProto(*o_shape).GetDims();
      for (size_t i = 1; i < input_shape.size(); i++) {
        if (input_shape[i] < 1) {
          return false;
        }
      }
    }
    const auto* x_type = x_arg.TypeAsProto();
    if (x_type == nullptr ||
        (x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
         x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8 &&
         x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8)) {
      break;
    }
    support = true;
  } while (false);
  return support;
}

Split::Split(const OpKernelInfo& info) : XnnpackKernel{info}, SplitBase(info) {
  const auto& node = info.node();
  auto input_defs = node.InputDefs();
  int x_dtype = 0;
  OpQuantParam quant_param;
  ORT_ENFORCE(GetType(*input_defs[0], x_dtype));
  if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    op_type_ = OpComputeType::op_compute_type_fp32;
  } else if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
             x_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    op_type_ = OpComputeType::op_compute_type_qu8;  // use to represent 8bit quantized data
    quant_param = ParseQuantParamForOp(info, x_dtype, 1);
  } else {
    auto stype = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*input_defs[0]->TypeAsProto()));
    ORT_THROW("unsupported dtype in Split, we have FLOAT|UINT8, but got ", stype);
  }

  axis_ = HandleNegativeAxis(axis_, input_defs[0]->Shape()->dim_size());
  ORT_THROW_IF_ERROR(CreateXnnpackSubgraph(node, subgraph_, axis_, external_tensors_, quant_param, op_type_, false));
  ORT_THROW_IF_ERROR(CreateXnnRuntime(subgraph_, runtime_, workspace_, GetThreadPool(), op_type_));
}

// how to handle batchsize
Status Split::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);

  auto& input_shape = X.Shape();
  auto num_outputs = context->OutputCount();
  int64_t axis = axis_;
  int before_dims = 0;
  int after_dims_including_split_axis = 0;
  int after_dims_excluding_split = 0;
  std::vector<int64_t> split_sizes;

  const Tensor* split_tensor = context->Input<Tensor>(1);
  if (split_tensor != nullptr) {
    // override the attribute value with the input value for split
    ORT_ENFORCE(split_tensor->Shape().NumDimensions() == 1, "An split tensor must be a vector tensor.");
    auto nDims = static_cast<size_t>(split_tensor->Shape()[0]);
    const auto* data = split_tensor->Data<int64_t>();
    split_sizes.assign(data, data + nDims);
  } else {
    split_sizes.assign(split_sizes_.begin(), split_sizes_.end());
  }
  ORT_RETURN_IF_ERROR(PrepareForCompute(input_shape,
                                        num_outputs,
                                        axis,
                                        before_dims,
                                        after_dims_including_split_axis,
                                        after_dims_excluding_split,
                                        split_sizes));

  // copy dimensions so we can update the selected axis in place
  InlinedVector<xnn_external_value> external(external_tensors_.size());
  external[0] = {external_tensors_[0], const_cast<void*>(X.DataRaw())};
  auto output_dimensions = input_shape.AsShapeVector();

  for (int i = 0; i < num_outputs; ++i) {
    // update size of dimension for axis we're splitting on
    auto split_size = gsl::narrow<int>(split_sizes[i]);
    output_dimensions[axis] = split_size;
    auto* Y = context->Output(i, TensorShape{output_dimensions});
    external[1 + i] = {external_tensors_[1 + i], (Y->MutableDataRaw())};
  }
  ORT_RETURN_IF_ERROR(XnnSetupAndRun(external, runtime_.get(), op_type_));

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat, kOnnxDomain, 4, 10, kXnnpackExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<uint8_t>(),
                                            DataTypeImpl::GetTensorType<int8_t>()}),
    Concat);

// Opset 11 starts to support Neg Axis.
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Concat, kOnnxDomain, 11, 12, kXnnpackExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<uint8_t>(),
                                            DataTypeImpl::GetTensorType<int8_t>()}),
    Concat);

// Opset 13 .
ONNX_OPERATOR_KERNEL_EX(
    Concat, kOnnxDomain, 13, kXnnpackExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Concat);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Split, kOnnxDomain, 2, 10, kXnnpackExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<uint8_t>(),
                                            DataTypeImpl::GetTensorType<int8_t>()}),
    Split);
// Opset 11 starts to support Neg Axis.
ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    Split, kOnnxDomain, 11, 12, kXnnpackExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                            DataTypeImpl::GetTensorType<uint8_t>(),
                                            DataTypeImpl::GetTensorType<int8_t>()}),
    Split);

// Opset 13 .
ONNX_OPERATOR_KERNEL_EX(
    Split, kOnnxDomain, 13, kXnnpackExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::AllTensorTypes()),
    Split);
}  // namespace xnnpack
}  // namespace onnxruntime

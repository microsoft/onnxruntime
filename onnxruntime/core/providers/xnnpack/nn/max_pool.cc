// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "max_pool.h"

#include "core/graph/graph.h"
#include "core/providers/utils.h"
#include "core/framework/tensorprotoutils.h"

// to sanity check output shape
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace xnnpack {
namespace {
bool IsQuantizedMaxPool(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearMaxPool) ||
         (quant_op_type == QuantizedOpType::QDQMaxPool);
}

bool IsValidQuantMaxPool(const NodeUnit& node_unit, const GraphViewer& graph) {
  TensorQuantType x_input_type = GetTensorQuantType(node_unit, 0, false, graph);
  TensorQuantType output_type = GetTensorQuantType(node_unit, 0, true, graph);
  if (x_input_type != output_type ||
      (x_input_type != TensorTypeUint8 &&
       x_input_type != TensorTypeInt8)) {
    return false;
  }
  return true;
}
}  // namespace

// MaxPool doesn't have any quantization params
bool MaxPool::IsOnnxNodeSupported(const NodeUnit& node_unit,
                                  const GraphViewer& graph) {
  bool supported = false;
  auto qtype = GetQuantizedOpType(node_unit);
  if (IsQuantizedMaxPool(qtype) && IsValidQuantMaxPool(node_unit, graph) == false) {
    return false;
  }
  const onnxruntime::Node& node = node_unit.GetNode();
  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    if (node_unit.SinceVersion() < 8) {
      break;
    }

    // MaxPool has 1 input.
    auto input_defs = node.InputDefs();
    const auto& x_arg = *input_defs[0];

    // we only support float and u8/s8 currently
    const auto* x_type = x_arg.TypeAsProto();
    // input of maxpool could be fp16/fp32/fp64,i8/u8 according to ONNX
    if (x_type == nullptr ||
        (x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
         x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8 &&
         x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8)) {
      break;
    }

    // we only support 2D (4 dims with batch and channel)
    const auto* x_shape = x_arg.Shape();
    if (!x_shape || x_shape->dim_size() != 4) {
      break;
    }

    // require C, H, W to be known so we can construct the xnnpack kernel prior to Compute
    if (!x_shape->dim(1).has_dim_value() ||
        !x_shape->dim(2).has_dim_value() ||
        !x_shape->dim(3).has_dim_value()) {
      break;
    }

    // we don't support creating the optional 'I' output
    const auto& output_defs = node.OutputDefs();
    if (output_defs.size() == 2 && output_defs[1]->Exists()) {
      break;
    }

    ProtoHelperNodeContext nc(node);
    OpNodeProtoHelper info(&nc);
    PoolAttributes pool_attrs(info, "MaxPool", node.SinceVersion());

    // xnnpack doesn't appear to support using 'ceil' to calculate the output shape
    // https://github.com/google/XNNPACK/blob/3caa8b9de973839afa1e2a1462ff356e6927a66b/src/operators/max-pooling-nhwc.c#L256
    // calls compute_output_dimension but there's no ability to specify rounding that value up.
    if (pool_attrs.ceil_mode != 0) {
      break;
    }

    if (!IsPaddingTypeSupported(pool_attrs.auto_pad)) {
      break;
    }

    if ((pool_attrs.kernel_shape.size() != 2) ||
        (pool_attrs.kernel_shape[0] == 1 && pool_attrs.kernel_shape[1] == 1)) {
      // XNNPack doesn't support 1x1 maxpool.
      break;
    }

    supported = true;
  } while (false);

  return supported;
}

MaxPool::MaxPool(const OpKernelInfo& info)
    : XnnpackKernel(info),
      pool_attrs_{info, "MaxPool", info.node().SinceVersion()} {
  uint32_t input_padding_top = gsl::narrow<uint32_t>(pool_attrs_.pads[0]);
  uint32_t input_padding_left = gsl::narrow<uint32_t>(pool_attrs_.pads[1]);
  uint32_t input_padding_bottom = gsl::narrow<uint32_t>(pool_attrs_.pads[2]);
  uint32_t input_padding_right = gsl::narrow<uint32_t>(pool_attrs_.pads[3]);

  uint32_t pooling_height = gsl::narrow<uint32_t>(pool_attrs_.kernel_shape[0]);
  uint32_t pooling_width = gsl::narrow<uint32_t>(pool_attrs_.kernel_shape[1]);
  uint32_t stride_height = gsl::narrow<uint32_t>(pool_attrs_.strides[0]);
  uint32_t stride_width = gsl::narrow<uint32_t>(pool_attrs_.strides[1]);
  uint32_t dilation_height = gsl::narrow<uint32_t>(pool_attrs_.dilations[0]);
  uint32_t dilation_width = gsl::narrow<uint32_t>(pool_attrs_.dilations[1]);

  // get values from any fusion with an activation
  if (std::string activation; info.GetAttr<std::string>("activation", &activation).IsOK()) {
    if (activation == "Clip" || activation == "Relu") {
      std::vector<float> activation_params;

      // min/max could be from Clip or Relu
      if (info.GetAttrs<float>("activation_params", activation_params).IsOK()) {
        if (activation_params.size() == 2) {
          clip_min_max_ = {activation_params[0], activation_params[1]};
        }
      }
    }
  }

  uint32_t flags = 0;
  if (pool_attrs_.auto_pad == AutoPadType::SAME_UPPER) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }

  // input is NHWC and we only support input with 4 dims. we checked C, H, W were all known in the op support checker
  const auto& X_arg = *Node().InputDefs()[0];
  auto X_shape = utils::GetTensorShapeFromTensorShapeProto(*X_arg.Shape());

  int64_t H = X_shape[1];
  int64_t W = X_shape[2];
  int64_t C = X_shape[3];

  // create NCHW shape to calculate most of the output shape. 'N' is set in Compute.
  TensorShapeVector input_shape{1, C, H, W};
  auto pads = pool_attrs_.pads;
  auto nchw_output_dims = pool_attrs_.SetOutputSize(input_shape, C, &pads);
  output_dims_ = {-1, nchw_output_dims[2], nchw_output_dims[3], nchw_output_dims[1]};

  // TEMPORARY sanity check. If C, H and W are known, the output shape should have been able to be inferred, with the
  // exception of the batch size. Can be removed once we've run more models using xnnpack MaxPool.
  auto inferred_output_shape = utils::GetTensorShapeFromTensorShapeProto(*Node().OutputDefs()[0]->Shape());
  ORT_ENFORCE(inferred_output_shape[1] == output_dims_[1] &&
                  inferred_output_shape[2] == output_dims_[2] &&
                  inferred_output_shape[3] == output_dims_[3],
              "Shape mismatch between inferred value and calculated value.");
  auto input_dtype = X_arg.TypeAsProto()->tensor_type().elem_type();
  xnn_status status = xnn_status_invalid_state;
  struct xnn_operator* p = nullptr;
  float foutput_min = clip_min_max_ ? clip_min_max_->first : -INFINITY;
  float foutput_max = clip_min_max_ ? clip_min_max_->second : INFINITY;
  if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    maxpool_type_ = OpComputeType::op_compute_type_fp32;
    status = xnn_create_max_pooling2d_nhwc_f32(input_padding_top, input_padding_right,
                                               input_padding_bottom, input_padding_left,
                                               pooling_height, pooling_width,
                                               stride_height, stride_width,
                                               dilation_height, dilation_width,
                                               C, C, C,  // channels, input_pixel_stride, output_pixel_stride
                                               foutput_min, foutput_max, flags, &p);
  } else if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    maxpool_type_ = OpComputeType::op_compute_type_qu8;
    const uint8_t output_min = 0;
    const uint8_t output_max = 255;
    status = xnn_create_max_pooling2d_nhwc_u8(input_padding_top, input_padding_right,
                                              input_padding_bottom, input_padding_left,
                                              pooling_height, pooling_width,
                                              stride_height, stride_width,
                                              dilation_height, dilation_width,
                                              C, C, C,  // channels, input_pixel_stride, output_pixel_stride
                                              output_min, output_max, flags, &p);
  } else if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    maxpool_type_ = OpComputeType::op_compute_type_qs8;
    const int8_t output_min = -128;
    const int8_t output_max = 127;
    status = xnn_create_max_pooling2d_nhwc_s8(input_padding_top, input_padding_right,
                                              input_padding_bottom, input_padding_left,
                                              pooling_height, pooling_width,
                                              stride_height, stride_width,
                                              dilation_height, dilation_width,
                                              C, C, C,  // channels, input_pixel_stride, output_pixel_stride
                                              output_min, output_max, flags, &p);
  } else {
    auto stype = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*X_arg.TypeAsProto()));
    ORT_THROW("unsupported Conv in maxpool, we have FLOAT|UINT8, but got ", stype);
  }
  ORT_ENFORCE(status == xnn_status_success, "xnn_create_max_pooling2d_nhwc_",
              OpTypeToString(maxpool_type_), "failed. Status:", status);

  op0_.reset(p);
}

Status MaxPool::Compute(OpKernelContext* context) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto& X_shape = X.Shape();

  int64_t N = X_shape[0];
  int64_t H = X_shape[1];
  int64_t W = X_shape[2];

  // set the N dim to the correct value
  TensorShapeVector output_dims{output_dims_};
  output_dims[0] = N;
  Tensor* Y = context->Output(0, output_dims);

  // empty input
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  pthreadpool_t threadpool = GetThreadPool();

  auto reshape_fn = xnn_reshape_max_pooling2d_nhwc_f32;
  if (maxpool_type_ == OpComputeType::op_compute_type_qu8)
    reshape_fn = xnn_reshape_max_pooling2d_nhwc_u8;
  else if (maxpool_type_ == OpComputeType::op_compute_type_qs8) {
    reshape_fn = xnn_reshape_max_pooling2d_nhwc_s8;
  }

  auto status = reshape_fn(op0_.get(), N, H, W,
                           /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                           threadpool);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_reshape_max_pooling2d_nhwc_",
                           OpTypeToString(maxpool_type_), " returned ", status);
  }

  if (maxpool_type_ == OpComputeType::op_compute_type_fp32) {
    status = xnn_setup_max_pooling2d_nhwc_f32(op0_.get(), X.Data<float>(), Y->MutableData<float>());
  } else if (maxpool_type_ == OpComputeType::op_compute_type_qu8) {
    status = xnn_setup_max_pooling2d_nhwc_u8(op0_.get(), X.Data<uint8_t>(), Y->MutableData<uint8_t>());
  } else {
    status = xnn_setup_max_pooling2d_nhwc_s8(op0_.get(), X.Data<int8_t>(), Y->MutableData<int8_t>());
  }

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_max_pooling2d_nhwc_",
                           OpTypeToString(maxpool_type_), " returned ", status);
  }

  status = xnn_run_operator(op0_.get(), threadpool);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(MaxPool, kMSInternalNHWCDomain, 8, 9, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                          DataTypeImpl::GetTensorType<uint8_t>(),
                                                                          DataTypeImpl::GetTensorType<int8_t>()}),
                                  MaxPool);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(MaxPool, kMSInternalNHWCDomain, 10, 10, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                          DataTypeImpl::GetTensorType<uint8_t>(),
                                                                          DataTypeImpl::GetTensorType<int8_t>()}),
                                  MaxPool);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(MaxPool, kMSInternalNHWCDomain, 11, 11, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                          DataTypeImpl::GetTensorType<uint8_t>(),
                                                                          DataTypeImpl::GetTensorType<int8_t>()}),
                                  MaxPool);

ONNX_OPERATOR_KERNEL_EX(MaxPool, kMSInternalNHWCDomain, 12, kXnnpackExecutionProvider,
                        KernelDefBuilder()
                            .TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                  DataTypeImpl::GetTensorType<uint8_t>(),
                                                  DataTypeImpl::GetTensorType<int8_t>()}),
                        MaxPool);
}  // namespace xnnpack
}  // namespace onnxruntime

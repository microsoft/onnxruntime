// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/nn/resize.h"

#include <algorithm>
#include <utility>

#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/initializer.h"

namespace onnxruntime {
namespace xnnpack {

bool Resize::IsOnnxNodeSupported(const NodeUnit& node_unit,
                                 const GraphViewer& graph_viewer) {
  bool supported = false;
  do {
    // Resize has 1-4 input.
    const auto& inputs = node_unit.Inputs();
    const auto& x_arg = inputs[0].node_arg;

    const auto* x_type = x_arg.TypeAsProto();
    if (x_type == nullptr ||
        (x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
         x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8 &&
         x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8)) {
      break;
    }

    const auto* x_shape = x_arg.Shape();
    //'bilinear' == 2-D input or 4-D input with outermost 2 scales as 1 (NCHW) or
    // 4-D input with outermost and innermost scales as 1 (NHWC)
    // but we just support 4-d tensor for now, and the channel must be known.
    if (!x_shape || x_shape->dim_size() != 4 || x_shape->dim(1).dim_value() <= 0) {
      break;
    }

    const auto* output_shape = node_unit.Outputs()[0].node_arg.Shape();
    bool length_resized_compatible_pytorch_half_pixel = true;
    // when length_resized > 1, there is no difference between pytorch_half_pixel and half_pixel
    // according onnx spec.
    // if coordinate_transformation_mode is "half_pixel",
    // x_original = (x_resized + 0.5) / scale - 0.5
    //
    // if coordinate_transformation_mode is "pytorch_half_pixel",
    // x_original = length_resized > 1 ? (x_resized + 0.5) / scale - 0.5 : 0
    //
    if (output_shape->dim(2).dim_value() <= 1 || output_shape->dim(1).dim_value() <= 1) {
      length_resized_compatible_pytorch_half_pixel = false;
    }

    // Refer to onnxruntime/core/providers/cpu/tensor/upsamplebase.h,
    size_t scale_idx = 2;
    size_t size_idx = 3;
    auto opset_version = node_unit.SinceVersion();
    if (opset_version == 10) {
      scale_idx = 1;
    }

    ProtoHelperNodeContext nc(node_unit.GetNode());
    OpNodeProtoHelper info(&nc);

    std::string mode;
    info.GetAttrOrDefault<std::string>("mode", &mode, "nearest");
    if (mode != "linear") {
      break;
    }

    // check opset 18
    int64_t antialias = 0;
    info.GetAttrOrDefault<int64_t>("antialias", &antialias, 0);
    if (antialias != 0) {
      break;
    }

    std::vector<int64_t> axes;
    if (info.GetAttrs<int64_t>("axes", axes).IsOK() && axes.size() > 0) {
      break;
    }

    std::string keep_aspect_ratio_policy = info.GetAttrOrDefault<std::string>("keep_aspect_ratio_policy", "stretch");
    if (keep_aspect_ratio_policy != "stretch") {
      break;
    }

    auto extrapolation_value = info.GetAttrOrDefault<float>("extrapolation_value", 0.0f);
    if (extrapolation_value != 0.0F) {
      break;
    }
    ///////////

    // Coordinate transformation mode attr was introduced in version 11.
    // before that asymmetric mode was the only available transformation mode
    std::string coordinate_transform_mode_name =
        opset_version > 10
            ? info.GetAttrOrDefault<std::string>("coordinate_transformation_mode", "half_pixel")
            : "asymmetric";

    if (coordinate_transform_mode_name != "asymmetric" &&
        coordinate_transform_mode_name != "half_pixel" &&
        coordinate_transform_mode_name != "align_corners" &&
        (coordinate_transform_mode_name != "pytorch_half_pixel" || !length_resized_compatible_pytorch_half_pixel)) {
      break;
    }

    auto exclude_outside = info.GetAttrOrDefault<int64_t>("exclude_outside", 0) == 0 ? false : true;
    if (exclude_outside) {
      break;
    }

    // roi  only takes effect when coordinate_transformation_mode is "tf_crop_and_resize"

    // size or scales shouldnt't be provided in the same time but should at least be provided one of them
    const auto* scale_tensor = inputs.size() >= scale_idx + 1
                                   ? graph_viewer.GetConstantInitializer(inputs[scale_idx].node_arg.Name(), true)
                                   : nullptr;
    const auto* size_tensor = inputs.size() >= size_idx + 1
                                  ? graph_viewer.GetConstantInitializer(inputs[size_idx].node_arg.Name(), true)
                                  : nullptr;

    bool has_size = false;
    bool has_scale = false;
    InlinedVector<float> scale(4, 1.0F);
    if (scale_tensor) {
      const Initializer scale_val(*scale_tensor, node_unit.ModelPath());
      auto scale_span = scale_val.DataAsSpan<float>();
      if (scale_span.size() == 4) {
        has_scale = true;
        std::copy(scale_span.begin(), scale_span.end(), scale.begin());
      }
    }

    if (size_tensor) {
      auto input_shape = utils::GetTensorShapeFromTensorShapeProto(*x_shape);
      const Initializer size_val(*size_tensor, node_unit.ModelPath());

      auto size_span = size_val.DataAsSpan<int64_t>();
      if (size_span.size() == 4) {
        has_size = true;
        scale = {size_span[0] / static_cast<float>(input_shape[0]),
                 size_span[1] / static_cast<float>(input_shape[1]),
                 size_span[2] / static_cast<float>(input_shape[2]),
                 size_span[3] / static_cast<float>(input_shape[3])};
      }
    }

    if ((has_size && has_scale) || (!has_size && !has_scale)) {
      break;
    }

    if (scale[0] != 1.0F || (scale[1] != 1.0F && scale[3] != 1.0F)) {
      break;
    }

    // only support xnn_create_resize_bilinear2d_nchw_f32
    const bool is_NHWC = scale[3] == 1.0F;
    if (!is_NHWC && (x_type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_UINT8 ||
                     x_type->tensor_type().elem_type() == ONNX_NAMESPACE::TensorProto_DataType_INT8)) {
      break;
    }

    supported = true;
  } while (false);

  return supported;
}

Resize::Resize(const OpKernelInfo& info) : UpsampleBase(info), XnnpackKernel{info} {
  const auto& node = info.node();
  auto input_defs = node.InputDefs();
  int x_dtype = 0;
  ORT_ENFORCE(GetType(*input_defs[0], x_dtype));
  switch (x_dtype) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      op_type_ = OpComputeType::op_compute_type_fp32;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      op_type_ = OpComputeType::op_compute_type_qu8;
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      op_type_ = OpComputeType::op_compute_type_qs8;
      break;
    default:
      auto stype = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*input_defs[0]->TypeAsProto()));
      ORT_THROW("unsupported op in Resize, we have FLOAT|UINT8|INT8, but get ", stype);
  }

  const auto* x_shape = input_defs[0]->Shape();
  auto input_shape = utils::GetTensorShapeFromTensorShapeProto(*x_shape);
  const Tensor* sizes{nullptr};
  if (sizes_input_idx_ > 0) {
    info.TryGetConstantInput(sizes_input_idx_, &sizes);
  }

  const auto input_dims = input_shape.NumDimensions();
  // if input shape (H,W) is known ahead, we can calculate output shape
  if (input_shape[input_dims - 1] > 0 && input_shape[input_dims - 2] > 0 &&
      (input_dims == 2 || (input_dims > 2 && input_shape[1] > 0))) {
    output_dims_.resize(input_dims);
    if (sizes && sizes->Shape().Size() == 4) {
      scales_.resize(input_shape.NumDimensions());
      ORT_THROW_IF_ERROR(ParseSizesData(sizes, output_dims_, input_shape.GetDims()));
      ORT_THROW_IF_ERROR(ParseScalesDataAndAdjustOutputSize(output_dims_, input_shape.GetDims(), scales_));
      scales_cached_ = true;
    } else {
      ComputeOutputShape(scales_, input_shape.GetDims(), output_dims_);
    }
  }

  is_NHWC_ = scales_[3] == 1.0F;
  int64_t channels = x_shape->dim(is_NHWC_ ? 3 : 1).dim_value();

  uint32_t flags = 0;
  ORT_ENFORCE(mode_ == UpsampleMode::LINEAR, "only support bilinear resize");
  if (coordinate_transform_mode_ == ResizeCoordinateTransformationMode::ALIGN_CORNERS) {
    flags |= XNN_FLAG_ALIGN_CORNERS;
  } else if (!(coordinate_transform_mode_ == ResizeCoordinateTransformationMode::HALF_PIXEL ||
               coordinate_transform_mode_ == ResizeCoordinateTransformationMode::PYTORCH_HALF_PIXEL)) {
    flags |= XNN_FLAG_TENSORFLOW_LEGACY_MODE;
  }

  xnn_status xstatus = xnn_status_invalid_state;
  struct xnn_operator* p = nullptr;
  if (op_type_ == OpComputeType::op_compute_type_fp32) {
    auto create_func = is_NHWC_ ? xnn_create_resize_bilinear2d_nhwc_f32 : xnn_create_resize_bilinear2d_nchw_f32;
    xstatus = create_func(
        channels, channels, channels, flags, &p);
  } else if (op_type_ == OpComputeType::op_compute_type_qu8) {
    xstatus = xnn_create_resize_bilinear2d_nhwc_u8(
        channels, channels, channels, flags, &p);
  } else {
    xstatus = xnn_create_resize_bilinear2d_nhwc_s8(
        channels, channels, channels, flags, &p);
  }
  ORT_ENFORCE(xstatus == xnn_status_success, "xnn_create_resize_bilinear2d_nhwc_",
              OpTypeToString(op_type_), " failed. Status:", xstatus);
  op0_.reset(p);
}

// compute method of Resize
Status Resize::ComputeInternal(OpKernelContext* ctx, const Tensor* input,
                               const TensorShapeVector& output_dims) const {
  const auto& X_shape = input->Shape();
  auto N = X_shape[0];
  auto H = is_NHWC_ ? X_shape[1] : X_shape[2];
  auto W = is_NHWC_ ? X_shape[2] : X_shape[3];
  Tensor* output = ctx->Output(0, TensorShape(output_dims));

  pthreadpool_t t_pool = GetThreadPool();
  xnn_status status = xnn_status_invalid_state;
  if (op_type_ == OpComputeType::op_compute_type_fp32) {
    auto oH = is_NHWC_ ? output_dims[1] : output_dims[2];
    auto oW = is_NHWC_ ? output_dims[2] : output_dims[3];
    auto setup_func = is_NHWC_ ? xnn_setup_resize_bilinear2d_nhwc_f32 : xnn_setup_resize_bilinear2d_nchw_f32;
    status = setup_func(
        op0_.get(),
        N,
        H, W, oH, oW,
        input->Data<float>(),
        output->MutableData<float>(),
        t_pool);
  } else if (op_type_ == OpComputeType::op_compute_type_qu8) {
    status = xnn_setup_resize_bilinear2d_nhwc_u8(
        op0_.get(),
        N,
        H, W, output_dims[1], output_dims[2],
        input->Data<uint8_t>(),
        output->MutableData<uint8_t>(),
        t_pool);
  } else {
    status = xnn_setup_resize_bilinear2d_nhwc_s8(
        op0_.get(),
        N,
        H, W, output_dims[1], output_dims[2],
        input->Data<int8_t>(),
        output->MutableData<int8_t>(),
        t_pool);
  }
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_resize_bilinear2d_nhwc_",
                           OpTypeToString(op_type_), " returned ", status);
  }
  status = xnn_run_operator(op0_.get(), t_pool);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }
  return Status::OK();
}

Status Resize::Compute(OpKernelContext* ctx) const {
  const auto* X = ctx->Input<Tensor>(0);
  TensorShapeVector output_shape(output_dims_);
  if (output_shape.empty()) {
    output_shape.resize(X->Shape().NumDimensions());

    // Get scales data
    const auto* scales = ctx->Input<Tensor>(scales_input_idx_);
    std::vector<float> scales_array(X->Shape().GetDims().size());

    if (scales != nullptr && scales->Shape().Size() != 0) {
      ORT_RETURN_IF_ERROR(ParseScalesData(scales, scales_array, output_shape.size()));
      // Compute output shape from scales and input dims
      ComputeOutputShape(scales_array, X->Shape().GetDims(), output_shape);
    } else {
      const Tensor* sizes = ctx->Input<Tensor>(sizes_input_idx_);
      // When sizes input is available directly populate it into the output_dims array.
      ORT_RETURN_IF_ERROR(ParseSizesData(sizes, output_shape, X->Shape().GetDims()));
      ORT_RETURN_IF_ERROR(ParseScalesDataAndAdjustOutputSize(output_shape, X->Shape().GetDims(), scales_array));
    }
  }
  output_shape[0] = X->Shape()[0];
  return ComputeInternal(ctx, X, output_shape);
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Resize, kOnnxDomain, 10, 10, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                          DataTypeImpl::GetTensorType<uint8_t>(),
                                                                          DataTypeImpl::GetTensorType<int8_t>()}),
                                  Resize);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Resize, kOnnxDomain, 11, 12, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                                                                           DataTypeImpl::GetTensorType<uint8_t>(),
                                                                           DataTypeImpl::GetTensorType<int8_t>()}),
                                  Resize);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Resize, kOnnxDomain, 13, 17, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                                                                           DataTypeImpl::GetTensorType<uint8_t>(),
                                                                           DataTypeImpl::GetTensorType<int8_t>()}),
                                  Resize);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Resize, kOnnxDomain, 18, 18, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                                                                           DataTypeImpl::GetTensorType<uint8_t>(),
                                                                           DataTypeImpl::GetTensorType<int8_t>()}),
                                  Resize);

ONNX_OPERATOR_KERNEL_EX(Resize, kOnnxDomain, 19, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                                                                 DataTypeImpl::GetTensorType<uint8_t>(),
                                                                 DataTypeImpl::GetTensorType<int8_t>()}),
                        Resize);
}  // namespace xnnpack
}  // namespace onnxruntime

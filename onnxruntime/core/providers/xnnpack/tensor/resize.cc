// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/xnnpack/tensor/resize.h"

#include <algorithm>
#include <utility>

#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"
#include "core/framework/op_kernel.h"
#include "core/optimizer/initializer.h"
#include "core/providers/xnnpack/xnnpack_init.h"

namespace onnxruntime {
namespace xnnpack {

bool Resize::IsOnnxNodeSupported(const NodeUnit& node_unit,
                                 const GraphViewer& graph_viewer) {
  bool supported = false;
  do {
    if (node_unit.SinceVersion() < 10) {
      break;
    }

    // Resize has 1-4 input.
    const auto& inputs = node_unit.Inputs();
    const auto& x_arg = inputs[0].node_arg;

    const auto* x_type = x_arg.TypeAsProto();
    if (x_type == nullptr || (x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
                              x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8 &&
                              x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8)) {
      break;
    }

    const auto* x_shape = x_arg.Shape();

    // 'bilinear' == 2-D input or 4-D input with outermost 2 scales as 1 (NCHW) can be supported.
    // we only support 4-d tensor for now, and the channel must be known.
    // we assume the input in NCHW for this test.
    if (!x_shape || x_shape->dim_size() != 4 || x_shape->dim(1).dim_value() <= 0) {
      break;
    }

    // validate it is in fact NCHW
    //
    // opset 10 had `scales` as input 1 and no sizes. later opsets added roi as input 1 followed by scales and sizes.
    auto opset_version = node_unit.SinceVersion();
    size_t scale_idx = opset_version == 10 ? 1 : 2;
    size_t size_idx = 3;

    // onnx shape inferencing validates that one and not both of sizes and scales are provided
    const auto* scale_tensor = inputs.size() >= scale_idx + 1
                                   ? graph_viewer.GetConstantInitializer(inputs[scale_idx].node_arg.Name(), true)
                                   : nullptr;
    const auto* size_tensor = opset_version > 10 && inputs.size() >= size_idx + 1
                                  ? graph_viewer.GetConstantInitializer(inputs[size_idx].node_arg.Name(), true)
                                  : nullptr;

    // if both scales and sizes are nullptr the one that was provided was not a constant initializer
    if (!scale_tensor && !size_tensor) {
      break;
    }

    // check the scale for the second dim is 1 or the size of the second dim matches the input shape.
    // if not, it is not the C dim as a Resize will not change the number of channels.
    InlinedVector<float> scale(4, 1.0F);
    if (scale_tensor) {
      const Initializer scale_val(*scale_tensor, node_unit.ModelPath());
      if (scale_val.DataAsSpan<float>()[1] != 1.0F) {
        break;
      }
    }

    if (size_tensor) {
      const Initializer size_val(*size_tensor, node_unit.ModelPath());
      if (size_val.DataAsSpan<int64_t>()[1] != x_shape->dim(1).dim_value()) {
        break;
      }
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
    if (output_shape->dim(2).dim_value() <= 1 || output_shape->dim(3).dim_value() <= 1) {
      // we don't know the output H or W so we don't know if it will be compatible
      length_resized_compatible_pytorch_half_pixel = false;
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
      // TODO: We should be able to handle this if required
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
        opset_version > 10 ? info.GetAttrOrDefault<std::string>("coordinate_transformation_mode", "half_pixel")
                           : "asymmetric";

    // TODO: Opset 19 added half_pixel_symmetric. Need to see if that can be supported.

    if (coordinate_transform_mode_name != "asymmetric" &&
        coordinate_transform_mode_name != "half_pixel" &&
        coordinate_transform_mode_name != "align_corners" &&
        (coordinate_transform_mode_name != "pytorch_half_pixel" || !length_resized_compatible_pytorch_half_pixel)) {
      break;
    }

    if (info.GetAttrOrDefault<int64_t>("exclude_outside", 0) != 0) {
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

  int64_t channels = x_shape->dim(3).dim_value();

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
    xstatus = xnn_create_resize_bilinear2d_nhwc_f32(channels, channels, channels, flags, &p);
  } else if (op_type_ == OpComputeType::op_compute_type_qu8) {
    xstatus = xnn_create_resize_bilinear2d_nhwc_u8(channels, channels, channels, flags, &p);
  } else {
    xstatus = xnn_create_resize_bilinear2d_nhwc_s8(channels, channels, channels, flags, &p);
  }

  ORT_ENFORCE(xstatus == xnn_status_success, "xnn_create_resize_bilinear2d_nhwc_", OpTypeToString(op_type_), " failed. Status:",
              xstatus);

  op0_.reset(p);
}

// compute method of Resize
Status Resize::ComputeInternal(OpKernelContext* ctx, const Tensor* input,
                               const TensorShapeVector& output_dims) const {
  const auto& X_shape = input->Shape();
  auto N = X_shape[0];
  auto H = X_shape[1];
  auto W = X_shape[2];
  Tensor* output = ctx->Output(0, TensorShape(output_dims));

  pthreadpool_t threadpool = GetThreadPool();

  // setup allocator/automated dellocate for workspace
  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  xnn_allocator* allocator = GetStoredAllocator().second;
  auto deallocator = [allocator](void* ptr) { allocator->aligned_deallocate(allocator->context, ptr); };
  std::unique_ptr<void, decltype(deallocator)> workspace(nullptr, deallocator);

  auto reshape_fn = xnn_reshape_resize_bilinear2d_nhwc_f32;
  if (op_type_ == OpComputeType::op_compute_type_qu8) {
    reshape_fn = xnn_reshape_resize_bilinear2d_nhwc_u8;
  } else if (op_type_ == OpComputeType::op_compute_type_qs8) {
    reshape_fn = xnn_reshape_resize_bilinear2d_nhwc_s8;
  }

  auto status = reshape_fn(op0_.get(), N, H, W, output_dims[1], output_dims[2],
                           &workspace_size, &workspace_alignment, threadpool);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_reshape_resize_bilinear2d_nhwc_", OpTypeToString(op_type_),
                           " returned ", status);
  }

  workspace.reset(allocator->aligned_allocate(allocator->context, XNN_ALLOCATION_ALIGNMENT, workspace_size));

  if (op_type_ == OpComputeType::op_compute_type_fp32) {
    status = xnn_setup_resize_bilinear2d_nhwc_f32(op0_.get(), workspace.get(), input->Data<float>(),
                                                  output->MutableData<float>());
  } else if (op_type_ == OpComputeType::op_compute_type_qu8) {
    status = xnn_setup_resize_bilinear2d_nhwc_u8(op0_.get(), workspace.get(), input->Data<uint8_t>(),
                                                 output->MutableData<uint8_t>());
  } else {
    status = xnn_setup_resize_bilinear2d_nhwc_s8(op0_.get(), workspace.get(), input->Data<int8_t>(),
                                                 output->MutableData<int8_t>());
  }

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_resize_bilinear2d_nhwc_",
                           OpTypeToString(op_type_), " returned ", status);
  }

  status = xnn_run_operator(op0_.get(), threadpool);
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
    InlinedVector<float> scales_array(X->Shape().GetDims().size());

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

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Resize, kMSInternalNHWCDomain, 10, 10, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", {DataTypeImpl::GetTensorType<float>(),
                                                                          DataTypeImpl::GetTensorType<uint8_t>(),
                                                                          DataTypeImpl::GetTensorType<int8_t>()}),
                                  Resize);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Resize, kMSInternalNHWCDomain, 11, 12, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                                                                           DataTypeImpl::GetTensorType<uint8_t>(),
                                                                           DataTypeImpl::GetTensorType<int8_t>()}),
                                  Resize);
ONNX_OPERATOR_VERSIONED_KERNEL_EX(Resize, kMSInternalNHWCDomain, 13, 17, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                                                                           DataTypeImpl::GetTensorType<uint8_t>(),
                                                                           DataTypeImpl::GetTensorType<int8_t>()}),
                                  Resize);

ONNX_OPERATOR_VERSIONED_KERNEL_EX(Resize, kMSInternalNHWCDomain, 18, 18, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                                                                           DataTypeImpl::GetTensorType<uint8_t>(),
                                                                           DataTypeImpl::GetTensorType<int8_t>()}),
                                  Resize);

ONNX_OPERATOR_KERNEL_EX(Resize, kMSInternalNHWCDomain, 19, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                                                                 DataTypeImpl::GetTensorType<uint8_t>(),
                                                                 DataTypeImpl::GetTensorType<int8_t>()}),
                        Resize);
}  // namespace xnnpack
}  // namespace onnxruntime

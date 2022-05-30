// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "max_pool.h"

#include "core/graph/graph.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/detail/utils.h"

// to sanity check output shape
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace xnnpack {
bool MaxPool::IsOnnxNodeSupported(const onnxruntime::Node& node, const GraphViewer& /*graph*/) {
  bool supported = false;

  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // MaxPool has 1 input.
    auto input_defs = node.InputDefs();
    const auto& x_arg = *input_defs[0];

    // we only support float currently
    const auto* x_type = x_arg.TypeAsProto();
    if (x_type == nullptr ||
        x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
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
    : OpKernel(info),
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

  float output_min = clip_min_max_ ? clip_min_max_->first : -INFINITY;
  float output_max = clip_min_max_ ? clip_min_max_->second : INFINITY;

  uint32_t flags = 0;
  if (pool_attrs_.auto_pad == AutoPadType::SAME_UPPER) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }

  // input is NHWC and we only support input with 4 dims. we checked C, H, W were all known in the op support checker
  const auto& X_arg = *Node().InputDefs()[0];
  const auto& X_shape = *X_arg.Shape();
  int64_t H = X_shape.dim(1).dim_value();
  int64_t W = X_shape.dim(2).dim_value();
  int64_t C = X_shape.dim(3).dim_value();

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

  xnn_status status;
  struct xnn_operator* p;
  status = xnn_create_max_pooling2d_nhwc_f32(input_padding_top, input_padding_right,
                                             input_padding_bottom, input_padding_left,
                                             pooling_height, pooling_width,
                                             stride_height, stride_width,
                                             dilation_height, dilation_width,
                                             C, C, C,  // channels, input_pixel_stride, output_pixel_stride
                                             output_min, output_max, flags, &p);
  ORT_ENFORCE(status == xnn_status_success, "xnn_create_max_pooling2d_nhwc_f32 failed. Status:", status);

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

  xnn_status status = xnn_setup_max_pooling2d_nhwc_f32(op0_.get(), N, H, W,
                                                       X.Data<float>(), Y->MutableData<float>(),
                                                       nullptr /*threadpool */);  // TBD: how to handle threading

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_max_pooling2d_nhwc_f32 returned ", status);
  }

  status = xnn_run_operator(op0_.get(), nullptr);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(MaxPool, kMSInternalNHWCDomain, 11, 11, kXnnpackExecutionProvider,
                                  KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                                  MaxPool);

ONNX_OPERATOR_KERNEL_EX(MaxPool, kMSInternalNHWCDomain, 12, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        MaxPool);

}  // namespace xnnpack
}  // namespace onnxruntime

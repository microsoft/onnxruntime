// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/framework/transpose_helper.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace xnnpack {

namespace {
Status CreateXnnpackKernel(const ConvAttributes& conv_attrs,
                           int64_t C, int64_t M,
                           const TensorShapeVector& kernel_shape,
                           const std::optional<std::pair<float, float>>& clip_min_max,
                           bool depthwise,
                           const Tensor& W, const float* B_data,
                           struct xnn_operator*& p) {
  uint32_t kernel_height = gsl::narrow<uint32_t>(kernel_shape[0]);
  uint32_t kernel_width = gsl::narrow<uint32_t>(kernel_shape[1]);

  uint32_t input_padding_top = gsl::narrow<uint32_t>(conv_attrs.pads[0]);
  uint32_t input_padding_left = gsl::narrow<uint32_t>(conv_attrs.pads[1]);
  uint32_t input_padding_bottom = gsl::narrow<uint32_t>(conv_attrs.pads[2]);
  uint32_t input_padding_right = gsl::narrow<uint32_t>(conv_attrs.pads[3]);

  uint32_t subsampling_height = gsl::narrow<uint32_t>(conv_attrs.strides[0]);
  uint32_t subsampling_width = gsl::narrow<uint32_t>(conv_attrs.strides[1]);
  uint32_t dilation_height = gsl::narrow<uint32_t>(conv_attrs.dilations[0]);
  uint32_t dilation_width = gsl::narrow<uint32_t>(conv_attrs.dilations[1]);

  float output_min = clip_min_max ? clip_min_max->first : -INFINITY;
  float output_max = clip_min_max ? clip_min_max->second : INFINITY;

  uint32_t flags = 0;
  if (conv_attrs.auto_pad == AutoPadType::SAME_UPPER) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }

  xnn_status status = xnn_status::xnn_status_uninitialized;
  p = nullptr;

  const float* W_data = W.Data<float>();

  if (depthwise) {
    // C == group and M % group == 0 so M/group is safe
    uint32_t group_output_channels = gsl::narrow<uint32_t>(M / conv_attrs.group);

    status = xnn_create_convolution2d_nhwc_f32(
        input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
        kernel_height, kernel_width,
        subsampling_height, subsampling_width,
        dilation_height, dilation_width,
        // groups, group_input_channels, group_output_channels
        gsl::narrow<uint32_t>(conv_attrs.group), 1, group_output_channels,
        C, M,  // input channel stride, output channel stride
        W_data, B_data,
        output_min, output_max, flags, &p);

  } else {
    status = xnn_create_convolution2d_nhwc_f32(
        input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
        kernel_height, kernel_width,
        subsampling_height, subsampling_width,
        dilation_height, dilation_width,
        gsl::narrow<uint32_t>(conv_attrs.group), C, M,  // groups, group_input_channels, group_output_channels
        C, M,                                           // input channel stride, output channel stride
        W_data, B_data,
        output_min, output_max, flags, &p);
  }

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to create xnnpack kernel. xnn_create_convolution2d_nhwc_f32 returned ", status);
  }

  return Status::OK();
}
}  // namespace

// helper to check whether an ONNX Conv node is supported by the NHWC version
// if this returns true, the layout transformer will be run by GraphPartitioner to convert the first input/output to
// NHWC format, and move the node to the internal NHWC domain.
bool Conv::IsOnnxNodeSupported(const onnxruntime::Node& node, const GraphViewer& graph) {
  bool supported = false;

  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // Conv has at least 2 inputs.
    auto input_defs = node.InputDefs();
    const auto& x_arg = *input_defs[0];
    const auto& weight_arg = *input_defs[1];

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

    // weight must be constant and also rank 4
    const auto* weight = graph.GetConstantInitializer(weight_arg.Name(), true);
    if (weight == nullptr || weight->dims_size() != 4) {
      break;
    }

    // if there's a bias input it must be constant
    if (input_defs.size() == 3) {
      const auto& bias_arg = *input_defs[2];
      if (bias_arg.Exists() && !graph.IsConstantInitializer(bias_arg.Name(), true)) {
        break;
      }
    }

    ProtoHelperNodeContext nc(node);
    OpNodeProtoHelper info(&nc);

    // 'group' value needs to be 1 or C.
    // the second dim of weight is C/group, so if that == 1, group == C
    int64_t group = 0;
    info.GetAttrOrDefault<int64_t>("group", &group, 1);
    if (group != 1 && weight->dims(1) != 1) {
      break;
    }

    // if 'pads' is not specified we use 'auto_pad'
    if (graph_utils::GetNodeAttribute(node, "pads") == nullptr) {
      AutoPadType auto_pad = AutoPadType::NOTSET;

      std::string auto_pad_str;
      if (info.GetAttr<std::string>("auto_pad", &auto_pad_str).IsOK()) {
        // auto_pad was set
        //
        // The "auto_pad_str" string must be either NOTSET, SAME_UPPER, SAME_LOWER or VALID
        // tf2onnx converter doesn't use SAME_LOWER.
        // SAME_UPPER maps to TF SAME padding.
        // TODO: What does PT converter use? We need to support models from PT in mobile.
        auto_pad = StringToAutoPadType(auto_pad_str);
        if (!IsPaddingTypeSupported(auto_pad)) {
          break;
        }
      }
    }

    supported = true;
  } while (false);

  return supported;
}

Conv::Conv(const OpKernelInfo& info) : OpKernel(info), conv_attrs_{info} {
  // get values from any fusion with an activation
  if (info.GetAttr<std::string>("activation", &conv_attrs_.activation).IsOK()) {
    std::vector<float> activation_params;

    // min/max could be from Clip or Relu
    if (info.GetAttrs<float>("activation_params", activation_params).IsOK()) {
      if (activation_params.size() == 2) {
        clip_min_max_ = {activation_params[0], activation_params[1]};
      }
    }
  }

  const auto& node{Node()};

  const auto& input_defs = node.InputDefs();
  const NodeArg& X = *input_defs[0];
  C_ = X.Shape()->dim(3).dim_value();  // input is NHWC. op support checker made sure C dim was known

  // as the weight input is a constant initializer we can calculate all the sizes here instead of in Compute
  const Tensor* W = nullptr;
  ORT_ENFORCE(info.TryGetConstantInput(1, &W),
              "Weight input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
              node.Name());

  // 'M' is first dim of weight. Prepacking will alter the layout of W later
  M_ = W->Shape()[0];

  // this happens before PrePack, so the W input is still in the ONNX spec format
  ORT_THROW_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape_));

  if (conv_attrs_.pads.empty()) {
    conv_attrs_.pads.resize(kernel_shape_.size() * 2, 0);
  }

  if (conv_attrs_.dilations.empty()) {
    conv_attrs_.dilations.resize(kernel_shape_.size(), 1);
  }

  if (conv_attrs_.strides.empty()) {
    conv_attrs_.strides.resize(kernel_shape_.size(), 1);
  }

  // we only take nodes with no bias, or a constant bias.
  bool has_bias = input_defs.size() == 3 && input_defs[2]->Exists();

  ORT_ENFORCE(has_bias == false || info.TryGetConstantInput(2, &B_),
              "Invalid Node with non-constant Bias input. XNNPACK EP should not have asked for the node. Node name:",
              node.Name());

  // have to delay creating the xnnpack kernel until after the weights are pre-packed.
}

// use PrePack to handle the weight layout change as that's not a simple NCHW -> NHWC transpose
Status Conv::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                     /*out*/ bool& is_packed,
                     /*out*/ PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;

  // only layout of weight input is adjusted via PrePack
  if (input_idx == 1) {
    // Transpose from {M, C/group, kH, kW} to {M, kH, kW, C/group}
    auto orig_shape = tensor.Shape();

    std::vector<size_t> perm{0, 2, 3, 1};
    std::vector<int64_t> new_dims{orig_shape[0],
                                  orig_shape[2],
                                  orig_shape[3],
                                  orig_shape[1]};

    packed_w_ = Tensor::Create(tensor.DataType(), TensorShape(new_dims), alloc);

    SingleAxisTranspose(perm, tensor, *packed_w_, /*from*/ 1, /*to*/ 3);

    is_packed = true;

    // we can create the kernel now
    struct xnn_operator* p = nullptr;
    ORT_RETURN_IF_ERROR(CreateXnnpackKernel(conv_attrs_, C_, M_, kernel_shape_, clip_min_max_, IsDepthwise(),
                                            *packed_w_, B_ ? B_->Data<float>() : nullptr, p));

    op0_.reset(p);
  }

  return Status::OK();
}

Status Conv::Compute(OpKernelContext* context) const {
  const Tensor& X = *context->Input<Tensor>(0);  // this is in NHWC format
  const auto& X_shape = X.Shape();
  const int64_t N = X_shape[0];  // input is NHWC
  const int64_t H = X_shape[1];
  const int64_t W = X_shape[2];

  // We don't need to call ValidateInputShape as we checked validity in ConvChecker.
  // We also can't use ValidateInputShape as-is as the weight tensor was pre-packed and the layout was changed there.
  // ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(&X, &W));

  // CPU Conv starts with TensorShapeVector Y_dims({N, M}); and passes in X->Shape().Slice(2);
  // We know this is 2D in NHWC format so we need to start with 'N', pass in the H, W, and append M last
  TensorShapeVector Y_dims({N});
  TensorShape input_shape = {H, W};

  ConvAttributes::ConvPadVector pads(conv_attrs_.pads);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape_,
                                                          conv_attrs_.strides, conv_attrs_.dilations, pads,
                                                          Y_dims));

  Y_dims.push_back(M_);
  Tensor* Y = context->Output(0, TensorShape(Y_dims));

  // Bail out early if one of the dimensions is zero.
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  xnn_status status = xnn_setup_convolution2d_nhwc_f32(op0_.get(), N, H, W, X.Data<float>(), Y->MutableData<float>(),
                                                       nullptr /*threadpool*/);  // TBD: how to handle threading

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_convolution2d_nhwc_f32 returned ", status);
  }

  status = xnn_run_operator(op0_.get(), nullptr);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(Conv, kMSInternalNHWCDomain, 11, kXnnpackExecutionProvider,
                        KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
                        Conv);

}  // namespace xnnpack
}  // namespace onnxruntime

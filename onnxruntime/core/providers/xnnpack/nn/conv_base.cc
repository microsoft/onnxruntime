// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "conv_base.h"

#include "core/common/inlined_containers_fwd.h"
#include "core/graph/constants.h"
#include "core/graph/graph.h"
#include "core/graph/graph_utils.h"
#include "core/framework/transpose_helper.h"
#include "core/providers/utils.h"
#include "core/providers/xnnpack/detail/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/xnnpack/xnnpack_kernel.h"

namespace onnxruntime {
namespace xnnpack {

namespace {
Status CreateXnnpackKernel(const ConvAttributes* conv_attrs_ptr,
                           int64_t C, int64_t M,
                           const TensorShapeVector& kernel_shape,
                           const std::optional<std::pair<float, float>>& clip_min_max,
                           const Tensor& Weight, const Tensor* Bias,
                           XnnpackOperator& op_uptr,
                           xnn_code_cache_t code_cache,
                           xnn_weights_cache_t weights_cache,
                           const OpQuantParam& quant_param,
                           OpComputeType conv_type,
                           bool is_transpose = false) {
  struct xnn_operator* p = nullptr;

  const uint32_t kernel_height = gsl::narrow<uint32_t>(kernel_shape[0]);
  const uint32_t kernel_width = gsl::narrow<uint32_t>(kernel_shape[1]);

  const auto& conv_attrs = *conv_attrs_ptr;
  const uint32_t input_padding_top = gsl::narrow<uint32_t>(conv_attrs.pads[0]);
  const uint32_t input_padding_left = gsl::narrow<uint32_t>(conv_attrs.pads[1]);
  const uint32_t input_padding_bottom = gsl::narrow<uint32_t>(conv_attrs.pads[2]);
  const uint32_t input_padding_right = gsl::narrow<uint32_t>(conv_attrs.pads[3]);

  const uint32_t subsampling_height = gsl::narrow<uint32_t>(conv_attrs.strides[0]);
  const uint32_t subsampling_width = gsl::narrow<uint32_t>(conv_attrs.strides[1]);
  const uint32_t dilation_height = gsl::narrow<uint32_t>(conv_attrs.dilations[0]);
  const uint32_t dilation_width = gsl::narrow<uint32_t>(conv_attrs.dilations[1]);

  uint32_t flags = 0;
  if (conv_attrs.auto_pad == AutoPadType::SAME_UPPER) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }

  xnn_status status = xnn_status::xnn_status_uninitialized;
  p = nullptr;
  float foutput_min = clip_min_max ? clip_min_max->first : -INFINITY;
  float foutput_max = clip_min_max ? clip_min_max->second : INFINITY;
  // with the following IC and OC number, we can cover depthwise and regular conv at the same time
  // the equation 'IC (group_input_channels) == C ' set up when group_count==1 (regular convolution)
  // and OC (group_output_channels) follows the same rule.
  // also, in the case of DepthWiseConv, group_count = C, IC is 1 constantly, OC is what DPconv require.
  // So we can unify it with IC and OC.
  // group is either 1 (for regular conv) or C (for depth-wise conv), and hence M % group == 0 so M/group is safe
  uint32_t group_count = gsl::narrow<uint32_t>(conv_attrs.group);
  size_t group_input_channels = gsl::narrow<size_t>(C / group_count);   // either C or 1
  size_t group_output_channels = gsl::narrow<size_t>(M / group_count);  // either M or M/C
  if (conv_type == OpComputeType::op_compute_type_fp32) {
    auto* B_data = Bias ? Bias->Data<float>() : nullptr;
    auto create_func = is_transpose ? xnn_create_deconvolution2d_nhwc_f32
                                    : xnn_create_convolution2d_nhwc_f32;
    status = create_func(
        input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
        kernel_height, kernel_width,
        subsampling_height, subsampling_width,
        dilation_height, dilation_width,
        group_count,
        group_input_channels, group_output_channels,  // groups, group_input_channels, group_output_channels
        C, M,                                         // input channel stride, output channel stride
        Weight.Data<float>(), B_data,
        foutput_min, foutput_max, flags,
        code_cache, weights_cache,
        &p);
  } else if (conv_type == OpComputeType::op_compute_type_qs8) {
    const float output_scale = quant_param[2].first[0];
    const int8_t output_zero_point = quant_param[2].second;
    const int8_t output_min = xnn_u8s8_quantize<int8_t>(foutput_min, output_scale, output_zero_point);
    const int8_t output_max = xnn_u8s8_quantize<int8_t>(foutput_max, output_scale, output_zero_point);
    auto* B_data = Bias ? Bias->Data<int32_t>() : nullptr;
    auto create_func = is_transpose ? xnn_create_deconvolution2d_nhwc_qs8
                                    : xnn_create_convolution2d_nhwc_qs8;
    status = create_func(
        input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
        kernel_height, kernel_width,
        subsampling_height, subsampling_width,
        dilation_height, dilation_width,
        group_count,
        group_input_channels,
        group_output_channels,
        C, M,
        quant_param[0].second, quant_param[0].first[0],
        quant_param[1].first[0], Weight.Data<int8_t>(), B_data,
        quant_param[2].second, quant_param[2].first[0],
        output_min, output_max,
        flags,
        code_cache, weights_cache,
        &p);
  } else if (conv_type == OpComputeType::op_compute_type_qs8_per_channel) {
    auto* B_data = Bias ? Bias->Data<int32_t>() : nullptr;
    const float output_scale = quant_param[2].first[0];
    const int8_t output_zero_point = quant_param[2].second;
    const int8_t output_min = xnn_u8s8_quantize<int8_t>(foutput_min, output_scale, output_zero_point);
    const int8_t output_max = xnn_u8s8_quantize<int8_t>(foutput_max, output_scale, output_zero_point);
    status = xnn_create_convolution2d_nhwc_qs8_qc8w(
        input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
        kernel_height, kernel_width,
        subsampling_height, subsampling_width,
        dilation_height, dilation_width,
        group_count,
        group_input_channels,
        group_output_channels,
        C, M,
        // zero_point will convert to int8_t automatically
        quant_param[0].second, quant_param[0].first[0],
        quant_param[1].first.data(),
        Weight.Data<int8_t>(), B_data,
        quant_param[2].second, quant_param[2].first[0],
        output_min, output_max,
        flags,
        code_cache, weights_cache,
        &p);
  } else if (conv_type == OpComputeType::op_compute_type_qu8) {
    const auto* B_data = Bias ? Bias->Data<int32_t>() : nullptr;
    const float output_scale = quant_param[2].first[0];
    const uint8_t output_zero_point = quant_param[2].second;
    const uint8_t output_min = xnn_u8s8_quantize<uint8_t>(foutput_min, output_scale, output_zero_point);
    const uint8_t output_max = xnn_u8s8_quantize<uint8_t>(foutput_max, output_scale, output_zero_point);
    auto create_func = is_transpose ? xnn_create_deconvolution2d_nhwc_qu8
                                    : xnn_create_convolution2d_nhwc_qu8;
    status = create_func(
        input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
        kernel_height, kernel_width,
        subsampling_height, subsampling_width,
        dilation_height, dilation_width,
        group_count,
        group_input_channels,
        group_output_channels,
        C, M,
        quant_param[0].second, quant_param[0].first[0],
        quant_param[1].second, quant_param[1].first[0],
        Weight.Data<uint8_t>(), B_data,
        quant_param[2].second, quant_param[2].first[0],
        output_min, output_max,
        flags,
        code_cache, weights_cache,
        &p);
  }

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                           "Failed to create xnnpack kernel. xnn_create_",
                           is_transpose ? "deconvolution2d" : "convolution2d", "_nhwc_",
                           OpTypeToString(conv_type), " returned ", status);
  }

  op_uptr.reset(p);
  return Status::OK();
}

OpComputeType ParseQuantParamAndConType(const OpKernelInfo& info, OpQuantParam& quant_param, int32_t x_dtype) {
  quant_param = ParseQuantParamForOp(info, x_dtype, 2);
  OpComputeType conv_type = OpComputeType::op_compute_type_invalid;
  if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8) {
    // The rules of per-channel quantization is that:
    // X-tensor share the same scalar-scale and zp under per-tensor quantization
    // But we have separate conv weight quantization params for each conv output-channel,
    // and there is total output-channels of scales
    if (quant_param[1].first.size() > 1) {
      conv_type = OpComputeType::op_compute_type_qs8_per_channel;
    } else {
      conv_type = OpComputeType::op_compute_type_qs8;
    }
  } else if (x_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    conv_type = OpComputeType::op_compute_type_qu8;
  }
  return conv_type;
}

// if bias type is int32 and it has no quantparam, the dtype check will be failed GetTensorQuantType
// however, it should be fine.
TensorQuantType TryGetConvBiasDtype(const NodeUnit& node_unit,
                                    const GraphViewer& graph_viewer) {
  // we are not check the legality of io_index here
  const NodeUnitIODef& iodef = node_unit.Inputs()[2];
  int32_t input_type = 0;
  if (!GetType(iodef.node_arg, input_type) || input_type != ONNX_NAMESPACE::TensorProto_DataType_INT32) {
    return TensorTypeInvalid;
  }
  // bias must be a ConstantInitializer
  return graph_viewer.GetConstantInitializer(iodef.node_arg.Name(), true) ? TensorTypeInt32 : TensorTypeInvalid;
}

// this function is refereed to Xnnpack_conv, u8s8 is not support
OpComputeType GetConvCompType(
    TensorQuantType input_datatype,
    TensorQuantType filter_datatype,
    TensorQuantType* bias_datatype,  // could be nullptr
    TensorQuantType output_datatype) {
  switch (filter_datatype) {
    case TensorTypeFp32:
      if (input_datatype == TensorTypeFp32 &&
          (!bias_datatype || *bias_datatype == TensorTypeFp32) &&
          output_datatype == TensorTypeFp32) {
        return op_compute_type_fp32;
      }
      break;
    case TensorTypeInt8:
      if (input_datatype == TensorTypeInt8 &&
          (!bias_datatype || *bias_datatype == TensorTypeInt32) &&
          output_datatype == TensorTypeInt8) {
        return op_compute_type_qs8;
      }
      break;
    case TensorTypeInt8_Per_Channel:
      if (input_datatype == TensorTypeInt8 &&
          // what the bias should be for per-channel quantization?
          // (!bias_datatype || *bias_datatype == TensorTypeInt32_Per_Channel) &&
          output_datatype == TensorTypeInt8) {
        return op_compute_type_qs8_per_channel;
      }
      break;
    case TensorTypeUint8:
      if (input_datatype == TensorTypeUint8 &&
          (!bias_datatype || *bias_datatype == TensorTypeInt32) &&
          output_datatype == TensorTypeUint8) {
        return op_compute_type_qu8;
      }
      break;
    default:
      break;
  }
  LOGS_DEFAULT(VERBOSE) << "unsupported Conv in/out data type:"
                        << "[input_datatype]=" << TensorQtypeToString(input_datatype)
                        << "[filter_datatype]=" << TensorQtypeToString(filter_datatype)
                        << "[bias_datatype]="
                        << (bias_datatype ? TensorQtypeToString(*bias_datatype)
                                          : "")
                        << "[output_datatype]=" << TensorQtypeToString(output_datatype);
  return op_compute_type_invalid;
}

// xnnpack support qc8|qs8|qu8
/*
 * | conv type| input dtype|weight dtype| per channel|zero point handle|
 * | qc8      |  i8        | i8         |  yes       |zero
 * | qcu8     |  xx        | xx         |  yes       |not supported yet
 * | qs8      |  i8        | i8         |  no        |orig_zp
 * | qu8      |  u8        | u8         |  no        |orig_zp + 128
 */
//
bool IsValidQuantConv(const NodeUnit& node_unit, const GraphViewer& graph) {
  bool supported = false;
  do {
    TensorQuantType x_input_type, w_input_type, bias_input_type, output_type;
    TensorQuantType* bias_input_type_ptr = nullptr;
    // quant conv has at least two inputs, x_tensor and weight
    const auto& inputs = node_unit.Inputs();
    x_input_type = GetTensorQuantType(node_unit, 0, false, graph);
    w_input_type = GetTensorQuantType(node_unit, 1, false, graph);
    if (inputs.size() > 2) {
      bias_input_type = TryGetConvBiasDtype(node_unit, graph);
      bias_input_type_ptr = &bias_input_type;
    }
    output_type = GetTensorQuantType(node_unit, 0, true, graph);
    auto qconv_type = GetConvCompType(x_input_type, w_input_type, bias_input_type_ptr, output_type);
    if (op_compute_type_invalid == qconv_type || (qconv_type == op_compute_type_qs8_per_channel && node_unit.OpType() == "QLinearConvTranspose")) {
      break;
    }
    supported = true;
  } while (false);
  return supported;
}

bool IsQuantizedConv(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QLinearConv) ||
         (quant_op_type == QuantizedOpType::QDQConv) ||
         (quant_op_type == QuantizedOpType::QLinearConvTranspose) ||
         (quant_op_type == QuantizedOpType::QDQConvTranspose);
}
}  // namespace

// helper to check whether an ONNX Conv node is supported by the NHWC version
// if this returns true, the layout transformer will be run by GraphPartitioner to convert the first input/output to
// NHWC format, and move the node to the internal NHWC domain.
bool ConvBase::IsOnnxNodeSupported(const NodeUnit& node_unit, const GraphViewer& graph) {
  bool supported = false;
  auto qtype = GetQuantizedOpType(node_unit);
  if (IsQuantizedConv(qtype) && IsValidQuantConv(node_unit, graph) == false) {
    return false;
  }

  const onnxruntime::Node& node = node_unit.GetNode();
  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // Conv has at least 2 inputs.
    const auto& inputs = node_unit.Inputs();
    const auto& x_arg = inputs[0].node_arg;
    const auto& weight_arg = inputs[1].node_arg;

    // we only support 2D (4 dims with batch and channel)
    const auto* x_shape = x_arg.Shape();
    if (!x_shape || x_shape->dim_size() != 4) {
      break;
    }
    // we only support float and u8 currently
    const auto* x_type = x_arg.TypeAsProto();
    if (x_type == nullptr ||
        (x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
         x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8 &&
         x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8)) {
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
    int32_t bias_index = qtype == QuantizedOpType::QLinearConv ? 8 : 2;
    if (inputs.size() == size_t(bias_index + 1)) {
      const auto& bias_arg = node_unit.Inputs()[bias_index].node_arg;
      if (bias_arg.Exists() && !graph.IsConstantInitializer(bias_arg.Name(), true)) {
        break;
      }
    }

    ProtoHelperNodeContext nc(node);
    OpNodeProtoHelper info(&nc);

    // xnnpack support group conv, especially for group = 1 or group = in_channel

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

ConvBase::ConvBase(const OpKernelInfo& info, bool is_transpose)
    : XnnpackKernel(info, /*enable_caches*/ true),
      conv_attrs_(info),
      conv_transpose_attrs_(info),
      convbase_attrs_ref_(is_transpose ? conv_transpose_attrs_ : conv_attrs_),
      is_transpose_(is_transpose) {
  // get values from any fusion with an activation
  if (info.GetAttr<std::string>("activation", &convbase_attrs_ref_.activation).IsOK()) {
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
  auto X_shape = utils::GetTensorShapeFromTensorShapeProto(*X.Shape());
  C_ = X_shape[3];  // input is NHWC. op support checker made sure C dim was known

  // as the weight input is a constant initializer we can calculate all the sizes here instead of in Compute
  const Tensor* Weight = nullptr;
  int weight_index = 1;
  auto input_dtype = X.TypeAsProto()->tensor_type().elem_type();
  if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    conv_type_ = OpComputeType::op_compute_type_fp32;
  } else if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_INT8 ||
             input_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    weight_index = 3;
    conv_type_ = ParseQuantParamAndConType(info, quant_param_, input_dtype);
  } else {
    auto stype = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*X.TypeAsProto()));
    ORT_THROW("unsupported Conv in XnnpackEP, we have FLOAT|UINT8|INT8, but got ", stype);
  }

  ORT_ENFORCE(info.TryGetConstantInput(weight_index, &Weight),
              "Weight input was not constant initializer. XNNPACK EP should not have asked for the node. Node name:",
              node.Name());
  M_ = Weight->Shape()[0];

  // this happens before PrePack, so the weight input is still in the ONNX spec format
  ORT_THROW_IF_ERROR(convbase_attrs_ref_.ComputeKernelShape(Weight->Shape(), kernel_shape_));

  if (convbase_attrs_ref_.pads.empty()) {
    convbase_attrs_ref_.pads.resize(kernel_shape_.size() * 2, 0);
  }

  if (convbase_attrs_ref_.dilations.empty()) {
    convbase_attrs_ref_.dilations.resize(kernel_shape_.size(), 1);
  }

  if (convbase_attrs_ref_.strides.empty()) {
    convbase_attrs_ref_.strides.resize(kernel_shape_.size(), 1);
  }

  // we only take nodes with no bias, or a constant bias.
  bool has_bias = input_defs.size() == 3 && input_defs[2]->Exists();
  if (conv_type_ == OpComputeType::op_compute_type_fp32) {
    ORT_ENFORCE(has_bias == false || info.TryGetConstantInput(2, &B_),
                "Invalid Node with non-constant Bias input. XNNPACK EP should not have asked for the node. Node name:",
                node.Name());
  } else {
    has_bias = input_defs.size() == (8 + 1) && input_defs[8]->Exists();
    ORT_ENFORCE(has_bias == false || info.TryGetConstantInput(8, &B_),
                "Invalid Node with non-constant Bias input. XNNPACK EP should not have asked for the node. Node name:",
                node.Name());
  }
  const TensorShape input_shape{X_shape[1], X_shape[2]};

  if (is_transpose) {
    // Group_num group_size
    M_ = Weight->Shape()[1] * convbase_attrs_ref_.group;
    if (conv_transpose_attrs_.output_padding.empty()) {
      conv_transpose_attrs_.output_padding.resize(kernel_shape_.size(), 0);
    }

    conv_transpose_attrs_.ComputePadsAndOutputShape(
        input_shape, M_, kernel_shape_,
        conv_transpose_attrs_.strides, conv_transpose_attrs_.dilations,
        conv_transpose_attrs_.output_padding, 1, &conv_transpose_attrs_.pads, &output_shape_);
    output_shape_[1] = output_shape_[2];
    output_shape_[2] = output_shape_[3];
    output_shape_[3] = M_;
  } else {
    ConvAttributes::ConvPadVector pads(conv_attrs_.pads);
    output_shape_.push_back(1);
    ORT_THROW_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape_,
                                                           conv_attrs_.strides, conv_attrs_.dilations, pads,
                                                           output_shape_));

    output_shape_.push_back(M_);
  }
  // have to delay creating the xnnpack kernel until after the weights are pre-packed.
}

Status ConvBase::CreateKernel() {
  auto ret = CreateXnnpackKernel(&convbase_attrs_ref_, C_, M_, kernel_shape_, clip_min_max_, packed_w_,
                                 B_, op0_,
                                 GetCodeCache(), GetWeightsCache(),
                                 quant_param_, conv_type_, is_transpose_);
  return ret;
}
}  // namespace xnnpack
}  // namespace onnxruntime

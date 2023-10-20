// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/providers/xnnpack/nn/average_pool.h"

#include "core/common/status.h"
#include "core/graph/graph.h"
#include "core/providers/utils.h"
#include "core/framework/tensorprotoutils.h"
#include "core/providers/xnnpack/xnnpack_init.h"
#include "core/providers/xnnpack/detail/utils.h"

namespace onnxruntime {
namespace xnnpack {
namespace {
Status CreateXnnpackKernel(const PoolAttributes& pool_attrs,
                           int64_t C,
                           const std::optional<std::pair<float, float>>& clip_min_max,
                           struct xnn_operator*& p,
                           const OpQuantParam& quant_param,
                           OpComputeType avgpool_type) {
  uint32_t input_padding_top = gsl::narrow<uint32_t>(pool_attrs.pads[0]);
  uint32_t input_padding_left = gsl::narrow<uint32_t>(pool_attrs.pads[1]);
  uint32_t input_padding_bottom = gsl::narrow<uint32_t>(pool_attrs.pads[2]);
  uint32_t input_padding_right = gsl::narrow<uint32_t>(pool_attrs.pads[3]);

  uint32_t pooling_height = gsl::narrow<uint32_t>(pool_attrs.kernel_shape[0]);
  uint32_t pooling_width = gsl::narrow<uint32_t>(pool_attrs.kernel_shape[1]);
  uint32_t stride_height = gsl::narrow<uint32_t>(pool_attrs.strides[0]);
  uint32_t stride_width = gsl::narrow<uint32_t>(pool_attrs.strides[1]);

  uint32_t flags = 0;
  if (pool_attrs.auto_pad == AutoPadType::SAME_UPPER) {
    flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
  }
  float foutput_min = clip_min_max ? clip_min_max->first : -INFINITY;
  float foutput_max = clip_min_max ? clip_min_max->second : INFINITY;
  xnn_status status = xnn_status_unsupported_parameter;
  if (avgpool_type == OpComputeType::op_compute_type_fp32) {
    status = xnn_create_average_pooling2d_nhwc_f32(input_padding_top, input_padding_right,
                                                   input_padding_bottom, input_padding_left,
                                                   pooling_height, pooling_width,
                                                   stride_height, stride_width,
                                                   C, C, C,  // channels, input_pixel_stride, output_pixel_stride
                                                   foutput_min, foutput_max, flags, &p);
  } else if (avgpool_type == OpComputeType::op_compute_type_qu8) {
    const float output_scale = quant_param[1].first[0];
    const uint8_t output_zero_point = quant_param[1].second;
    const uint8_t output_min = xnn_u8s8_quantize<uint8_t>(foutput_min, output_scale, output_zero_point);
    const uint8_t output_max = xnn_u8s8_quantize<uint8_t>(foutput_max, output_scale, output_zero_point);
    status = xnn_create_average_pooling2d_nhwc_qu8(input_padding_top, input_padding_right,
                                                   input_padding_bottom, input_padding_left,
                                                   pooling_height, pooling_width,
                                                   stride_height, stride_width,
                                                   C, C, C,  // channels, input_pixel_stride, output_pixel_stride
                                                   quant_param[0].second,
                                                   quant_param[0].first[0],
                                                   quant_param[1].second,
                                                   quant_param[1].first[0],
                                                   output_min, output_max, flags, &p);
  }

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_create_average_pooling2d_nhwc_",
                           OpTypeToString(avgpool_type), " failed. Status:", status);
  }
  return Status::OK();
}

bool IsQuantAvgPoolSupported(const NodeUnit& node_unit, const GraphViewer& graph) {
  TensorQuantType x_input_type = GetTensorQuantType(node_unit, 0, false, graph);
  TensorQuantType output_type = GetTensorQuantType(node_unit, 0, true, graph);
  return (x_input_type == TensorTypeUint8 && output_type == TensorTypeUint8);
}

bool IsQuantizedAvgPool(QuantizedOpType quant_op_type) {
  return (quant_op_type == QuantizedOpType::QlinearAvgPool) ||
         (quant_op_type == QuantizedOpType::QDQAvgPool);
}

}  // namespace

bool AveragePool::IsOnnxNodeSupported(const NodeUnit& node_unit,
                                      const GraphViewer& graph) {
  bool supported = false;
  auto qtype = GetQuantizedOpType(node_unit);
  // we check quant-conditions first, if this quant-node is not supported, return directly.
  if (IsQuantizedAvgPool(qtype) && IsQuantAvgPoolSupported(node_unit, graph) == false) {
    return false;
  }
  // share the common checks here for fp32 and quant-op
  const auto& inputs = node_unit.Inputs();
  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // AveragePool has 1 input.
    const auto& x_arg = inputs[0].node_arg;

    // we only support 2D (4 dims with batch and channel)
    const auto* x_shape = x_arg.Shape();
    if (!x_shape || x_shape->dim_size() != 4) {
      break;
    }
    // we only support float and u8 currently
    const auto* x_type = x_arg.TypeAsProto();
    if (x_type == nullptr ||
        (x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT &&
         x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8)) {
      break;
    }

    // require C, H, W to be known so we can construct the xnnpack kernel prior to Compute
    if (!x_shape->dim(1).has_dim_value() ||
        !x_shape->dim(2).has_dim_value() ||
        !x_shape->dim(3).has_dim_value()) {
      break;
    }

    ProtoHelperNodeContext nc(node_unit.GetNode());
    OpNodeProtoHelper info(&nc);
    PoolAttributes pool_attrs(info, "AveragePool", node_unit.SinceVersion());

    // xnnpack doesn't appear to support using 'ceil' to calculate the output shape
    // https://github.com/google/XNNPACK/blob/3caa8b9de973839afa1e2a1462ff356e6927a66b/src/operators/average-pooling-nhwc.c#L643
    // calls compute_output_dimension but there's no ability to specify rounding that value up.
    if (pool_attrs.ceil_mode != 0) {
      break;
    }

    if (!IsPaddingTypeSupported(pool_attrs.auto_pad)) {
      break;
    }

    if ((pool_attrs.kernel_shape.size() != 2) ||
        (pool_attrs.kernel_shape[0] == 1 && pool_attrs.kernel_shape[1] == 1)) {
      // XNNPack doesn't support 1x1 average pool.
      break;
    }

    // a weird design in xnnpack, fp32 should have count_include_pad false,
    // while quant should have count_include_pad true
    bool cp_in_op = pool_attrs.count_include_pad ^ IsQuantizedAvgPool(qtype);
    if (cp_in_op) {
      break;
    }

    supported = true;
  } while (false);

  return supported;
}

AveragePool::AveragePool(const OpKernelInfo& info)
    : XnnpackKernel(info),
      pool_attrs_{info, "AveragePool", info.node().SinceVersion()} {
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

  OpQuantParam quant_param;
  // TEMPORARY sanity check. If C, H and W are known, the output shape should have been able to be inferred, with the
  // exception of the batch size. Can be removed once we've run more models using xnnpack AveragePool.
  auto inferred_output_shape = utils::GetTensorShapeFromTensorShapeProto(*Node().OutputDefs()[0]->Shape());
  ORT_ENFORCE(inferred_output_shape[1] == output_dims_[1] &&
                  inferred_output_shape[2] == output_dims_[2] &&
                  inferred_output_shape[3] == output_dims_[3],
              "Shape mismatch between inferred value and calculated value.");
  const auto& input_dtype = X_arg.TypeAsProto()->tensor_type().elem_type();
  if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    avgpool_type_ = OpComputeType::op_compute_type_fp32;
  } else if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_UINT8) {
    // the order of input tensor, x,x_scale, x_zp, y_scale, y_zp
    quant_param = ParseQuantParamForOp(info, input_dtype, 1);
    avgpool_type_ = OpComputeType::op_compute_type_qu8;
  } else {
    auto stype = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*X_arg.TypeAsProto()));
    ORT_THROW("unsupported AveragePool in XnnpackEP, we have FLOAT|UINT8, but got ", stype);
  }
  struct xnn_operator* p;
  auto ret = CreateXnnpackKernel(pool_attrs_, C, clip_min_max_, p,
                                 quant_param, avgpool_type_);
  ORT_ENFORCE(ret.IsOK(), ret.ErrorMessage());
  op0_.reset(p);
}

Status AveragePool::Compute(OpKernelContext* context) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto& X_shape = X.Shape();

  int64_t N = X_shape[0];
  int64_t H = X_shape[1];
  int64_t W = X_shape[2];

  // set the N dim to the correct value
  TensorShapeVector output_dims{output_dims_};
  output_dims[0] = N;
  Tensor& Y = *context->Output(0, output_dims);

  // empty input
  if (Y.Shape().Size() == 0) {
    return Status::OK();
  }

  pthreadpool_t threadpool = GetThreadPool();

  // setup allocator/automated dellocate for workspace
  size_t workspace_size = 0;
  size_t workspace_alignment = 0;
  xnn_allocator* allocator = GetStoredAllocator().second;
  auto deallocator = [allocator](void* ptr) { allocator->aligned_deallocate(allocator->context, ptr); };

  std::unique_ptr<void, decltype(deallocator)> workspace(nullptr, deallocator);

  auto reshape_fn = (avgpool_type_ == OpComputeType::op_compute_type_fp32)
                        ? xnn_reshape_average_pooling2d_nhwc_f32
                        : xnn_reshape_average_pooling2d_nhwc_qu8;

  auto status = reshape_fn(op0_.get(), N, H, W,
                           &workspace_size, &workspace_alignment,
                           /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
                           threadpool);

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_reshape_average_pooling2d_nhwc_", OpTypeToString(avgpool_type_),
                           " returned ", status);
  }

  workspace.reset(allocator->aligned_allocate(allocator->context, workspace_size, XNN_ALLOCATION_ALIGNMENT));
  status = xnn_setup_average_pooling2d_nhwc_f32(op0_.get(), workspace.get(),
                                                X.Data<float>(), Y.MutableData<float>());

  if (avgpool_type_ == OpComputeType::op_compute_type_fp32) {
    status = xnn_setup_average_pooling2d_nhwc_f32(op0_.get(), workspace.get(),
                                                  X.Data<float>(), Y.MutableData<float>());

  } else if (avgpool_type_ == OpComputeType::op_compute_type_qu8) {
    status = xnn_setup_average_pooling2d_nhwc_qu8(op0_.get(), workspace.get(),
                                                  X.Data<uint8_t>(), Y.MutableData<uint8_t>());
  }

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_average_pooling2d_nhwc_", OpTypeToString(avgpool_type_),
                           " returned ", status);
  }

  status = xnn_run_operator(op0_.get(), threadpool);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

ONNX_OPERATOR_KERNEL_EX(
    AveragePool, kMSInternalNHWCDomain, 11,
    kXnnpackExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    AveragePool);

ONNX_OPERATOR_KERNEL_EX(
    QLinearAveragePool, kMSInternalNHWCDomain, 1,
    kXnnpackExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<uint8_t>()),
    AveragePool);

}  // namespace xnnpack
}  // namespace onnxruntime

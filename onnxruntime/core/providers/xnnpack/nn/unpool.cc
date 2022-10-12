// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "unpool.h"
#include <algorithm>

#include "core/common/common.h"
#include "core/framework/tensor_shape.h"
#include "core/graph/graph.h"
#include "core/providers/cpu/nn/pool_attributes.h"
#include "core/providers/utils.h"
#include "core/framework/tensorprotoutils.h"

// to sanity check output shape
#include "core/framework/tensorprotoutils.h"
#include "gsl/gsl-lite.hpp"

namespace onnxruntime {
namespace xnnpack {
namespace {

TensorShapeVector InferOutputSizeForUnPool(const PoolAttributes& pool_attrs,
                                           const TensorShape& input_shape) {
  // Calculate output tensor shape from attributes
  TensorShapeVector inferred_output_dims(input_shape.NumDimensions());

  // Copy batch and channel dims
  inferred_output_dims[0] = input_shape[0];
  inferred_output_dims[3] = input_shape[3];

  // For feature dims calculate reversing the formula used for MaxPool
  for (size_t dim = 0; dim < pool_attrs.kernel_shape.size(); ++dim) {
    inferred_output_dims[dim + 1] =
        (input_shape[dim + 1] - 1) * pool_attrs.strides[dim] -
        (pool_attrs.pads[dim] + pool_attrs.pads[pool_attrs.kernel_shape.size() + dim]) +
        pool_attrs.kernel_shape[dim];
  }

  return inferred_output_dims;
}
}  // namespace

// MaxUnpool doesn't have any quantization params
bool MaxUnpool::IsOnnxNodeSupported(const NodeUnit& node_unit,
                                    const GraphViewer&) {
  bool supported = false;
  const onnxruntime::Node& node = node_unit.GetNode();
  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // MaxUnpool has 2 inputs.
    auto input_defs = node.InputDefs();
    const auto& x_arg = *input_defs[0];

    const auto* x_type = x_arg.TypeAsProto();
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

    ProtoHelperNodeContext nc(node);
    OpNodeProtoHelper info(&nc);
    PoolAttributes pool_attrs(info, "MaxUnpool", node.SinceVersion());

    if (!IsPaddingTypeSupported(pool_attrs.auto_pad)) {
      break;
    }

    supported = true;
  } while (false);

  return supported;
}

MaxUnpool::MaxUnpool(const OpKernelInfo& info)
    : XnnpackKernel(info),
      pool_attrs_{info, "MaxUnpool", info.node().SinceVersion()} {
  num_inputs_ = OpKernel::Node().InputDefs().size();

  // input is NHWC and we only support input with 4 dims. we checked C, H, W were all known in the op support checker
  const auto& X_arg = *Node().InputDefs()[0];
  auto X_shape = utils::GetTensorShapeFromTensorShapeProto(*X_arg.Shape());

  int64_t H = X_shape[1];
  int64_t W = X_shape[2];
  int64_t C = X_shape[3];

  // create NCHW shape to calculate most of the output shape. 'N' is set in Compute.
  TensorShapeVector input_shape{1, C, H, W};
  auto pads = pool_attrs_.pads;
  output_dims_ = InferOutputSizeForUnPool(pool_attrs_, X_shape);

  if (num_inputs_ == 3) {
    const Tensor* output_shape_tensor = nullptr;
    ORT_ENFORCE(info.TryGetConstantInput(2, &output_shape_tensor), "Get output shape tensor failed");
    const auto out_sp = output_shape_tensor->DataAsSpan<int64_t>();
    if (std::accumulate(pool_attrs_.pads.begin(), pool_attrs_.pads.end(), 0LL) != 0) {
      // use to calculate output shape for xnnpack
      pool_attrs_.pads[0] = out_sp[2] - (H - 1) * pool_attrs_.strides[0] + pool_attrs_.kernel_shape[0];
      pool_attrs_.pads[1] = 0;
      pool_attrs_.pads[2] = out_sp[3] - (W - 1) * pool_attrs_.strides[1] + pool_attrs_.kernel_shape[1];
      pool_attrs_.pads[3] = 0;
    }
    output_dims_.assign(out_sp.cbegin(), out_sp.cend());
    std::swap(output_dims_[1], output_dims_[2]);
    std::swap(output_dims_[3], output_dims_[2]);
  }

  // TEMPORARY sanity check. If C, H and W are known, the output shape should have been able to be inferred, with the
  // exception of the batch size. Can be removed once we've run more models using xnnpack MaxUnpool.
  auto inferred_output_shape = utils::GetTensorShapeFromTensorShapeProto(*Node().OutputDefs()[0]->Shape());
  ORT_ENFORCE(inferred_output_shape[1] == output_dims_[1] &&
                  inferred_output_shape[2] == output_dims_[2] &&
                  inferred_output_shape[3] == output_dims_[3],
              "Shape mismatch between inferred value and calculated value.");
  uint32_t input_padding_top = gsl::narrow<uint32_t>(pool_attrs_.pads[0]);
  uint32_t input_padding_left = gsl::narrow<uint32_t>(pool_attrs_.pads[1]);
  uint32_t input_padding_bottom = gsl::narrow<uint32_t>(pool_attrs_.pads[2]);
  uint32_t input_padding_right = gsl::narrow<uint32_t>(pool_attrs_.pads[3]);

  uint32_t pooling_height = gsl::narrow<uint32_t>(pool_attrs_.kernel_shape[0]);
  uint32_t pooling_width = gsl::narrow<uint32_t>(pool_attrs_.kernel_shape[1]);

  auto input_dtype = X_arg.TypeAsProto()->tensor_type().elem_type();
  xnn_status status = xnn_status_invalid_state;
  struct xnn_operator* p = nullptr;
  if (input_dtype == ONNX_NAMESPACE::TensorProto_DataType_FLOAT) {
    op_type_ = OpComputeType::op_compute_type_fp32;
    status = xnn_create_unpooling2d_nhwc_x32(input_padding_top, input_padding_right,
                                             input_padding_bottom, input_padding_left,
                                             pooling_height, pooling_width,
                                             C, C, C,  // channels, input_pixel_stride, output_pixel_stride
                                             0, &p);
  } else {
    auto stype = DataTypeImpl::ToString(DataTypeImpl::TypeFromProto(*X_arg.TypeAsProto()));
    ORT_THROW("unsupported Conv in MaxUnpool, we have FLOAT, but got ", stype);
  }
  ORT_ENFORCE(status == xnn_status_success, "xnn_create_max_unpooling2d_nhwc_",
              OpTypeToString(op_type_), "failed. Status:", status);

  op0_.reset(p);
}

Status MaxUnpool::Compute(OpKernelContext* context) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto& indice = *context->Input<Tensor>(1);
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

  AllocatorPtr alloc;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));

  // allocate memory for indice
  size_t indice_size = indice.Shape().Size();
  auto u32_indice_ptr = IAllocator::MakeUniquePtr<uint32_t>(alloc, indice_size);
  auto u32_indice_span = gsl::make_span(u32_indice_ptr.get(), indice_size);
  std::transform(indice.DataAsSpan<int64_t>().cbegin(), indice.DataAsSpan<int64_t>().cend(),
                 u32_indice_span.begin(),
                 [](int64_t i64_d) { return gsl::narrow_cast<uint32_t>(i64_d); });

  pthreadpool_t t_pool = GetThreadPool();
  xnn_status status = xnn_status_invalid_state;
  status = xnn_setup_unpooling2d_nhwc_x32(op0_.get(), N, H, W,
                                          X.Data<float>(), u32_indice_span.data(), Y->MutableData<float>(),
                                          t_pool /*threadpool */);

  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_setup_unpooling2d_nhwc_",
                           OpTypeToString(op_type_), " returned ", status);
  }

  status = xnn_run_operator(op0_.get(), t_pool);
  if (status != xnn_status_success) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "xnn_run_operator returned ", status);
  }

  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MaxUnpool, kMSInternalNHWCDomain, 9, 10, kXnnpackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    MaxUnpool);

ONNX_OPERATOR_KERNEL_EX(
    MaxUnpool, kMSInternalNHWCDomain, 11, kXnnpackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    MaxUnpool);
}  // namespace xnnpack
}  // namespace onnxruntime

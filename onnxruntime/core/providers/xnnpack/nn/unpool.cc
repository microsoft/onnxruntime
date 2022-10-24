// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "unpool.h"

#include "core/common/common.h"
#include "core/common/inlined_containers_fwd.h"
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

// NHWC input/output
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
                                    const GraphViewer& graph_viewer) {
  bool supported = false;
  const onnxruntime::Node& node = node_unit.GetNode();
  // use do {} while(false) so it's easier to set a breakpoint on the return
  do {
    // MaxUnpool has 2-3 inputs.
    auto input_defs = node.InputDefs();

    // the third input is output_shape.
    if (input_defs.size() == 3) {
      // only support fixed output_shape with dimension 4
      const auto* output_shape = graph_viewer.GetConstantInitializer(input_defs[2]->Name(), true);
      if (output_shape == nullptr || output_shape->dims_size() != 1 || output_shape->dims(0) != 4) {
        break;
      }
    }
    const auto& x_arg = *input_defs[0];

    const auto* x_type = x_arg.TypeAsProto();
    if (x_type == nullptr ||
        ((x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_FLOAT) &&
         (x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_INT8) &&
         (x_type->tensor_type().elem_type() != ONNX_NAMESPACE::TensorProto_DataType_UINT8))) {
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
  const auto& input_defs = info.node().InputDefs();
  num_inputs_ = input_defs.size();

  auto X_shape = utils::GetTensorShapeFromTensorShapeProto(*input_defs[0]->Shape());
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
      ORT_THROW("unsupported op in Resize, we have FLOAT|UINT8, but got ", stype);
  }

  // create NCHW shape to calculate most of the output shape. 'N' is set in Compute.
  auto pads = pool_attrs_.pads;
  output_dims_ = InferOutputSizeForUnPool(pool_attrs_, X_shape);
  if (num_inputs_ == 3) {
    const Tensor* output_shape_tensor = nullptr;
    info.TryGetConstantInput(2, &output_shape_tensor);
    if (output_shape_tensor) {
      const auto out_sp = output_shape_tensor->DataAsSpan<int64_t>();
      output_dims_.assign(out_sp.cbegin(), out_sp.cend());
      // MaxUnpool is layout sensitive, so we handle the other inputs manually.
      // NCHW-->NHWC
      std::swap(output_dims_[1], output_dims_[2]);
      std::swap(output_dims_[3], output_dims_[2]);
    } else {
      output_dims_.clear();
    }
  }
  // we wouldn't have a chance to create the xnnpack kernel given that indices is impossible produced by xnnpack
}

template <int nBits>
struct XbitType;

template <>
struct XbitType<32> {
  using XType = int32_t;
};

template <>
struct XbitType<8> {
  using XType = int8_t;
};

template <typename T>
inline void ComputeMaxUnpool(const T* X_data, T* out, const int64_t* I_data,
                             int64_t HW, int64_t oHW, int64_t Channel,
                             std::ptrdiff_t first, std::ptrdiff_t last) {
  const int64_t oCHW = Channel * oHW;
  const int64_t CHW = Channel * HW;

  for (std::ptrdiff_t nhw1 = first; nhw1 < last; ++nhw1) {
    const int64_t n1 = nhw1 / HW;
    const int64_t hw1 = nhw1 % HW;

    const int64_t src_base = n1 * CHW + hw1 * Channel;
    const int64_t dst_base = n1 * CHW + hw1;
    for (int c1 = 0; c1 < Channel; ++c1) {
      const int64_t dst_ind_in_nchw = I_data[c1 * HW + dst_base];
      const int64_t hw_p = dst_ind_in_nchw % oHW;  //(dst_ind_in_nchw - n_p * oCHW - c_p * oHW);
      const int64_t n_p = (dst_ind_in_nchw / oCHW);
      const int64_t c_p = (dst_ind_in_nchw - n_p * oCHW) / oHW;

      out[n_p * oCHW + c_p + hw_p * Channel] = X_data[src_base + c1];
    }
  }
}

Status MaxUnpool::Compute(OpKernelContext* context) const {
  const auto& X = *context->Input<Tensor>(0);
  const auto& indice = *context->Input<Tensor>(1);
  const auto& X_shape = X.Shape();
  const int64_t N = X_shape[0];
  // set the N dim to the correct value
  TensorShapeVector output_dims{output_dims_};
  if (num_inputs_ == 3 && output_dims.empty()) {
    const auto& output_shape = *context->Input<Tensor>(2);
    auto output_shape_span = output_shape.DataAsSpan<int64_t>();
    output_dims[1] = output_shape_span[2];
    output_dims[2] = output_shape_span[3];
    output_dims[3] = output_shape_span[1];
  }

  output_dims[0] = N;  // batch size could be dynamic

  Tensor* Y = context->Output(0, output_dims);
  const int64_t HW = X_shape[1] * X_shape[2];
  const int64_t Channel = X_shape[3];
  const int64_t oHW = output_dims[1] * output_dims[2];

  // No computation is involved, just copy and paste so we convert data from float to int32 to make it faster
  const auto* X_data_x32 = reinterpret_cast<const int32_t*>(X.DataRaw());
  // for any int8/uint8
  const auto* X_data_x8 = reinterpret_cast<const int8_t*>(X.DataRaw());
  const auto* I_data = indice.Data<int64_t>();
  auto out_x32 = gsl::make_span(reinterpret_cast<int32_t*>(Y->MutableDataRaw()), Y->Shape().Size());
  auto out_x8 = gsl::make_span(reinterpret_cast<int8_t*>(Y->MutableDataRaw()), Y->Shape().Size());
  if (op_type_ != OpComputeType::op_compute_type_fp32) {
    std::fill_n(out_x8.data(), out_x8.size(), int8_t(0));
  } else {
    std::fill_n(out_x32.data(), out_x32.size(), int32_t(0));
  }

  using onnxruntime::TensorOpCost;
  using onnxruntime::concurrency::ThreadPool;
  if (op_type_ == OpComputeType::op_compute_type_fp32) {
    ThreadPool::TryParallelFor(
        context->GetOperatorThreadPool(), N * HW,
        // Read 3*N (max,sum,div) write N (div), computation=Read
        TensorOpCost{static_cast<double>(2),
                     static_cast<double>(1),
                     static_cast<double>(10)},
        [X_data_x32, out_x32, I_data, HW, oHW, Channel](std::ptrdiff_t first, std::ptrdiff_t last) {
          using xType = typename XbitType<32>::XType;
          ComputeMaxUnpool<xType>(X_data_x32, out_x32.data(), I_data, HW, oHW, Channel, first, last);
        });

  } else {
    ThreadPool::TryParallelFor(
        context->GetOperatorThreadPool(), N * HW,
        // Read 3*N (max,sum,div) write N (div), computation=Read
        TensorOpCost{static_cast<double>(2),
                     static_cast<double>(1),
                     static_cast<double>(10)},
        [X_data_x8, out_x8, I_data, HW, oHW, Channel](std::ptrdiff_t first, std::ptrdiff_t last) {
          using xType = typename XbitType<8>::XType;
          ComputeMaxUnpool<xType>(X_data_x8, out_x8.data(), I_data, HW, oHW, Channel, first, last);
        });
  }
  return Status::OK();
}

ONNX_OPERATOR_VERSIONED_KERNEL_EX(
    MaxUnpool, kMSInternalNHWCDomain, 9, 10, kXnnpackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                               DataTypeImpl::GetTensorType<int8_t>(),
                               DataTypeImpl::GetTensorType<uint8_t>()})
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    MaxUnpool);

ONNX_OPERATOR_KERNEL_EX(
    MaxUnpool, kMSInternalNHWCDomain, 11, kXnnpackExecutionProvider,
    KernelDefBuilder()
        .TypeConstraint("T1", {DataTypeImpl::GetTensorType<float>(),
                               DataTypeImpl::GetTensorType<int8_t>(),
                               DataTypeImpl::GetTensorType<uint8_t>()})
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int64_t>()),
    MaxUnpool);
}  // namespace xnnpack
}  // namespace onnxruntime

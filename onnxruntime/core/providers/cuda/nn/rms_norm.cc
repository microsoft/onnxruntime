// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cuda/nn/rms_norm.h"
#include "core/providers/cuda/nn/layer_norm.h"
#include "core/providers/cuda/nn/layer_norm_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cpu/nn/layer_norm_helper.h"
#include <limits>
#include <vector>

namespace onnxruntime {
namespace cuda {

// The default stage-one stash is FP32 independently of input and scale types.
#define REGISTER_KERNEL_TYPED(T, V)                                                                 \
  ONNX_OPERATOR_TYPED_KERNEL_EX(RMSNormalization, kOnnxDomain, 23, T##_##V, kCudaExecutionProvider, \
                                (*KernelDefBuilder::Create())                                       \
                                    .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())          \
                                    .TypeConstraint("V", DataTypeImpl::GetTensorType<V>()),         \
                                RMSNorm<T, float, V>);

REGISTER_KERNEL_TYPED(float, float)
REGISTER_KERNEL_TYPED(float, MLFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, MLFloat16)
REGISTER_KERNEL_TYPED(MLFloat16, float)
REGISTER_KERNEL_TYPED(BFloat16, BFloat16)
REGISTER_KERNEL_TYPED(double, double)

// The following code is shared from "core/providers/cuda/nn/layer_norm.cc".
// It is used to implement the RMSNorm kernel, which is a simplified version of LayerNorm.
template <typename T, typename U, typename V>
RMSNorm<T, U, V>::RMSNorm(const OpKernelInfo& op_kernel_info)
    : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());
  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;
  stash_type_ = op_kernel_info.GetAttrOrDefault<int64_t>("stash_type", 1);
  ORT_ENFORCE(stash_type_ == 1 ||
                  (stash_type_ == 11 && std::is_same_v<T, double> && std::is_same_v<V, double>),
              "CUDA RMSNormalization supports stash_type=1 (FLOAT), or stash_type=11 (DOUBLE) "
              "with double input and scale.");
}

template <typename T, typename U, typename V>
Status RMSNorm<T, U, V>::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<T>::MappedType CudaT;
  typedef typename ToCudaType<U>::MappedType CudaU;
  typedef typename ToCudaType<V>::MappedType CudaV;
  // Inputs
  const Tensor* X = ctx->Input<Tensor>(0);
  const Tensor* scale = ctx->Input<Tensor>(1);

  auto X_data = reinterpret_cast<const CudaT*>(X->Data<T>());
  auto scale_data = reinterpret_cast<const CudaV*>(scale->Data<V>());
  auto bias_data = nullptr;

  const TensorShape& x_shape = X->Shape();
  auto x_num_dims = x_shape.NumDimensions();
  const int64_t axis = HandleNegativeAxis(axis_, x_num_dims);

  const TensorShape& scale_shape = scale->Shape();
  const TensorShape& bias_shape = TensorShape();

  LayerNormParams params;
  ORT_RETURN_IF_ERROR(LayerNormHelper::CheckInputs(x_shape, scale_shape, bias_shape, bias_data != nullptr, axis, params));

  // Outputs
  Tensor* Y = ctx->Output(0, x_shape);
  auto Y_data = reinterpret_cast<CudaV*>(Y->MutableData<V>());

  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  // Validate that norm_size won't cause integer overflow in host-side arithmetic.
  // HostApplyLayerNorm computes: (n2 + 4 * warp_size - 1) where warp_size is typically 32.
  // Maximum safe value: INT_MAX - (4 * max_warp_size) to prevent overflow.
  // Using 256 as a conservative upper bound for 4 * warp_size.
  constexpr int MAX_WARP_FACTOR = 256;
  const int MAX_NORM_SIZE = std::numeric_limits<int>::max() - MAX_WARP_FACTOR;

  ORT_RETURN_IF(params.num_rows > 0 &&
                    (params.norm_size > MAX_NORM_SIZE ||
                     params.norm_size > std::numeric_limits<int>::max() / params.num_rows),
                "RMSNormalization input is too large for CUDA kernel indexing: norm_size exceeds safe limits or num_rows * norm_size exceeds INT_MAX");

  // Unlike historical SimplifiedLayerNormalization, ONNX RMSNorm rounds before scaling.
  if constexpr (std::is_same_v<CudaT, double> && std::is_same_v<CudaV, double>) {
    if (stash_type_ == 11) {
      HostApplyLayerNorm<CudaT, double, CudaV, true, true>(
          GetDeviceProp(), Stream(ctx), Y_data, nullptr, nullptr, X_data,
          onnxruntime::narrow<int>(params.num_rows), onnxruntime::narrow<int>(params.norm_size), epsilon_,
          scale_data, bias_data,
          onnxruntime::narrow<int>(params.broadcast_param));
    } else {
      HostApplyLayerNorm<CudaT, CudaU, CudaV, true, true>(
          GetDeviceProp(), Stream(ctx), Y_data, nullptr, nullptr, X_data,
          onnxruntime::narrow<int>(params.num_rows), onnxruntime::narrow<int>(params.norm_size), epsilon_,
          scale_data, bias_data,
          onnxruntime::narrow<int>(params.broadcast_param));
    }
  } else {
    HostApplyLayerNorm<CudaT, CudaU, CudaV, true, true>(
        GetDeviceProp(), Stream(ctx), Y_data, nullptr, nullptr, X_data,
        onnxruntime::narrow<int>(params.num_rows), onnxruntime::narrow<int>(params.norm_size), epsilon_,
        scale_data, bias_data,
        onnxruntime::narrow<int>(params.broadcast_param));
  }
  CUDA_RETURN_IF_ERROR(cudaGetLastError());
  return Status::OK();
}
}  // namespace cuda
}  // namespace onnxruntime

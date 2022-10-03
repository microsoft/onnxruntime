// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/quantization/qordered_ops/qordered_layer_norm.h"
#include "contrib_ops/cuda/quantization/qordered_ops/qordered_layer_norm_impl.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    QOrderedLayerNormalization,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("Q", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("S", DataTypeImpl::GetTensorType<float>())
        .TypeConstraint("F", BuildKernelDefConstraints<float, MLFloat16>())
        .InputMemoryType(OrtMemTypeCPUInput, 1)   // scale_X
        .InputMemoryType(OrtMemTypeCPUInput, 4),  // scale_Y
    QOrderedLayerNormalization);

QOrderedLayerNormalization::QOrderedLayerNormalization(const OpKernelInfo& op_kernel_info)
    : CudaKernel(op_kernel_info) {
  ORT_ENFORCE(op_kernel_info.GetAttr("axis", &axis_).IsOK());

  float tmp_epsilon;
  ORT_ENFORCE(op_kernel_info.GetAttr<float>("epsilon", &tmp_epsilon).IsOK());
  epsilon_ = tmp_epsilon;

  ORT_ENFORCE(op_kernel_info.GetAttr("order_X", &order_X_).IsOK());

  ORT_ENFORCE(op_kernel_info.GetAttr("order_Y", &order_Y_).IsOK());

  ORT_ENFORCE(order_X_ == 1, "QOrderedLayerNormlalization: Only Row major data ordering is currently supported");

  ORT_ENFORCE(order_X_ == order_Y_, "QOrderedLayerNormlalization: Input ordering should match the output ordering");
}

Status QOrderedLayerNormalization::ComputeInternal(OpKernelContext* ctx) const {
  typedef typename ToCudaType<int8_t>::MappedType CudaQ;

  // Inputs
  const Tensor* X = ctx->Input<Tensor>(0);
  const auto* X_data = reinterpret_cast<const CudaQ*>(X->Data<int8_t>());
  const TensorShape& x_shape = X->Shape();
  ORT_ENFORCE(x_shape.GetDims().size() == 3,
              "QOrderedLayerNormlalization: Input shape must be {batch, rows, cols}");

  const Tensor* scale = ctx->Input<Tensor>(2);
  const void* scale_data = scale->DataRaw();

  const Tensor* bias = ctx->Input<Tensor>(3);
  const void* bias_data = (nullptr == bias) ? nullptr : bias->DataRaw();

  const int64_t axis = HandleNegativeAxis(axis_, x_shape.NumDimensions());
  ORT_ENFORCE(axis == 2,
              "QOrderedLayerNormlalization: Implementation only "
              "supports on normalizing on innermost axis");

  unsigned int batch = gsl::narrow<unsigned int>(x_shape.GetDims()[0]);
  unsigned int rows = gsl::narrow<unsigned int>(x_shape.GetDims()[1]);
  unsigned int cols = gsl::narrow<unsigned int>(x_shape.GetDims()[2]);

  if (cols & 0x03) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "QOrderedLayerNormlalization: Cols MUST be a multiple of 4");
  }

  // Outputs
  Tensor* Y = ctx->Output(0, x_shape);

  if (x_shape.Size() == 0) {
    return Status::OK();
  }

  auto* Y_data = reinterpret_cast<CudaQ*>(Y->MutableData<int8_t>());

  const float* scale_x = ctx->Input<Tensor>(1)->Data<float>();
  const float* scale_y = ctx->Input<Tensor>(4)->Data<float>();

  if (scale->IsDataType<MLFloat16>()) {
    return QOrderedLayerNorm(Stream(ctx), GetDeviceProp(), static_cast<cublasLtOrder_t>(order_X_),
                             X_data, *scale_x, Y_data, *scale_y, static_cast<const __half*>(scale_data),
                             static_cast<const __half*>(bias_data),
                             static_cast<float>(epsilon_), batch, rows, cols);
  } 
  return QOrderedLayerNorm(Stream(ctx), GetDeviceProp(), static_cast<cublasLtOrder_t>(order_X_),
                           X_data, *scale_x, Y_data, *scale_y, static_cast<const float*>(scale_data),
                           static_cast<const float*>(bias_data),
                           static_cast<float>(epsilon_), batch, rows, cols);
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <string>

#include "contrib_ops/cuda/bert/deepseek_v4_compression_common.h"
#include "contrib_ops/cuda/bert/hash_router.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace {

template <typename T>
class HashRouter final : public CudaKernel {
 public:
  explicit HashRouter(const OpKernelInfo& info) : CudaKernel(info) {
    const std::string function = info.GetAttrOrDefault<std::string>("score_function", "sigmoid");
    ORT_ENFORCE(function == "sigmoid" || function == "sqrtsoftplus");
    score_function_ = function == "sigmoid" ? 0 : 1;
    scaling_factor_ = info.GetAttrOrDefault<float>("routed_scaling_factor", 1.0f);
    epsilon_ = info.GetAttrOrDefault<float>("epsilon", 1e-20f);
    ORT_ENFORCE(epsilon_ > 0.0f);
  }
  Status ComputeInternal(OpKernelContext* context) const override {
    const Tensor* hidden = context->Input<Tensor>(0);
    const Tensor* ids = context->Input<Tensor>(1);
    const Tensor* weight = context->Input<Tensor>(2);
    const Tensor* table = context->Input<Tensor>(3);
    ORT_RETURN_IF_NOT(hidden && ids && weight && table && hidden->Shape().NumDimensions() >= 1 &&
                          weight->Shape().NumDimensions() == 2 && table->Shape().NumDimensions() == 2,
                      "HashRouter input rank mismatch.");
    const int64_t hidden_size = hidden->Shape()[hidden->Shape().NumDimensions() - 1];
    const int64_t token_count = hidden->Shape().Size() / hidden_size;
    const int64_t num_experts = weight->Shape()[0];
    const int64_t selected_count = table->Shape()[1];
    ORT_RETURN_IF_NOT(weight->Shape()[1] == hidden_size && ids->Shape().Size() == token_count &&
                          ids->DataType() == table->DataType(), "HashRouter input shape or integer type mismatch.");
    TensorShape logits_shape = hidden->Shape();
    logits_shape[logits_shape.NumDimensions() - 1] = num_experts;
    TensorShapeVector selected_dims(ids->Shape().GetDims().begin(), ids->Shape().GetDims().end());
    selected_dims.push_back(selected_count);
    const TensorShape selected_shape(selected_dims);
    Tensor* logits = context->Output(0, logits_shape);
    Tensor* routing = context->Output(1, selected_shape);
    Tensor* experts = context->Output(2, selected_shape);
    using CudaT = typename ToCudaType<T>::MappedType;
    ORT_RETURN_IF_ERROR(ProjectTransposed<T>(
      *this, context, *hidden, *weight, narrow<int>(token_count), narrow<int>(hidden_size),
      narrow<int>(num_experts), reinterpret_cast<CudaT*>(logits->MutableData<T>())));
    if (ids->IsDataType<int64_t>()) {
      return LaunchHashRouterKernel<CudaT, int64_t>(
          Stream(context), reinterpret_cast<CudaT*>(routing->MutableData<T>()), experts->MutableData<int64_t>(),
          reinterpret_cast<const CudaT*>(logits->Data<T>()), ids->Data<int64_t>(), table->Data<int64_t>(),
          narrow<int>(token_count), narrow<int>(num_experts), narrow<int>(selected_count), narrow<int>(table->Shape()[0]),
          score_function_, scaling_factor_, epsilon_, GetDeviceProp().maxThreadsPerBlock);
    }
    ORT_RETURN_IF_NOT(ids->IsDataType<int32_t>(), "HashRouter ids must be int32 or int64.");
    return LaunchHashRouterKernel<CudaT, int32_t>(
        Stream(context), reinterpret_cast<CudaT*>(routing->MutableData<T>()), experts->MutableData<int32_t>(),
        reinterpret_cast<const CudaT*>(logits->Data<T>()), ids->Data<int32_t>(), table->Data<int32_t>(),
        narrow<int>(token_count), narrow<int>(num_experts), narrow<int>(selected_count), narrow<int>(table->Shape()[0]),
        score_function_, scaling_factor_, epsilon_, GetDeviceProp().maxThreadsPerBlock);
  }
 private:
  int score_function_{};
  float scaling_factor_{};
  float epsilon_{};
};


}  // namespace

#define REGISTER_KERNEL(T)                                               \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      HashRouter, kMSDomain, 1, T, kCudaExecutionProvider,                    \
      (*KernelDefBuilder::Create())                                     \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())       \
          .TypeConstraint("I", {DataTypeImpl::GetTensorType<int32_t>(), \
                                  DataTypeImpl::GetTensorType<int64_t>()}), \
      HashRouter<T>);

REGISTER_KERNEL(MLFloat16)
REGISTER_KERNEL(BFloat16)

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "orttraining/training_ops/cuda/math/batch_scale.h"
#include "orttraining/training_ops/cuda/math/batch_scale_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    BatchScale,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>()),
    BatchScale);

// Put implementation in the anonymous namespace to avoid name collision in the global namespace.
namespace {

template <typename T>
struct BatchScaleFunctor {
  void operator()(cudaStream_t stream,
                  int64_t input_element_count,
                  const Tensor* input_tensor,
                  const std::vector<float>& scales,
                  const std::vector<Tensor*>& output_tensors) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    std::vector<CudaT*> output_data_ptrs;
    output_data_ptrs.reserve(output_tensors.size());
    for (Tensor* output_tensor : output_tensors) {
      output_data_ptrs.push_back(reinterpret_cast<CudaT*>(output_tensor->MutableData<T>()));
    }

    BatchScaleImpl<CudaT>(stream, input_element_count, reinterpret_cast<const CudaT*>(input_tensor->Data<T>()),
                          scales, output_data_ptrs);
  }
};
}  // namespace

Status BatchScale::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input_tensor = context->Input<Tensor>(0);

  size_t output_count = scale2_.has_value() ? 3 : 2;
  const auto& input_tensor_shape = input_tensor->Shape();
  std::vector<Tensor*> output_tensors;
  output_tensors.reserve(output_count);
  for (size_t i = 0; i < output_count; ++i) {
    output_tensors.push_back(context->Output(static_cast<int>(i), input_tensor_shape));
  }

  std::vector<float> scales{scale0_, scale1_};
  if (output_count == 3) {
    scales.push_back(scale2_.value());
  }

  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(input_tensor->GetElementType());
  t_disp.Invoke<BatchScaleFunctor>(Stream(context), input_tensor_shape.Size(),
                                   input_tensor, scales, output_tensors);
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

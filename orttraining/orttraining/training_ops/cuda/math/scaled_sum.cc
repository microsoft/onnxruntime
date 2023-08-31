// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>

#include "orttraining/training_ops/cuda/math/scaled_sum.h"
#include "orttraining/training_ops/cuda/math/scaled_sum_impl.h"

namespace onnxruntime {
namespace cuda {

ONNX_OPERATOR_KERNEL_EX(
    ScaledSum,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>()),
    ScaledSum);

// Put implementation in the anonymous namespace to avoid name collision in the global namespace.
namespace {

template <typename T>
struct ScaledSumFunctor {
  void operator()(cudaStream_t stream,
                  int64_t input_element_count,
                  const std::vector<const Tensor*>& input_tensors,
                  const std::vector<float>& scales,
                  Tensor* output_tensor) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    std::vector<const CudaT*> input_data_ptrs;
    input_data_ptrs.reserve(input_tensors.size());
    for (const Tensor* input_tensor : input_tensors) {
      input_data_ptrs.push_back(reinterpret_cast<const CudaT*>(input_tensor->Data<T>()));
    }

    ScaledSumImpl<CudaT>(stream, input_element_count, input_data_ptrs, scales,
                         reinterpret_cast<CudaT*>(output_tensor->MutableData<T>()));
  }
};
}  // namespace

Status ScaledSum::ComputeInternal(OpKernelContext* context) const {
  std::vector<const Tensor*> input_tensors;
  input_tensors.reserve(3);

  for (size_t i = 0; i < 3; ++i) {
    const Tensor* input_tensor = context->Input<Tensor>(static_cast<int>(i));
    if (!input_tensor)
      continue;
    input_tensors.push_back(input_tensor);
  }

  ORT_ENFORCE(input_tensors.size() > 1, "Number of input tensors must be greater than 1.");

  const auto& first_input_tensor_shape = input_tensors[0]->Shape();
  for (size_t i = 1; i < input_tensors.size(); ++i) {
    ORT_ENFORCE(input_tensors[i]->Shape() == first_input_tensor_shape,
                "Shape of input tensors must be the same.");
  }

  std::vector<float> scales{scale0_, scale1_};
  if (input_tensors.size() == 3) {
    ORT_ENFORCE(scale2_.has_value(), "Scale 2 must be specified.");
    scales.push_back(scale2_.value());
  }

  Tensor* output_tensor = context->Output(0, first_input_tensor_shape);
  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(input_tensors[0]->GetElementType());

  t_disp.Invoke<ScaledSumFunctor>(Stream(context), first_input_tensor_shape.Size(),
                                  input_tensors, scales, output_tensor);
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

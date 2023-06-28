// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .InputMemoryType(OrtMemTypeCPUInput, 5)
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>()),
    ScaledSum);

// Put implementation in the anonymous namespace to avoid name collision in the global namespace.
namespace {

template <typename T>
struct ScaledSumFunctor {
  void operator()(cudaStream_t stream,
                  int64_t input_element_count,
                  std::vector<const Tensor*>& input_tensors,
                  std::vector<const Tensor*>& scale_tensors,
                  Tensor* output_tensor) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    std::vector<const CudaT*> input_data_ptrs;
    input_data_ptrs.reserve(input_tensors.size());
    for (const Tensor* input_tensor : input_tensors) {
      input_data_ptrs.push_back(reinterpret_cast<const CudaT*>(input_tensor->Data<T>()));
    }

    std::vector<const CudaT*> scale_data_ptrs;
    scale_data_ptrs.reserve(scale_tensors.size());
    for (const Tensor* scale_tensor : scale_tensors) {
      scale_data_ptrs.push_back(reinterpret_cast<const CudaT*>(scale_tensor->Data<T>()));
    }

    ScaledSumImpl<CudaT>(stream, input_element_count, input_data_ptrs, scale_data_ptrs,
                         reinterpret_cast<CudaT*>(output_tensor->MutableData<T>()));
  }
};
}  // namespace

Status ScaledSum::ComputeInternal(OpKernelContext* context) const {
  // std::cout << "ScaledSum::ComputeInternal Starts" << std::endl;
  std::vector<const Tensor*> input_tensors;
  input_tensors.reserve(3);
  std::vector<const Tensor*> scale_tensors;
  scale_tensors.reserve(3);

  for (size_t i = 0; i < 6; ++i) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    if (!input_tensor)
      continue;
    if (i % 2 == 0) {
      input_tensors.push_back(input_tensor);
    } else {
      scale_tensors.push_back(input_tensor);
    }
  }

  ORT_ENFORCE(input_tensors.size() > 1, "Number of input tensors must be greater than 1.");

  ORT_ENFORCE(input_tensors.size() == scale_tensors.size(),
              "Number of input tensors and scale tensors must be the same.");

  const auto& first_input_tensor_shape = input_tensors[0]->Shape();
  for (size_t i = 1; i < input_tensors.size(); ++i) {
    ORT_ENFORCE(input_tensors[i]->Shape() == first_input_tensor_shape,
                "Shape of input tensors must be the same.");
  }

  for (size_t i = 0; i < input_tensors.size(); ++i) {
    ORT_ENFORCE(scale_tensors[i]->Shape().Size() == 1,
                "Scale tensor must be a scalar.");
  }

  Tensor* output_tensor = context->Output(0, first_input_tensor_shape);
  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(input_tensors[0]->GetElementType());
  // std::cout << "ScaledSum::ComputeInternal Launch" << std::endl;
  t_disp.Invoke<ScaledSumFunctor>(Stream(context), first_input_tensor_shape.Size(),
                                  input_tensors, scale_tensors, output_tensor);
  // std::cout << "ScaledSum::ComputeInternal Ends" << std::endl;
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

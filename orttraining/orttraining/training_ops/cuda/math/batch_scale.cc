// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

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
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 3)
        .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float, double, BFloat16>()),
    BatchScale);

// Put implementation in the anonymous namespace to avoid name collision in the global namespace.
namespace {

template <typename T>
struct BatchScaleFunctor {
  void operator()(cudaStream_t stream,
                  int64_t input_element_count,
                  const Tensor* input_tensor,
                  const std::vector<const Tensor*>& scale_tensors,
                  const std::vector<Tensor*>& output_tensors) const {
    typedef typename ToCudaType<T>::MappedType CudaT;

    std::vector<CudaT*> output_data_ptrs;
    output_data_ptrs.reserve(output_tensors.size());
    for (Tensor* output_tensor : output_tensors) {
      output_data_ptrs.push_back(reinterpret_cast<CudaT*>(output_tensor->MutableData<T>()));
    }

    std::vector<const CudaT*> scale_data_ptrs;
    scale_data_ptrs.reserve(scale_tensors.size());
    for (const Tensor* scale_tensor : scale_tensors) {
      scale_data_ptrs.push_back(reinterpret_cast<const CudaT*>(scale_tensor->Data<T>()));
    }

    BatchScaleImpl<CudaT>(stream, input_element_count, reinterpret_cast<const CudaT*>(input_tensor->Data<T>()),
                          scale_data_ptrs, output_data_ptrs);
  }
};
}  // namespace

Status BatchScale::ComputeInternal(OpKernelContext* context) const {
  // std::cout << "BatchScale::ComputeInternal Starts" << std::endl;
  std::vector<Tensor*> output_tensors;
  output_tensors.reserve(3);
  std::vector<const Tensor*> scale_tensors;
  scale_tensors.reserve(3);

  for (size_t i = 1; i < 4; ++i) {
    const Tensor* input_tensor = context->Input<Tensor>(i);
    if (!input_tensor)
      continue;

    scale_tensors.push_back(input_tensor);
  }

  ORT_ENFORCE(scale_tensors.size() > 1, "Number of scale tensors must be greater than 1.");

  const Tensor* input_tensor = context->Input<Tensor>(0);

  const auto& input_tensor_shape = input_tensor->Shape();
  for (size_t i = 0; i < scale_tensors.size(); ++i) {
    ORT_ENFORCE(scale_tensors[i]->Shape().Size() == 1,
                "Scale tensor must be a scalar.");
  }

  for (size_t i = 0; i < scale_tensors.size(); ++i) {
    output_tensors.push_back(context->Output(i, input_tensor_shape));
  }

  utils::MLTypeCallDispatcher<float, MLFloat16, double, BFloat16> t_disp(input_tensor->GetElementType());
  t_disp.Invoke<BatchScaleFunctor>(Stream(context), input_tensor_shape.Size(),
                                   input_tensor, scale_tensors, output_tensors);
  // std::cout << "BatchScale::ComputeInternal Ends" << std::endl;
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

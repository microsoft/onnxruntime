// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/training_ops/cuda/tensor/view.h"

namespace onnxruntime {
namespace cuda {

namespace {

constexpr int view_count_limit = 1024;  // limit of of output count

std::vector<std::pair<int, int>> GenerateAliasMapping() {
  std::vector<std::pair<int, int>> alias_pairs{};
  for (int i = 0; i < view_count_limit; ++i) {
    alias_pairs.emplace_back(std::make_pair(0, i));
  }
  return alias_pairs;
}

std::vector<int> GenerateInputMemoryType() {
  std::vector<int> input_indexes{};
  for (int i = 1; i < 1 + view_count_limit; ++i) {
    input_indexes.emplace_back(i);
  }
  return input_indexes;
}
}  // namespace

ONNX_OPERATOR_KERNEL_EX(
    View,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("T", DataTypeImpl::AllFixedSizeTensorTypes())
        .TypeConstraint("Shape", DataTypeImpl::GetTensorType<int64_t>())
        .InputMemoryType(OrtMemTypeCPUInput, GenerateInputMemoryType())  // all shape inputs are in CPU
        .Alias(GenerateAliasMapping()),                                  // all output tensors are sharing the same bffer as input[0],
                                                                         // execept that the byte_offset is different
    View);

Status View::ComputeInternal(OpKernelContext* context) const {
  const Tensor* X = context->Input<Tensor>(0);
  size_t bytes_per_elem = X->DataType()->Size();

  int view_count = context->InputCount() - 1;
  std::vector<TensorShape> y_shapes(view_count);
  std::vector<size_t> y_byte_offsets(view_count);
  size_t byte_offset = 0;
  for (int i = 0; i < view_count; ++i) {
    const Tensor* shape_tensor = context->Input<Tensor>(i + 1);
    if (shape_tensor->Shape().NumDimensions() != 1) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "A shape tensor must be a vector tensor, got ", shape_tensor->Shape().NumDimensions(), " dimensions");
    }

    size_t n_dims = static_cast<size_t>(shape_tensor->Shape()[0]);
    const int64_t* shape_data = shape_tensor->template Data<int64_t>();

    y_shapes[i] = TensorShape(shape_data, n_dims);
    y_byte_offsets[i] = byte_offset;
    byte_offset += y_shapes[i].Size() * bytes_per_elem;
  }

  if (byte_offset != X->SizeInBytes()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "The input view shapes doesn't adds up to match input buffer size.");
  }

  const void* X_data = X->DataRaw();
  for (int i = 0; i < view_count; ++i) {
    // Outputs are allowed to be unused.
    Tensor* Y = context->Output(i, y_shapes[i]);
    if (Y != nullptr) {
      if (X_data != Y->MutableDataRaw()) {
        // View output is not sharing the underlaying buffer of input, copy instead
        const void* source = static_cast<const char*>(X_data) + y_byte_offsets[i];
        void* target = Y->MutableDataRaw();
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(target, source, Y->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream()));
      } else {
        Y->SetByteOffset(y_byte_offsets[i]);
      }
    }
  }

  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime

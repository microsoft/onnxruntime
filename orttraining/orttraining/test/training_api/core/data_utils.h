// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace training {
namespace test {

template <typename T>
void OrtValueToVec(const OrtValue& val, std::vector<T>& output) {
  const Tensor& tensor = val.Get<Tensor>();
  int64_t num_elem = tensor.Shape().Size();
  const T* val_ptr = tensor.template Data<T>();
  output.assign(val_ptr, val_ptr + num_elem);
}

template <typename T>
void CudaOrtValueToCpuVec(const OrtValue& val, std::vector<T>& output,
                          std::shared_ptr<IExecutionProvider> cuda_provider,
                          std::shared_ptr<IExecutionProvider> cpu_provider) {
  const Tensor& src_tensor = val.Get<Tensor>();

  auto allocator = cpu_provider->GetAllocator(0, OrtMemTypeDefault);
  ORT_ENFORCE(allocator, "Cpu allocator is a nullptr.");
  auto dst_tensor = std::make_unique<Tensor>(src_tensor.DataType(), src_tensor.Shape(), allocator);

  auto data_transfer = cuda_provider->GetDataTransfer();
  ORT_ENFORCE(data_transfer, "Cuda data transfer is a nullptr.");

  ORT_THROW_IF_ERROR(data_transfer->CopyTensor(src_tensor, *dst_tensor));

  const T* val_ptr = dst_tensor->template Data<T>();
  output.assign(val_ptr, val_ptr + src_tensor.Shape().Size());
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime

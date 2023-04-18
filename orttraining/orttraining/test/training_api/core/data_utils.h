// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <vector>

#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "default_providers.h"

namespace onnxruntime {
namespace training {
namespace test {

template <typename T>
void CpuOrtValueToVec(const OrtValue& src_cpu_ortvalue, std::vector<T>& output) {
  const Tensor& tensor = src_cpu_ortvalue.Get<Tensor>();
  int64_t num_elem = tensor.Shape().Size();
  const T* val_ptr = tensor.template Data<T>();
  output.assign(val_ptr, val_ptr + num_elem);
}

template <typename T>
void CudaOrtValueToCpuVec(const OrtValue& src_cuda_ortvalue, std::vector<T>& output) {
  static std::unique_ptr<IExecutionProvider> cuda_provider = onnxruntime::test::DefaultCudaExecutionProvider();
  static std::unique_ptr<IExecutionProvider> cpu_provider = onnxruntime::test::DefaultCpuExecutionProvider();

  const Tensor& src_tensor = src_cuda_ortvalue.Get<Tensor>();

  auto allocator = cpu_provider->GetAllocator(OrtMemTypeDefault);
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

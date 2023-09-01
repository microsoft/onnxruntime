// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <random>
#include <vector>

#include "core/framework/ort_value.h"
#include "core/framework/tensor.h"
#include "default_providers.h"
#include "test/framework/test_utils.h"
#include "test/util/include/test_utils.h"

namespace onnxruntime::training::test {

template <typename T>
void CpuOrtValueToVec(const OrtValue& src_cpu_ortvalue, std::vector<T>& output) {
  const Tensor& tensor = src_cpu_ortvalue.Get<Tensor>();
  int64_t num_elem = tensor.Shape().Size();
  const T* val_ptr = tensor.template Data<T>();
  output.assign(val_ptr, val_ptr + num_elem);
}

template <typename T>
void CudaOrtValueToCpuVec(const OrtValue& src_cuda_ortvalue, std::vector<T>& output) {
  std::unique_ptr<IExecutionProvider> cuda_provider = onnxruntime::test::DefaultCudaExecutionProvider();
  std::unique_ptr<IExecutionProvider> cpu_provider = onnxruntime::test::DefaultCpuExecutionProvider();

  const Tensor& src_tensor = src_cuda_ortvalue.Get<Tensor>();

  auto allocator = cpu_provider->CreatePreferredAllocators()[0];
  ORT_ENFORCE(allocator, "Cpu allocator is a nullptr.");
  auto dst_tensor = std::make_unique<Tensor>(src_tensor.DataType(), src_tensor.Shape(), allocator);

  auto data_transfer = cuda_provider->GetDataTransfer();
  ORT_ENFORCE(data_transfer, "Cuda data transfer is a nullptr.");

  ORT_THROW_IF_ERROR(data_transfer->CopyTensor(src_tensor, *dst_tensor));

  const T* val_ptr = dst_tensor->template Data<T>();
  output.assign(val_ptr, val_ptr + src_tensor.Shape().Size());
}

inline void GenerateRandomData(std::vector<float>& data) {
  float scale = 1.f;
  float mean = 0.f;
  float seed = 123.f;

  std::default_random_engine generator_float{gsl::narrow_cast<uint32_t>(seed)};
  std::normal_distribution<float> distribution_float{mean, scale};
  std::for_each(data.begin(), data.end(),
                [&generator_float, &distribution_float](float& value) { value = distribution_float(generator_float); });
}

inline void GenerateRandomInput(gsl::span<const int64_t> dims, OrtValue& input) {
  TensorShape shape(dims);
  std::vector<float> data(shape.Size());
  GenerateRandomData(data);
  input = onnxruntime::test::CreateInputOrtValueOnCPU<float>(dims, data);
}

}  // namespace onnxruntime::training::test

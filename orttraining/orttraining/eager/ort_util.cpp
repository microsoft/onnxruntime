// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <core/providers/cpu/cpu_execution_provider.h>

#include "ort_util.h"
#include "ort_backends.h"

namespace torch_ort {
namespace eager {

using namespace onnxruntime;


void CreateMLValue(onnxruntime::AllocatorPtr alloc, 
                   onnxruntime::MLDataType element_type, 
                   const std::vector<int64_t>& dims, 
                   OrtValue* p_mlvalue) {
  onnxruntime::TensorShape shape(dims);
  std::unique_ptr<onnxruntime::Tensor> p_tensor = std::make_unique<onnxruntime::Tensor>(element_type,
                                                                      shape,
                                                                      alloc);
  p_mlvalue->Init(p_tensor.release(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());
}

void CreateMLValue(void* data_ptr, onnxruntime::MLDataType element_type, const std::vector<int64_t>& dims, OrtValue* p_mlvalue) {
  onnxruntime::TensorShape shape(dims);
  OrtMemoryInfo *cpu_info;
  Ort::ThrowOnError(Ort::GetApi().CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &cpu_info));
  std::unique_ptr<onnxruntime::Tensor> p_tensor = std::make_unique<onnxruntime::Tensor>(element_type,
                                                                      shape,
                                                                      data_ptr,
                                                                      *cpu_info);
  
  p_mlvalue->Init(p_tensor.release(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
                  onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());
}

std::vector<int64_t> GetStrides(const std::vector<int64_t>& shape) {
  std::vector<int64_t> strides(shape.size(), 1);
  for (auto i = shape.size(); i > 1; --i) {
    strides[i - 2] = strides[i - 1] * shape[i - 1];
  }
  return strides;
}

} // namespace eager
} // namespace torch_ort
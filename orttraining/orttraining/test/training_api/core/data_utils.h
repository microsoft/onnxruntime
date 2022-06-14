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
void OrtValueToVec(const OrtValue& val, std::vector<T>& output, std::shared_ptr<IExecutionProvider> src_provider, std::shared_ptr<IExecutionProvider> dst_provider) {
  const Tensor& src_tensor = val.Get<Tensor>();

  auto dst_tensor = std::make_unique<Tensor>(src_tensor.DataType(), src_tensor.Shape(), dst_provider->GetAllocator(0, OrtMemTypeDefault));
  auto data_transfer = src_provider->GetDataTransfer();

  ORT_THROW_IF_ERROR(data_transfer->CopyTensor(src_tensor, *dst_tensor, 0));

  const T* val_ptr = dst_tensor->template Data<T>();
  output.assign(val_ptr, val_ptr + src_tensor.Shape().Size());
}

}  // namespace test
}  // namespace training
}  // namespace onnxruntime

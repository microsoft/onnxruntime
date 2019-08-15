// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

namespace Windows::AI::MachineLearning {
class MLValueHelpers {
 public:
  static auto CreateMLValue(onnxruntime::TensorShape shape, onnxruntime::MLDataType data_type, onnxruntime::BufferNakedPtr buffer) {
    auto registrations = onnxruntime::DeviceAllocatorRegistry::Instance().AllRegistrations();
    auto alloc = registrations[onnxruntime::CPU].factory(0);

    // Unowned raw tensor pointer passed to engine
    auto tensor = new onnxruntime::Tensor(
        data_type,
        shape,
        buffer,
        alloc->Info());

    OrtValue value;
    value.Init(tensor,
               onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
               onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());

    return value;
  }

 private:
  MLValueHelpers();
};
}  // namespace Windows::AI::MachineLearning
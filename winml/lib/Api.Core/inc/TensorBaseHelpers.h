// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#pragma once

namespace Windows::AI::MachineLearning {
class TensorBaseHelpers {
 public:
  static auto CreateGPUMLValue(std::shared_ptr<DMLResource>& resource, BindingContext& context, onnxruntime::TensorShape shape, onnxruntime::MLDataType data_type) {
    THROW_HR_IF_NULL(E_INVALIDARG, resource);
    THROW_HR_IF_NULL(E_UNEXPECTED, resource->ExecutionProviderAllocatedResource);

    auto session_impl = context.session.as<winrt::Windows::AI::MachineLearning::implementation::LearningModelSession>();
    auto provider = session_impl->GetExecutionProvider();

    WINML_THROW_HR_IF_TRUE_MSG(WINML_ERR_INVALID_BINDING,
                               "DmlExecutionProvider" != provider->Type(),
                               "Cannot creat GPU tensor on CPU device");

    auto tensor = new onnxruntime::Tensor(
        data_type,
        shape,
        resource->ExecutionProviderAllocatedResource,
        provider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault)->Info());

    OrtValue ml_value;
    ml_value.Init(tensor,
                 onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
                 onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());

    return ml_value;
  }

 private:
  TensorBaseHelpers();
};
}  // namespace Windows::AI::MachineLearning
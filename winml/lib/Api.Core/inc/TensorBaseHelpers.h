#pragma once

namespace Windows::AI::MachineLearning
{
    class TensorBaseHelpers
    {
    public:
        static auto CreateGPUMLValue(std::shared_ptr<DMLResource>& resource, BindingContext& context, onnxruntime::TensorShape shape, onnxruntime::MLDataType dataType)
        {
            THROW_HR_IF_NULL(E_INVALIDARG, resource);
            THROW_HR_IF_NULL(E_UNEXPECTED, resource->ExecutionProviderAllocatedResource);

            auto sessionImpl = context.Session.as<winrt::Windows::AI::MachineLearning::implementation::LearningModelSession>();
            auto pProvider = sessionImpl->GetExecutionProvider();

            WINML_THROW_HR_IF_TRUE_MSG(WINML_ERR_INVALID_BINDING,
                "DmlExecutionProvider" != pProvider->Type(),
                "Cannot creat GPU tensor on CPU device");

            auto pTensor = new onnxruntime::Tensor(
                dataType,
                shape,
                resource->ExecutionProviderAllocatedResource,
                pProvider->GetAllocator(0, ::OrtMemType::OrtMemTypeDefault)->Info());

            OrtValue mlValue;
            mlValue.Init(pTensor,
                onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
                onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());

            return mlValue;
        }

    private:
        TensorBaseHelpers();
    };
}
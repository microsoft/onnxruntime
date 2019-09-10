#pragma once

namespace Windows::AI::MachineLearning
{
    class MLValueHelpers
    {
    public:
        static auto CreateMLValue(onnxruntime::TensorShape shape, onnxruntime::MLDataType dataType, onnxruntime::BufferNakedPtr buffer)
        {
            auto registrations = onnxruntime::DeviceAllocatorRegistry::Instance().AllRegistrations();
            auto pAlloc = registrations[onnxruntime::CPU].factory(0);

            // Unowned raw tensor pointer passed to engine
            auto pTensor = new onnxruntime::Tensor(
                dataType,
                shape,
                buffer,
                pAlloc->Info());

            OrtValue value;
            value.Init(pTensor,
                onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>(),
                onnxruntime::DataTypeImpl::GetType<onnxruntime::Tensor>()->GetDeleteFunc());

            return value;
        }

    private:
        MLValueHelpers();
    };
}
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "core/providers/dml/DmlExecutionProvider/src/MLOperatorAuthorImpl.h"
#include "core/providers/dml/DmlExecutionProvider/src/ExecutionProvider.h"

using Windows::AI::MachineLearning::Adapter::TensorWrapper;

namespace Dml
{

class DmlOperatorLayerNormalization : public DmlOperator
{
public:
    DmlOperatorLayerNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1, 2};

        // Initialize Input, Scale and Bias tensors with same dimension count as Input tensor
        // because DML MVN1 has a validation which requires all 3 needs to have same dimension count
        // due to historical artifact.
        DmlOperator::Initialize(
            kernelCreationContext,
            kernelInputIndices,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            kernelCreationContext.GetTensorShapeDescription().GetInputTensorDimensionCount(0));

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);

        int32_t onnxAxis = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::Axis, -1);
        uint32_t inputDimCount = kernelCreationContext.GetTensorShapeDescription().GetInputTensorDimensionCount(0);
        onnxAxis = OperatorHelper::HandleNegativeAxis(onnxAxis, inputDimCount);
        std::vector<uint32_t> onnxAxes(inputDimCount - onnxAxis);
        std::iota(onnxAxes.begin(), onnxAxes.end(), onnxAxis);

        m_castedInputDescs.insert(m_castedInputDescs.end(), m_inputTensorDescs.begin(), m_inputTensorDescs.end());
        m_inputCastOps.resize(m_inputTensorDescs.size());
        m_castedOutputTensorDesc = m_outputTensorDescs[0];

        auto inputDataType = m_inputTensorDescs[0].GetDmlDataType();
        assert(inputDataType == DML_TENSOR_DATA_TYPE_FLOAT16 || inputDataType == DML_TENSOR_DATA_TYPE_FLOAT32);

        auto scaleDataType = m_inputTensorDescs[1].GetDmlDataType();
        assert(scaleDataType == DML_TENSOR_DATA_TYPE_FLOAT16 || scaleDataType == DML_TENSOR_DATA_TYPE_FLOAT32);

        // Scale and Bias always have the same data type
        assert(m_inputTensorDescs[2].GetDmlDataType() == DML_TENSOR_TYPE_INVALID || m_inputTensorDescs[2].GetDmlDataType() == scaleDataType);

        // Cast all tensors to the highest common precision
        if (inputDataType == DML_TENSOR_DATA_TYPE_FLOAT16 && scaleDataType == DML_TENSOR_DATA_TYPE_FLOAT32)
        {
            m_castedInputDescs[0] = TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_inputTensorDescs[0].GetSizes());
            m_inputCastOps[0] = InitializeCast(m_inputTensorDescs[0], m_castedInputDescs[0]);
        }
        else if (inputDataType == DML_TENSOR_DATA_TYPE_FLOAT32 && scaleDataType == DML_TENSOR_DATA_TYPE_FLOAT16)
        {
            m_castedInputDescs[1] = TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_inputTensorDescs[1].GetSizes());
            m_inputCastOps[1] = InitializeCast(m_inputTensorDescs[1], m_castedInputDescs[1]);

            if (m_inputTensorDescs[2].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
            {
                m_castedInputDescs[2] = TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_inputTensorDescs[2].GetSizes());
                m_inputCastOps[2] = InitializeCast(m_inputTensorDescs[2], m_castedInputDescs[2]);
            }
        }

        if (m_castedInputDescs[0].GetDmlDataType() != m_outputTensorDescs[0].GetDmlDataType())
        {
            // After the operator has been executed, we need to cast the "casted" output tensor to the original output tensor that TF expects
            m_castedOutputTensorDesc = TensorDesc(m_castedInputDescs[0].GetDmlDataType(), m_outputTensorDescs[0].GetSizes());
            m_outputCast = InitializeCast(m_castedOutputTensorDesc, m_outputTensorDescs[0]);
        }

        auto inputDesc = m_castedInputDescs[0].GetDmlDesc();
        auto scaleDesc = m_castedInputDescs[1].GetDmlDesc();
        auto biasDesc = m_castedInputDescs[2].GetDmlDesc();
        auto outputDesc = m_castedOutputTensorDesc.GetDmlDesc();

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDesc;
        operatorDesc.ScaleTensor = &scaleDesc;
        operatorDesc.BiasTensor = biasDesc.Desc != nullptr ? &biasDesc : nullptr;
        operatorDesc.OutputTensor = &outputDesc;
        operatorDesc.Axes = onnxAxes.data();
        operatorDesc.AxisCount = gsl::narrow_cast<uint32_t>(onnxAxes.size());
        operatorDesc.NormalizeVariance = true;
        operatorDesc.Epsilon = epsilon;
        operatorDesc.FusedActivation = nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }

    void Compute(const MLOperatorKernelContext& kernelContext) override
    {
        ExecutionProviderImpl* executionProvider = static_cast<ExecutionProviderImpl*>(m_executionProvider.Get());
        std::vector<IMLOperatorTensor*> originalInputTensors = GetInputTensorsForExecute(kernelContext);
        std::vector<IMLOperatorTensor*> originalOutputTensors = GetOutputTensorsForExecute(kernelContext);

        std::array<onnxruntime::Tensor, 3> inputOrtTensors;
        std::array<onnxruntime::Tensor, 1> outputOrtTensors;
        std::array<ComPtr<IMLOperatorTensor>, 3> castedInputTensorWrappers;
        std::array<ComPtr<IMLOperatorTensor>, 1> castedOutputTensorWrappers;
        std::array<IMLOperatorTensor*, 3> castedInputTensors;
        std::array<IMLOperatorTensor*, 1> castedOutputTensors;

        assert(m_castedInputDescs.size() == m_inputCastOps.size());
        for (size_t i = 0; i < m_castedInputDescs.size(); ++i)
        {
            const TensorDesc& inputDesc = m_castedInputDescs[i];

            if (m_inputCastOps[i])
            {
                std::vector<int64_t> inputSizes(inputDesc.GetSizes().size());
                for (size_t i = 0; i < inputSizes.size(); ++i)
                {
                    inputSizes[i] = inputDesc.GetSizes()[i];
                }

                auto dataType = Windows::AI::MachineLearning::Adapter::ToTensorDataType(GetMlDataTypeFromDmlDataType(inputDesc.GetDmlDataType()));
                inputOrtTensors[i] = onnxruntime::Tensor(dataType, onnxruntime::TensorShape(inputSizes), executionProvider->GetGpuAllocator());
                castedInputTensorWrappers[i] = wil::MakeOrThrow<TensorWrapper>(
                    &inputOrtTensors[i],
                    true,
                    executionProvider,
                    true);

                castedInputTensors[i] = castedInputTensorWrappers[i].Get();

                IMLOperatorTensor* inputTensors[] = {originalInputTensors[i]};
                IMLOperatorTensor* outputTensors[] = {castedInputTensorWrappers[i].Get()};
                ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
                    m_inputCastOps[i].Get(),
                    nullptr,
                    inputTensors,
                    outputTensors));
            }
            else
            {
                castedInputTensors[i] = originalInputTensors[i];
            }
        }

        if (m_outputCast)
        {
            std::vector<int64_t> outputSizes(m_castedOutputTensorDesc.GetSizes().size());
            for (size_t i = 0; i < outputSizes.size(); ++i)
            {
                outputSizes[i] = m_castedOutputTensorDesc.GetSizes()[i];
            }

            auto dataType = Windows::AI::MachineLearning::Adapter::ToTensorDataType(GetMlDataTypeFromDmlDataType(m_castedOutputTensorDesc.GetDmlDataType()));
            outputOrtTensors[0] = onnxruntime::Tensor(dataType, onnxruntime::TensorShape(outputSizes), executionProvider->GetGpuAllocator());
            castedOutputTensorWrappers[0] = wil::MakeOrThrow<TensorWrapper>(
                &outputOrtTensors[0],
                true,
                executionProvider,
                true);
            castedOutputTensors[0] = castedOutputTensorWrappers[0].Get();
        }
        else
        {
            castedOutputTensors[0] = originalOutputTensors[0];
        }

        ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            castedInputTensors,
            castedOutputTensors));

        // Finally, we can cast the tensor back to the original desired data type
        if (m_outputCast)
        {
            ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
                m_outputCast.Get(),
                nullptr,
                castedOutputTensors,
                originalOutputTensors));
        }
    }

private:
    std::vector<TensorDesc> m_castedInputDescs;
    std::vector<ComPtr<IDMLCompiledOperator>> m_inputCastOps;
    TensorDesc m_castedOutputTensorDesc;
    ComPtr<IDMLCompiledOperator> m_outputCast;
};

void CALLBACK QueryLayerNormalization(IMLOperatorSupportQueryContextPrivate* context, /*out*/ bool* isSupported)
{
    *isSupported = false;

    // Mean and InvStdDev are not supported outputs.
    // If only Scale tensor is present then fall back to CPU. This is temporary until
    // DML1.9.2 or DML1.10 gets released.
    if (context->GetInputCount() < 3 || context->GetOutputCount() > 1)
    {
        return;
    }

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(LayerNormalization, DmlOperatorLayerNormalization);
DML_OP_DEFINE_CREATION_FUNCTION(LayerNormalization17, DmlOperatorLayerNormalization);

} // namespace Dml

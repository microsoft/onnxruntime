// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

template <bool gpuOutput>
class DmlOperatorMemcpy : public DmlOperator
{
public:
    using Self = DmlOperatorMemcpy;

    DmlOperatorMemcpy(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1, "MemcpyFromHost/ToHost expects 1 input tensor.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "MemcpyFromHost/ToHost expects 1 output tensor.");

        Initialize(kernelCreationContext);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC opDesc = {};
        opDesc.InputTensor = inputDescs.data();
        opDesc.OutputTensor = outputDescs.data();

        SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &opDesc }, kernelCreationContext);
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensors(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensors(kernelContext);

        if (kernelContext.IsSequenceInputTensor(0))
        {
            auto dataType = kernelContext.GetSequenceInputDataType(0);
            kernelContext.PrepareSequenceOutput(0, dataType);

            const uint32_t numTensors = kernelContext.GetSequenceInputCount(0);
            inputTensors.reserve(numTensors);

            for (uint32_t sequenceIndex = 0; sequenceIndex < numTensors; ++sequenceIndex)
            {
                auto* inputTensor = kernelContext.GetSequenceInputTensor(0, sequenceIndex).GetInterface().Get();
                const uint32_t dimCount = inputTensor->GetDimensionCount();

                std::vector<uint32_t> dimensions(dimCount);
                inputTensor->GetShape(dimCount, dimensions.data());

                inputTensors.push_back(inputTensor);
                outputTensors.push_back(kernelContext.GetSequenceOutputTensor(
                    0,
                    sequenceIndex,
                    inputTensor->GetTensorDataType(),
                    dimCount,
                    dimensions.data(),
                    gpuOutput).GetInterface().Get());
            }
        }
        else
        {
            inputTensors = { kernelContext.GetInputTensor(0).GetInterface().Get() };
            outputTensors = { kernelContext.GetOutputTensor(0).GetInterface().Get() };
        }

        ORT_THROW_IF_FAILED(m_executionProvider->CopyTensors(outputTensors, inputTensors));
    }

private:
};

// MemcpyToHost is a special case which is hardcoded in MLOperatorAuthorImpl.cpp. If name changes this must be updated.
// Special case makes sure that the output resource is created using the CPU allocator.
DML_OP_DEFINE_CREATION_FUNCTION(MemcpyFromHost, DmlOperatorMemcpy<true>);
DML_OP_DEFINE_CREATION_FUNCTION(MemcpyToHost, DmlOperatorMemcpy<false>);

} // namespace Dml

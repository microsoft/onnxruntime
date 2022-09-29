// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorMemcpy : public DmlOperator
{
public:
    using Self = DmlOperatorMemcpy;

    DmlOperatorMemcpy(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1, "MemcpyFromHost/ToHost expects 1 input tensor.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "MemcpyFromHost/ToHost expects 1 output tensor.");

        DmlOperator::Initialize(kernelCreationContext);
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensors(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensors(kernelContext);
        assert(inputTensors.size() == 1);
        assert(outputTensors.size() == 1);

        if (!OperatorHelper::ContainsEmptyDimensions(MLOperatorTensor(inputTensors.front()).GetShape()))
        {
            ORT_THROW_IF_FAILED(m_executionProvider->CopyTensor(
                outputTensors.front(),
                inputTensors.front()
                ));
        }
    }

private:
};

// MemcpyToHost is a special case which is hardcoded in MLOperatorAuthorImpl.cpp. If name changes this must be updated.
// Special case makes sure that the output resource is created using the CPU allocator.
DML_OP_DEFINE_CREATION_FUNCTION(MemcpyFromHost, DmlOperatorMemcpy);
DML_OP_DEFINE_CREATION_FUNCTION(MemcpyToHost, DmlOperatorMemcpy);

} // namespace Dml

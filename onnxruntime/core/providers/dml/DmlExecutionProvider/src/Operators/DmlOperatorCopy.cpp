// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
// Copies first input and ignores others.  Used for operators which perform reshaping.
class DmlOperatorCopy : public DmlOperator
{
public:
    using Self = DmlOperatorCopy;

    DmlOperatorCopy(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        std::vector<std::optional<uint32_t>> kernelInputOutputIndices = {0};

        Initialize(kernelInfo, kernelInputOutputIndices);

        // DirectML requires the input & output dimensions to be identical, even if the
        // element counts are the same. All this operator does is copy the resource and
        // rearrange the dimensions, so we tell DML that the output dimensions are the
        // same as the input dimensions.
        m_outputTensorDescs.front() = m_inputTensorDescs.front();

        ComPtr<IMLOperatorKernelCreationContextPrivate> contextPrivate;
        ORT_THROW_IF_FAILED(kernelInfo.GetInterface()->QueryInterface(contextPrivate.GetAddressOf()));

        // Although we always compile the operator because we don't know where the memory will be allocated in the future,
        // we may not always end up executing it.
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC opDesc = {};
        opDesc.InputTensor = inputDescs.data();
        opDesc.OutputTensor = outputDescs.data();

        SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &opDesc }, kernelInfo);
    }

    void Compute(const MLOperatorKernelContext& kernelContext) final
    {
        MLOperatorTensor inputTensor = kernelContext.GetInputTensor(0);
        MLOperatorTensor outputTensor = kernelContext.GetOutputTensor(0);

        // If the input is aliasing the output (i.e. they share the same resource at the same offset),
        // we don't need to do anything. This is essentially a no-op.
        if (inputTensor.GetByteData() == outputTensor.GetByteData())
        {
            return;
        }

        // If the input is not aliasing the output but shares the same resource, we have to use an Identity operation
        // because the resource cannot simultaneously be in both the COPY_SOURCE and COPY_DEST states.
        if (inputTensor.GetDataInterface().Get() == outputTensor.GetDataInterface().Get())
        {
            DmlOperator::Compute(kernelContext);
        }
        else
        {
            // The input and the output don't share the same resource, so we can do a simple copy.
            ORT_THROW_IF_FAILED(m_executionProvider->CopyTensor(
                outputTensor.GetInterface().Get(),
                inputTensor.GetInterface().Get()));
        }
    }

private:
    // Aliasing means that both the input and the output start at the same exact offset in the same buffer
    bool m_aliasing = false;

    // The choice of using Identity or a copy depends on whether the input and the input are located in the same buffer
    bool m_inputSharesOutputBuffer = false;
};

DML_OP_DEFINE_CREATION_FUNCTION(Copy, DmlOperatorCopy);

} // namespace Dml

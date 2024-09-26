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

        std::vector<std::optional<uint32_t>> kernelInputOutputIndices  = {0};

        Initialize(kernelInfo, kernelInputOutputIndices);

        // DirectML requires the input & output dimensions to be identical, even if the
        // element counts are the same. All this operator does is copy the resource and
        // rearrange the dimensions, so we tell DML that the output dimensions are the
        // same as the input dimensions.
        m_outputTensorDescs.front() = m_inputTensorDescs.front();

        ComPtr<IMLOperatorKernelCreationContextPrivate> contextPrivate;
        ORT_THROW_IF_FAILED(kernelInfo.GetInterface()->QueryInterface(contextPrivate.GetAddressOf()));

        if (kernelInfo.GetNodeWrapperInterface())
        {
            ORT_THROW_IF_FAILED(kernelInfo.GetNodeWrapperInterface()->GetUtf8Name(sizeof(m_nodeName), m_nodeName));
        }
        else
        {
            m_nodeName[0] = '\0';
        }

        if (contextPrivate->IsDmlGraphNode())
        {
            std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
            std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC opDesc = {};
            opDesc.InputTensor = inputDescs.data();
            opDesc.OutputTensor = outputDescs.data();

            SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &opDesc }, kernelInfo);
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        MLOperatorTensor inputTensor = kernelContext.GetInputTensor(0);

        // Reshape the output tensor.
        MLOperatorTensor outputTensor = kernelContext.GetOutputTensor(0);

        // Avoid self copying.
        if (inputTensor.GetDataInterface().Get() != outputTensor.GetDataInterface().Get())
        {
            // Copy elements from input tensor to output tensor.
            ORT_THROW_IF_FAILED(m_executionProvider->CopyTensor(
                outputTensor.GetInterface().Get(),
                inputTensor.GetInterface().Get(),
                m_nodeName));
        }
    }

private:
    // Static buffer (might truncate name) to avoid excessive dynamic allocation only for debugging purposes.
    char m_nodeName[512];
};

DML_OP_DEFINE_CREATION_FUNCTION(Copy, DmlOperatorCopy);

} // namespace Dml

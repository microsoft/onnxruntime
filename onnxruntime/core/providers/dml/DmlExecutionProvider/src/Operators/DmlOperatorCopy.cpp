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

        // We don't need to compile any operator if the input aliases the output as it is essentially a no-op
        // (e.g. squeeze/unsqueeze/reshape). An exception to this rule is when the operator is part of the graph,
        // in which case we always need to compile and execute the operator (although this is something that we
        // could optimize in the future).
        if (!contextPrivate->IsDmlGraphNode())
        {
            std::vector<uint32_t> outputSizes = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);
            std::vector<int64_t> outputSizesInt64(outputSizes.begin(), outputSizes.end());
            onnxruntime::TensorShape outputShape(outputSizesInt64);
            ORT_THROW_IF_FAILED(kernelInfo.GetNodeWrapperInterface()->InputAliasesOutput(0, 0, outputShape, &m_aliasing));
            ORT_THROW_IF_FAILED(kernelInfo.GetNodeWrapperInterface()->InputSharesOutputBuffer(0, 0, outputShape, &m_inputSharesOutputBuffer));
        }

        if (contextPrivate->IsDmlGraphNode() || (!m_aliasing && m_inputSharesOutputBuffer))
        {
            std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
            std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC opDesc = {};
            opDesc.InputTensor = inputDescs.data();
            opDesc.OutputTensor = outputDescs.data();

            SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &opDesc }, kernelInfo);
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext) final
    {
        // If the input is aliasing the output, we don't need to do anything here
        if (m_aliasing)
        {
            return;
        }

        // If the input and the output share the same buffer, we need to do an identity operation
        if (m_inputSharesOutputBuffer)
        {
            DmlOperator::Compute(kernelContext);
            return;
        }

        // If the input and the output don't share the same buffer, we can do a standard copy operation instead
        MLOperatorTensor inputTensor = kernelContext.GetInputTensor(0);
        MLOperatorTensor outputTensor = kernelContext.GetOutputTensor(0);

        ORT_THROW_IF_FAILED(m_executionProvider->CopyTensor(
            outputTensor.GetInterface().Get(),
            inputTensor.GetInterface().Get()));
    }

private:
    // Aliasing means that both the input and the output start at the same exact offset in the same buffer
    bool m_aliasing = false;

    // The choice of using Identity or a copy depends on whether the input and the input are located in the same buffer
    bool m_inputSharesOutputBuffer = false;
};

DML_OP_DEFINE_CREATION_FUNCTION(Copy, DmlOperatorCopy);

} // namespace Dml

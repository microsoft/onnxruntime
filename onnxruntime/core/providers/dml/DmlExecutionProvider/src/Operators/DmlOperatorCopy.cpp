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

        bool aliasing = false;

        if (!contextPrivate->IsDmlGraphNode())
        {
            std::vector<uint32_t> outputSizes = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);
            std::vector<int64_t> outputSizesInt64(outputSizes.begin(), outputSizes.end());
            onnxruntime::TensorShape outputShape(outputSizesInt64);
            ORT_THROW_IF_FAILED(kernelInfo.GetNodeWrapperInterface()->InputAliasesOutput(0, 0, outputShape, &aliasing));
        }

        if (!aliasing)
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
        if (inputTensor.GetByteData() != outputTensor.GetByteData())
        {
            // Copy elements from input tensor to output tensor.
            ORT_THROW_IF_FAILED(m_executionProvider->CopyTensor(
                outputTensor.GetInterface().Get(),
                inputTensor.GetInterface().Get()));
        }
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Copy, DmlOperatorCopy);

} // namespace Dml

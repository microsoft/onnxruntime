// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorShape : public DmlOperator, ShapeHelper
{
public:
    DmlOperatorShape(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        ShapeHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1, "Shape expects 1 input tensor.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "Shape expects 1 output tensor.");

        m_startIndex = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::Start, 0);

        if (m_startIndex != 0)
        {
            // "start" is provided and is non-default (default is 0)
            m_needsSlicing = true;
        }

        if (kernelCreationContext.HasAttribute(AttrName::End, MLOperatorAttributeType::Int))
        {
            m_endIndex = kernelCreationContext.GetAttribute<int64_t>(AttrName::End);
            m_needsSlicing = true;
        }

        DmlOperator::Initialize(kernelCreationContext);
    }

    // Takes a tensor as input and outputs an 1D int64 tensor
    // containing the shape of the input tensor.
    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensors(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensors(kernelContext);
        assert(inputTensors.size() == 1);
        assert(outputTensors.size() == 1);

        IMLOperatorTensor *inputTensor = inputTensors[0];
        IMLOperatorTensor *outputTensor = outputTensors[0];
        assert(outputTensor->GetDimensionCount() == 1);

        uint32_t inputDimCount = inputTensor->GetDimensionCount();
        uint32_t outputSize;
        THROW_IF_FAILED(outputTensor->GetShape(1, &outputSize));
        assert(outputSize == inputDimCount);

        // Get output shape as a vector
        std::vector<uint32_t> outputShape;
        if (!m_needsSlicing)
        {
            outputShape.resize(inputDimCount);
            THROW_IF_FAILED(inputTensor->GetShape(inputDimCount, outputShape.data()));
        }
        else
        {
            int64_t trueStart = m_startIndex;
            int64_t trueEnd = m_endIndex;

            // Deal with negative(s) and clamp
            trueStart = trueStart < 0 ? trueStart + inputDimCount : trueStart;
            trueStart = trueStart < 0 ? 0 : ((trueStart > inputDimCount) ? inputDimCount : trueStart);

            trueEnd = trueEnd < 0 ? trueEnd + inputDimCount : trueEnd;
            trueEnd = trueEnd < 0 ? 0 : ((trueEnd > inputDimCount) ? inputDimCount : trueEnd);

            int64_t sliceLength = trueEnd - trueStart;
            outputShape.resize(sliceLength);

            if (sliceLength > 0)
            {
                THROW_IF_FAILED(inputTensor->GetShape(static_cast<uint32_t>(sliceLength), outputShape.data() + static_cast<uint32_t>(trueStart)));
            }
        }

        int64_t* outputData = reinterpret_cast<int64_t*>(outputTensor->GetData());

        // Write input shape to output data
        for (uint32_t i = 0U; i < outputShape.size(); ++i)
        {
            outputData[i] = (int64_t)outputShape[i];
        }
    }

private:
    bool m_needsSlicing = false;
    int64_t m_startIndex = 0;
    int64_t m_endIndex = std::numeric_limits<int64_t>::max();
};

// Shape is a special case which is hardcoded in MLOperatorAuthorImpl.cpp. If name changes this must be updated.
// Special case makes sure that the input/output resource is created using the CPU allocator.
DML_OP_DEFINE_CREATION_FUNCTION(Shape, DmlOperatorShape);

} // namespace Dml

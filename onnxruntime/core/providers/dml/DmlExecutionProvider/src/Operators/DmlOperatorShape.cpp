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
        DmlOperator::Initialize(kernelCreationContext);
    }

    // Takes a tensor as input and outputs a 1D int64 tensor containing the shape of the input tensor.
    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensors(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensors(kernelContext);
        assert(inputTensors.size() == 1);
        assert(outputTensors.size() == 1);

        const IMLOperatorTensor* inputTensor = inputTensors[0];
        IMLOperatorTensor* outputTensor = outputTensors[0];

        const uint32_t inputDimCount = inputTensor->GetDimensionCount();
        std::vector<uint32_t> inputShape(inputDimCount);
        THROW_IF_FAILED(inputTensor->GetShape(inputDimCount, inputShape.data()));

        assert(m_sliceEnd >= m_sliceStart);
        std::vector<uint32_t> outputShape(inputShape.begin() + m_sliceStart, inputShape.begin() + m_sliceEnd);

        ML_CHECK_VALID_ARGUMENT(outputTensor->IsCpuData(), "Output must be a CPU tensor.");
        int64_t* outputData = reinterpret_cast<int64_t*>(outputTensor->GetData());

        // Write input shape to output data
        for (uint32_t i = 0U; i < outputShape.size(); ++i)
        {
            outputData[i] = static_cast<int64_t>(outputShape[i]);
        }
    }
};

// Shape is a special case which is hardcoded in AbiCustomRegistry.cpp. If name changes this must be updated.
// Special case makes sure that the input/output resource is created using the CPU allocator.
DML_OP_DEFINE_CREATION_FUNCTION(Shape, DmlOperatorShape);

} // namespace Dml

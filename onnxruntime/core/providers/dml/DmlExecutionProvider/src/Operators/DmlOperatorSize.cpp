// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorSize : public DmlOperator, SizeHelper
{
public:
    DmlOperatorSize(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        SizeHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1, "Size expects 1 input tensor.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "Size expects 1 output tensor.");
        DmlOperator::Initialize(kernelCreationContext);
    }

    // Takes a tensor as input and outputs a scalar int64 tensor containing the size of the input tensor.
    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensors(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensors(kernelContext);
        assert(inputTensors.size() == 1);
        assert(outputTensors.size() == 1);

        // Get input shape as a vector and compute the number of elements
        const IMLOperatorTensor* inputTensor = inputTensors[0];
        const uint32_t inputDimCount = inputTensor->GetDimensionCount();
        std::vector<uint32_t> inputShape(inputDimCount);
        THROW_IF_FAILED(inputTensor->GetShape(inputDimCount, inputShape.data()));
        uint32_t numElements = ComputeElementCountFromDimensions(inputShape);

        // Write the number of elements to output data
        IMLOperatorTensor* outputTensor = outputTensors[0];

        ML_CHECK_VALID_ARGUMENT(outputTensor->IsCpuData(), "Output must be a CPU tensor.");
        int64_t* outputData = reinterpret_cast<int64_t*>(outputTensor->GetData());
        outputData[0] = static_cast<uint64_t>(numElements);
    }
};

// Size is a special case which is hardcoded in AbiCustomRegistry.cpp. If name changes this must be updated.
// Special case makes sure that the input/output resource is created using the CPU allocator.
DML_OP_DEFINE_CREATION_FUNCTION(Size, DmlOperatorSize);

} // namespace Dml

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorTranspose : public DmlOperator, public TransposeHelper
{
public:
    using Self = DmlOperatorTranspose;

    DmlOperatorTranspose(const MLOperatorKernelCreationContext& kernelInfo)
        :   DmlOperator(kernelInfo),
            TransposeHelper(kernelInfo, kernelInfo.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() >= 1);
        DmlOperator::Initialize(kernelInfo);

        const std::vector<uint32_t> originalSizes = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(m_permutations.size() == originalSizes.size());

        // Calculate strides from original shape.
        ML_CHECK_VALID_ARGUMENT(!originalSizes.empty());
        std::vector<uint32_t> inputStrides(originalSizes.size());
        Dml::GetDescendingPackedStrides(originalSizes, /*out*/ inputStrides);

        std::vector<uint32_t> sizes(inputStrides.size());
        std::vector<uint32_t> strides(inputStrides.size());

        // Permute the shape and strides.
        for (int dimInput = 0, dimCount = gsl::narrow_cast<int>(originalSizes.size()); dimInput < dimCount; ++dimInput)
        {
            int dimPermuted = m_permutations[dimInput];

            ML_CHECK_VALID_ARGUMENT(gsl::narrow_cast<size_t>(dimPermuted) < originalSizes.size());
            sizes[dimInput] = originalSizes[dimPermuted];
            strides[dimInput] = inputStrides[dimPermuted];
        }

        // Override the initial tensor descs. The output tensor is not strided.
        m_inputTensorDescs.front() = TensorDesc(m_inputTensorDescs.front().GetDmlDataType(), sizes, strides, 0);
        m_outputTensorDescs.front() = TensorDesc(m_inputTensorDescs.front().GetDmlDataType(), sizes, std::nullopt);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC opDesc = {};
        opDesc.InputTensor = inputDescs.data();
        opDesc.OutputTensor = outputDescs.data();

        SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &opDesc}, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Transpose,  DmlOperatorTranspose);

} // namespace Dml

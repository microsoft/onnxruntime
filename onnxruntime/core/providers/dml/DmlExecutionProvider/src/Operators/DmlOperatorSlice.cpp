// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorSlice : public DmlOperator, public SliceHelper
{
public:
    DmlOperatorSlice(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo),
        SliceHelper(kernelInfo, kernelInfo.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);
        DmlOperator::Initialize(kernelInfo);

        assert(m_inputTensorDescs[0].GetDimensionCount() >= gsl::narrow_cast<uint32_t>(m_offsets.size()));
        assert(m_inputTensorDescs[0].GetDimensionCount() >= gsl::narrow_cast<uint32_t>(m_sizes.size()));
        assert(m_inputTensorDescs[0].GetDimensionCount() >= gsl::narrow_cast<uint32_t>(m_strides.size()));

        // Pad the parameters to respect DML's requirements
        m_offsets.insert(
            m_offsets.begin(),
            m_inputTensorDescs[0].GetDimensionCount() - gsl::narrow_cast<uint32_t>(m_offsets.size()),
            0);

        m_sizes.insert(
            m_sizes.begin(),
            m_inputTensorDescs[0].GetDimensionCount() - gsl::narrow_cast<uint32_t>(m_sizes.size()),
            1);

        m_strides.insert(
            m_strides.begin(),
            m_inputTensorDescs[0].GetDimensionCount() - gsl::narrow_cast<uint32_t>(m_strides.size()),
            1);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_SLICE_OPERATOR_DESC sliceDesc = {};
        sliceDesc.InputTensor = inputDescs.data();
        sliceDesc.OutputTensor = outputDescs.data();
        sliceDesc.DimensionCount = gsl::narrow_cast<uint32_t>(m_offsets.size());
        sliceDesc.Offsets = m_offsets.data();
        sliceDesc.Sizes = m_sizes.data();
        sliceDesc.Strides = m_strides.data();

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_SLICE, &sliceDesc };
        
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Slice, DmlOperatorSlice);

} // namespace Dml

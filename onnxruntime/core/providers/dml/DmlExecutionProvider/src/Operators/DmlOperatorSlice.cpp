// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorSlice : public DmlOperator, public SliceHelperBase
{
public:
    DmlOperatorSlice(const MLOperatorKernelCreationContext& kernelInfo, uint32_t opsetVersion)
    :   DmlOperator(kernelInfo),
        SliceHelperBase(kernelInfo, kernelInfo.GetTensorShapeDescription(), opsetVersion)
    {
        uint32_t minInputCount = (opsetVersion < 10) ? 1 : 3;
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= minInputCount);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        // TODO (23108599): Slice V10 introduces an optional "Steps" input which the kernel does not yet support.
        THROW_HR_IF(E_NOTIMPL, kernelInfo.GetInputCount() > 4);

        std::vector<std::optional<uint32_t>> kernelInputIndices = { 0 };
        DmlOperator::Initialize(kernelInfo, kernelInputIndices);

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

// A specific type of operation for registration.
template <uint32_t opsetVersion>
class DmlOperatorSliceTemplate : public DmlOperatorSlice
{
public:
    DmlOperatorSliceTemplate(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperatorSlice(kernelInfo, opsetVersion)
    {
    }
};

void QuerySlice(IMLOperatorSupportQueryContextPrivate* context, bool *isSupported)
{
    *isSupported = (context->GetInputCount() <= 4);
}

DML_OP_DEFINE_CREATION_FUNCTION(Slice7,  DmlOperatorSliceTemplate<7>);
DML_OP_DEFINE_CREATION_FUNCTION(Slice10, DmlOperatorSliceTemplate<10>);
} // namespace Dml

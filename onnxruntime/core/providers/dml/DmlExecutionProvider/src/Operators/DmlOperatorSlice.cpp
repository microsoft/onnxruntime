// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorSlice : public DmlOperator, public SliceHelper
{
public:
    DmlOperatorSlice(const MLOperatorKernelCreationContext& kernelInfo, uint32_t opsetVersion)
    :   DmlOperator(kernelInfo),
        SliceHelper(kernelInfo, kernelInfo.GetTensorShapeDescription(), opsetVersion)
    {
        const uint32_t inputCount = kernelInfo.GetInputCount();
        ML_CHECK_VALID_ARGUMENT((opsetVersion <  10 && inputCount == 1)
                             || (opsetVersion >= 10 && inputCount >= 3 && inputCount <= 5));
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        std::vector<std::optional<uint32_t>> kernelInputIndices = { 0 }; // Only bind GPU to first 'data' tensor.
        DmlOperator::Initialize(kernelInfo, kernelInputIndices, std::nullopt, std::nullopt, std::nullopt, /*minimumDimensionCount*/ 1);

        const uint32_t inputTensorRank = m_inputTensorDescs[0].GetDimensionCount();
        assert(inputTensorRank >= gsl::narrow_cast<uint32_t>(m_offsets.size()));
        assert(inputTensorRank >= gsl::narrow_cast<uint32_t>(m_sizes.size()));
        assert(inputTensorRank >= gsl::narrow_cast<uint32_t>(m_strides.size()));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_SLICE1_OPERATOR_DESC sliceDesc = {};
        sliceDesc.InputTensor = inputDescs.data();
        sliceDesc.OutputTensor = outputDescs.data();
        sliceDesc.DimensionCount = gsl::narrow_cast<uint32_t>(m_offsets.size());
        sliceDesc.InputWindowOffsets = m_offsets.data();
        sliceDesc.InputWindowSizes = m_sizes.data();
        sliceDesc.InputWindowStrides = m_strides.data();

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_SLICE1, &sliceDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

void CALLBACK QuerySlice(IMLOperatorSupportQueryContextPrivate* context, bool* isSupported)
{
    *isSupported = (context->GetInputCount() <= 5);
}

DML_OP_DEFINE_CREATION_FUNCTION(Slice7,  VersionedKernel<DmlOperatorSlice, 7> );
DML_OP_DEFINE_CREATION_FUNCTION(Slice10, VersionedKernel<DmlOperatorSlice, 10>);
DML_OP_DEFINE_CREATION_FUNCTION(Slice11, VersionedKernel<DmlOperatorSlice, 11>);
DML_OP_DEFINE_CREATION_FUNCTION(Slice13, VersionedKernel<DmlOperatorSlice, 13>);
} // namespace Dml

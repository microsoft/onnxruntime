// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorConcat : public DmlOperator, public ConcatHelper
{
public:
    using Self = DmlOperatorConcat;

    DmlOperatorConcat(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo),
        ConcatHelper(kernelInfo, kernelInfo.GetTensorShapeDescription())
    {
        Initialize(kernelInfo);

        uint32_t dmlAxis = GetDmlAdjustedAxis(m_axis, kernelInfo, m_inputTensorDescs.front().GetDimensionCount());

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_JOIN_OPERATOR_DESC joinDesc = {};
        joinDesc.InputCount = gsl::narrow_cast<uint32_t>(inputDescs.size());
        joinDesc.InputTensors = inputDescs.data();
        joinDesc.OutputTensor = outputDescs.data();
        joinDesc.Axis = dmlAxis;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_JOIN, &joinDesc };

        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Concat, DmlOperatorConcat);

} // namespace Dml

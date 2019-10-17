// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorSplit : public DmlOperator, public SplitHelper
{
public:
    using Self = DmlOperatorSplit;

    DmlOperatorSplit(const MLOperatorKernelCreationContext& kernelInfo)
        : DmlOperator(kernelInfo),
          SplitHelper(kernelInfo, kernelInfo.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1, "DML only supports split on a single input tensor.");
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() > 0, "Runtime error no output stream specified.");
        DmlOperator::Initialize(kernelInfo);

        uint32_t dmlAxis = GetDmlAdjustedAxis(m_axis, kernelInfo, m_inputTensorDescs.front().GetDimensionCount());

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_SPLIT_OPERATOR_DESC splitDesc = {};
        splitDesc.InputTensor = inputDescs.data();
        splitDesc.OutputTensors = outputDescs.data();
        splitDesc.OutputCount = gsl::narrow_cast<uint32_t>(outputDescs.size());
        splitDesc.Axis = dmlAxis;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_SPLIT, &splitDesc };

        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Split, DmlOperatorSplit);

} // namespace Dml

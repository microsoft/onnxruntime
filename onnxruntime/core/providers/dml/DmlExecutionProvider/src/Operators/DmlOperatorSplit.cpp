// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorSplit : public DmlOperator, public SplitHelper
{
public:
    using Self = DmlOperatorSplit;

    DmlOperatorSplit(const MLOperatorKernelCreationContext& kernelInfo, uint32_t opsetVersion)
        : DmlOperator(kernelInfo),
          SplitHelper(kernelInfo, kernelInfo.GetTensorShapeDescription(), opsetVersion)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() > 0, "Splits needs an input tensor.");
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() > 0, "Splits needs an output tensor.");

        // Use only the first input tensor. Later opset versions may pass parameters
        // like splits as dynamic parameters via tensors rather than constants,
        // and that second parameter is CPU based.
        std::vector<std::optional<uint32_t>> inputIndices = {0};
        DmlOperator::Initialize(kernelInfo, inputIndices, std::nullopt);

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

DML_OP_DEFINE_CREATION_FUNCTION(Split7, VersionedKernel<DmlOperatorSplit, 7>);
DML_OP_DEFINE_CREATION_FUNCTION(Split11, VersionedKernel<DmlOperatorSplit, 11>);
DML_OP_DEFINE_CREATION_FUNCTION(Split13, VersionedKernel<DmlOperatorSplit, 13>);

} // namespace Dml

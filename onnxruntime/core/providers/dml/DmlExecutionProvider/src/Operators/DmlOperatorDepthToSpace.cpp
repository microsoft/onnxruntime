// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorDepthToSpace : public DmlOperator, public DepthToSpaceHelper
{
public:
    DmlOperatorDepthToSpace(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        DepthToSpaceHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        DmlOperator::Initialize(kernelCreationContext);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        ML_CHECK_VALID_ARGUMENT(inputDescs.size() == 1);
        ML_CHECK_VALID_ARGUMENT(outputDescs.size() == 1);

        std::string mode = kernelCreationContext.GetOptionalAttribute<std::string>(AttrName::Mode, "DCR");
        DML_DEPTH_SPACE_ORDER depthSpaceOrder = Dml::MapStringToDepthSpaceMode(mode);

        DML_DEPTH_TO_SPACE1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.BlockSize = m_blockSize;
        operatorDesc.Order = depthSpaceOrder;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_DEPTH_TO_SPACE1, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(DepthToSpace, DmlOperatorDepthToSpace);

} // namespace Dml

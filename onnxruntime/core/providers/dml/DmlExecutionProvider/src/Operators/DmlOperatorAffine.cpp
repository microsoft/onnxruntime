// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorAffine : public DmlOperator
{
public:
    using Self = DmlOperatorAffine;

    DmlOperatorAffine(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo)
    {
        Initialize(kernelInfo);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_SCALE_BIAS scaleBias = {};
        scaleBias.Scale = kernelInfo.GetOptionalAttribute<float>(AttrName::Alpha, 0.0f);
        scaleBias.Bias = kernelInfo.GetOptionalAttribute<float>(AttrName::Beta, 0.0f);

        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC opDesc = {};
        opDesc.InputTensor = inputDescs.data();
        opDesc.OutputTensor = outputDescs.data();
        opDesc.ScaleBias = &scaleBias;

        SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &opDesc}, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Affine, DmlOperatorAffine);

} // namespace Dml

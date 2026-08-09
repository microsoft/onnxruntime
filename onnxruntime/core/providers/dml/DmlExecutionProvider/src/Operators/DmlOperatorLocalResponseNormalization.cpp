// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorLocalResponseNormalization : public DmlOperator
{
public:
    DmlOperatorLocalResponseNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        DmlOperator::Initialize(kernelCreationContext);

        const int size = kernelCreationContext.GetOptionalAttribute<int>(AttrName::Size, 0);
        const float bias = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Bias, 0.0f);
        const float alpha = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Alpha, 0.0f);
        const float beta = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Beta, 0.0f);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        ML_CHECK_VALID_ARGUMENT(inputDescs.size() == 1);
        ML_CHECK_VALID_ARGUMENT(outputDescs.size() == 1);

        DML_LOCAL_RESPONSE_NORMALIZATION_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.CrossChannel = true; // crossChannel - ONNX only supports cross-channel.
        operatorDesc.LocalSize = gsl::narrow_cast<uint32_t>(size);
        operatorDesc.Alpha = alpha;
        operatorDesc.Beta = beta;
        operatorDesc.Bias = bias;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_LOCAL_RESPONSE_NORMALIZATION, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(LRN, DmlOperatorLocalResponseNormalization);

} // namespace Dml

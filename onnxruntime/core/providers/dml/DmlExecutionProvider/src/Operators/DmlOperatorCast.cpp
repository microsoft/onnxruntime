// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorCast : public DmlOperator
{
public:
    using Self = DmlOperatorCast;

    DmlOperatorCast(
        const MLOperatorKernelCreationContext& kernelInfo
        ) : DmlOperator(kernelInfo)
    {
        Initialize(kernelInfo);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_CAST_OPERATOR_DESC castDesc = {};
        castDesc.InputTensor = inputDescs.data();
        castDesc.OutputTensor = outputDescs.data();

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_CAST, &castDesc };
        
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensorsForExecute(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensorsForExecute(kernelContext);

        THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::make_span(inputTensors),
            gsl::make_span(outputTensors)));
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Cast, DmlOperatorCast);

} // namespace Dml

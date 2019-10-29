// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorTopK : public DmlOperator, public TopKHelper
{
public:
    using Self = DmlOperatorTopK;

    DmlOperatorTopK(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        TopKHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        DmlOperator::Initialize(kernelCreationContext);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        ML_CHECK_VALID_ARGUMENT(inputDescs.size() == 1);
        ML_CHECK_VALID_ARGUMENT(outputDescs.size() == 2);

        uint32_t dmlAxis = GetDmlAdjustedAxis(m_axis, kernelCreationContext, m_inputTensorDescs.front().GetDimensionCount());

        DML_TOP_K_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputValueTensor = &outputDescs[0];
        operatorDesc.OutputIndexTensor = &outputDescs[1];
        operatorDesc.Axis = dmlAxis;
        operatorDesc.K = m_k;

        // Index tensor is always of type int64. We need to create an extra DML operator to
        // initialize the tensor data.
        m_zeroOperator = InitializeZeroInt64Tensor(m_outputTensorDescs[1].GetBufferSizeInBytes());

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_TOP_K, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }

    void Compute(const MLOperatorKernelContext& kernelContext) override
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensorsForExecute(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensorsForExecute(kernelContext);

        ExecuteZeroInt64Tensor(m_zeroOperator.Get(), outputTensors[1]);

        THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::make_span(inputTensors),
            gsl::make_span(outputTensors)
            ));
    }

private:
    ComPtr<IDMLCompiledOperator> m_zeroOperator;
};

DML_OP_DEFINE_CREATION_FUNCTION(TopK, DmlOperatorTopK);

} // namespace Dml

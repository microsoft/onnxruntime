// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorTopK : public DmlOperator, public TopKHelper
{
public:
    using Self = DmlOperatorTopK;

    DmlOperatorTopK(const MLOperatorKernelCreationContext& kernelCreationContext, uint32_t opsetVersion)
    :   DmlOperator(kernelCreationContext),
        TopKHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription(), opsetVersion)
    {
        ML_CHECK_VALID_ARGUMENT(((opsetVersion >= 1 && opsetVersion < 10) && kernelCreationContext.GetInputCount() == 1)
                             || ((opsetVersion >= 10) && kernelCreationContext.GetInputCount() == 2));
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 2, "TopK expects 2 output tensors.");

        std::vector<std::optional<uint32_t>> inputIndices = { 0 }; // Use only the first tensor. The second tensor is CPU-based.
        std::vector<std::optional<uint32_t>> outputIndices = { 0, 1 };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices);
        m_outputTensorDescs[1].ForceUnsignedDataType(); // DML operator accepts uint32_t/uint64 for indices tensor.

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        ML_CHECK_VALID_ARGUMENT(inputDescs.size() == 1);
        ML_CHECK_VALID_ARGUMENT(outputDescs.size() == 2);

        uint32_t dmlAxis = GetDmlAdjustedAxis(m_axis, kernelCreationContext, m_inputTensorDescs.front().GetDimensionCount());
        int32_t largest = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::Largest, 1);
        // Note the "sorted" attribute is not needed because the specification only dictates that
        // when sorted is true that elements are returned in order, but when false, the order of
        // returned 'Values' and 'Indices' are undefined. So returning them in order is a superset
        // of returning them in arbitrary order.

        DML_TOP_K1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputDescs.data();
        operatorDesc.OutputValueTensor = &outputDescs[0];
        operatorDesc.OutputIndexTensor = &outputDescs[1];
        operatorDesc.Axis = dmlAxis;
        operatorDesc.K = m_k;
        operatorDesc.AxisDirection = largest ? DML_AXIS_DIRECTION_DECREASING : DML_AXIS_DIRECTION_INCREASING;

        // Index tensor is always of type int64. We need to create an extra DML operator to
        // initialize the tensor data.
        m_zeroOperator = InitializeZeroInt64Tensor(m_outputTensorDescs[1].GetBufferSizeInBytes());

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_TOP_K1, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }

    void Compute(const MLOperatorKernelContext& kernelContext) override
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensorsForExecute(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensorsForExecute(kernelContext);

        if (m_zeroOperator)
        {
            ExecuteZeroInt64Tensor(kernelContext.GetAllocator(), m_zeroOperator.Get(), outputTensors[1]);
        }

        ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            kernelContext.GetAllocator(),
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::make_span(inputTensors),
            gsl::make_span(outputTensors)
            ));
    }

private:
    ComPtr<IDMLCompiledOperator> m_zeroOperator;
};

DML_OP_DEFINE_CREATION_FUNCTION(TopK7,  VersionedKernel<DmlOperatorTopK, 7 >);
DML_OP_DEFINE_CREATION_FUNCTION(TopK10, VersionedKernel<DmlOperatorTopK, 10>);
DML_OP_DEFINE_CREATION_FUNCTION(TopK11, VersionedKernel<DmlOperatorTopK, 11>);

} // namespace Dml

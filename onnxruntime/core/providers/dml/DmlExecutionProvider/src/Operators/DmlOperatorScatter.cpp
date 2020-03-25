// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorScatter : public DmlOperator
{
public:
    DmlOperatorScatter(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 3, "Scatter expects 3 inputs.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "Scatter expects 1 output.");

        auto tensorShapeDescription = kernelCreationContext.GetTensorShapeDescription();
        std::vector<DimensionType> dataDimensions = tensorShapeDescription.GetInputTensorShape(0);
        std::vector<DimensionType> indicesDimensions = tensorShapeDescription.GetInputTensorShape(1);
        std::vector<DimensionType> updatesDimensions = tensorShapeDescription.GetInputTensorShape(2);
        std::vector<DimensionType> outputDimensions = tensorShapeDescription.GetOutputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(dataDimensions == outputDimensions);
        ML_CHECK_VALID_ARGUMENT(indicesDimensions == updatesDimensions);
        ML_CHECK_VALID_ARGUMENT(dataDimensions.size() == indicesDimensions.size());
        ML_CHECK_VALID_ARGUMENT(dataDimensions.size() <= OperatorHelper::NchwDimensionCount);

        // When the indices tensor is empty, Scatter is basically Identity. But since DML doesn't support empty or null
        // tensors, we have to special-case it outside of DML.
        if (OperatorHelper::ContainsEmptyDimensions(indicesDimensions))
        {
            std::vector<std::optional<uint32_t>> kernelInputIndices(1, 0);
            DmlOperator::Initialize(kernelCreationContext, kernelInputIndices);

            std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
            std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

            assert(inputDescs.size() == 1);
            assert(outputDescs.size() == 1);

            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC operatorDesc = {};
            operatorDesc.InputTensor = &inputDescs[0];
            operatorDesc.OutputTensor = outputDescs.data();
            operatorDesc.ScaleBias = nullptr;

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &operatorDesc };
            SetDmlOperatorDesc(opDesc, kernelCreationContext);
        }
        else
        {
            DmlOperator::Initialize(kernelCreationContext);

            std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
            std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
            assert(inputDescs.size() == 3);
            assert(outputDescs.size() == 1);

            m_inputTensorDescs[1].ForceUnsignedDataType();

            // Read the axis.
            int onnxAxis = kernelCreationContext.GetOptionalAttribute<int>(AttrName::Axis, 0);
            uint32_t dmlAxis = GetDmlAdjustedAxis(onnxAxis, kernelCreationContext, m_inputTensorDescs.front().GetDimensionCount());

            DML_SCATTER_OPERATOR_DESC operatorDesc = {};
            operatorDesc.InputTensor = &inputDescs[0];
            operatorDesc.IndicesTensor = &inputDescs[1];
            operatorDesc.UpdatesTensor = &inputDescs[2];
            operatorDesc.OutputTensor = outputDescs.data();
            operatorDesc.Axis = dmlAxis;

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_SCATTER, &operatorDesc };
            SetDmlOperatorDesc(opDesc, kernelCreationContext);
        }
    }
};

class DmlOperatorScatterNd : public DmlOperator
{
public:
    DmlOperatorScatterNd(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 3, "ScatterND expects 3 inputs.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "ScatterND expects 1 output.");

        auto tensorShapeDescription = kernelCreationContext.GetTensorShapeDescription();
        std::vector<DimensionType> dataDimensions = tensorShapeDescription.GetInputTensorShape(0);
        std::vector<DimensionType> indicesDimensions = tensorShapeDescription.GetInputTensorShape(1);
        std::vector<DimensionType> updatesDimensions = tensorShapeDescription.GetInputTensorShape(2);
        std::vector<DimensionType> outputDimensions = tensorShapeDescription.GetOutputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(dataDimensions == outputDimensions);
        ML_CHECK_VALID_ARGUMENT(dataDimensions.size() <= OperatorHelper::NchwDimensionCount);
        ML_CHECK_VALID_ARGUMENT(indicesDimensions.size() <= OperatorHelper::NchwDimensionCount);
        ML_CHECK_VALID_ARGUMENT(updatesDimensions.size() <= OperatorHelper::NchwDimensionCount);
        ML_CHECK_VALID_ARGUMENT(outputDimensions.size() <= OperatorHelper::NchwDimensionCount);

        DmlOperator::Initialize(kernelCreationContext);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        assert(inputDescs.size() == 3);
        assert(outputDescs.size() == 1);

        m_inputTensorDescs[1].ForceUnsignedDataType();

        DML_SCATTER_ND_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.IndicesTensor = &inputDescs[1];
        operatorDesc.UpdatesTensor = &inputDescs[2];
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.InputDimensionCount = static_cast<uint32_t>(dataDimensions.size());
        operatorDesc.IndicesDimensionCount = static_cast<uint32_t>(indicesDimensions.size());

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_SCATTER_ND, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Scatter9, DmlOperatorScatter);
DML_OP_DEFINE_CREATION_FUNCTION(Scatter11, DmlOperatorScatter);
DML_OP_DEFINE_CREATION_FUNCTION(ScatterElements, DmlOperatorScatter);
DML_OP_DEFINE_CREATION_FUNCTION(ScatterND, DmlOperatorScatterNd);

} // namespace Dml

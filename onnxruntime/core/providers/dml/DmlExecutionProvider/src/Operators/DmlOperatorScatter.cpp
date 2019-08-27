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

        DmlOperator::Initialize(kernelCreationContext);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        assert(inputDescs.size() == 3);
        assert(outputDescs.size() == 1);

        m_inputTensorDescs[1].ForceUnsignedDataType();

        auto tensorShapeDescription = kernelCreationContext.GetTensorShapeDescription();;
        std::vector<DimensionType> dataDimensions = tensorShapeDescription.GetInputTensorShape(0);
        std::vector<DimensionType> indicesDimensions = tensorShapeDescription.GetInputTensorShape(1);
        std::vector<DimensionType> updatesDimensions = tensorShapeDescription.GetInputTensorShape(2);
        std::vector<DimensionType> outputDimensions = tensorShapeDescription.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(dataDimensions == outputDimensions);
        ML_CHECK_VALID_ARGUMENT(indicesDimensions == updatesDimensions);
        ML_CHECK_VALID_ARGUMENT(dataDimensions.size() == indicesDimensions.size());
        ML_CHECK_VALID_ARGUMENT(dataDimensions.size() <= OperatorHelper::NchwDimensionCount);

        // Read the axis.
        int onnxAxis = kernelCreationContext.GetOptionalAttribute<int>(AttrName::Axis, 0);
        ML_CHECK_VALID_ARGUMENT(onnxAxis >= -int(dataDimensions.size()) && onnxAxis <= int(dataDimensions.size()) - 1);
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
};

DML_OP_DEFINE_CREATION_FUNCTION(Scatter, DmlOperatorScatter);

} // namespace Dml

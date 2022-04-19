// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorOneHot : public DmlOperator, OneHotHelper
{
public:
    using Self = DmlOperatorOneHot;

    DmlOperatorOneHot(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext), 
        OneHotHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 3);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);
        std::vector<std::optional<uint32_t>> inputIndices = { 0, 2 }; // The second tensor ('depth') is not bound, just 'indices' and 'values'.
        std::vector<std::optional<uint32_t>> outputIndices = { 0 };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices);
        
        // Unsqueeze the indices tensor by inserting a flat dimension of size 1,
        // and compute the output tensor by expanding along the active axis.
        // This way they are both size-compatible and directly consumable by DirectML.
        std::vector<uint32_t> indicesDimensions;
        indicesDimensions = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        indicesDimensions.insert(indicesDimensions.begin() + m_absoluteAxis, 1u);

        // Update the tensor descriptions with new sizes.
        m_inputTensorDescs[0] =
            TensorDesc(
                m_inputTensorDescs[0].GetMlOperatorDataType(),
                gsl::make_span(indicesDimensions),
                gsl::make_span(indicesDimensions),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                1, // minDimensionCount
                0
            );

        m_outputTensorDescs[0] =
            TensorDesc(
                m_outputTensorDescs[0].GetMlOperatorDataType(),
                gsl::make_span(m_outputDimensions),
                gsl::make_span(m_outputDimensions),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                1, // minDimensionCount
                0
            );

        // Adjust the axis so it's in DML's terms rather than the original ONNX indexing.
        uint32_t dmlAxis = GetDmlAdjustedAxis(
            m_absoluteAxis,
            gsl::narrow_cast<uint32_t>(indicesDimensions.size()),
            m_inputTensorDescs.front().GetDimensionCount()
        );
        
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ONE_HOT_OPERATOR_DESC operatorDesc = {};
        operatorDesc.IndicesTensor = &inputDescs[0];
        operatorDesc.ValuesTensor = &inputDescs[1];
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Axis = dmlAxis;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_ONE_HOT, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(OneHot, DmlOperatorOneHot);

} // namespace Dml

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorPadding : public DmlOperator, public PaddingHelper
{
public:
    DmlOperatorPadding(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo),
        PaddingHelper(kernelInfo, kernelInfo.GetTensorShapeDescription())
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);
        DmlOperator::Initialize(kernelInfo);

        assert(m_inputTensorDescs[0].GetDimensionCount() >= gsl::narrow_cast<uint32_t>(m_startPadding.size()));
        assert(m_inputTensorDescs[0].GetDimensionCount() >= gsl::narrow_cast<uint32_t>(m_endPadding.size()));

        // Pad the parameters to respect DML's requirements
        m_startPadding.insert(
            m_startPadding.begin(),
            m_inputTensorDescs[0].GetDimensionCount() - gsl::narrow_cast<uint32_t>(m_startPadding.size()),
            0);

        m_endPadding.insert(
            m_endPadding.begin(),
            m_inputTensorDescs[0].GetDimensionCount() - gsl::narrow_cast<uint32_t>(m_endPadding.size()),
            0);

        // Convert padding mode.
        DML_PADDING_MODE mode = DML_PADDING_MODE_CONSTANT;
        std::string modeString = kernelInfo.GetOptionalAttribute<std::string>(AttrName::Mode, AttrValue::Reflect);

        if (modeString == AttrValue::Constant)
        {
            mode = DML_PADDING_MODE_CONSTANT;
        }
        else if (modeString == AttrValue::Edge)
        {
            mode = DML_PADDING_MODE_EDGE;
        }
        else if (modeString == AttrValue::Reflect)
        {
            mode = DML_PADDING_MODE_REFLECTION;
        }
        else
        {
            ML_INVALID_ARGUMENT("Unknown Pad mode attribute.");
        }

        float value = kernelInfo.GetOptionalAttribute<float>(AttrName::Value, 0.0f);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_PADDING_OPERATOR_DESC paddingDesc = {};
        paddingDesc.InputTensor = inputDescs.data();
        paddingDesc.OutputTensor = outputDescs.data();
        paddingDesc.PaddingMode = mode;
        paddingDesc.PaddingValue = value;
        paddingDesc.DimensionCount = gsl::narrow_cast<uint32_t>(m_startPadding.size());
        paddingDesc.StartPadding = m_startPadding.data();
        paddingDesc.EndPadding = m_endPadding.data();

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_PADDING, &paddingDesc };

        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Pad, DmlOperatorPadding);

} // namespace Dml

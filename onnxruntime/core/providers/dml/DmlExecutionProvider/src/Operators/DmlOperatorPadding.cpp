// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorPadding : public DmlOperator, public PaddingHelper
{
public:
    DmlOperatorPadding(const MLOperatorKernelCreationContext& kernelInfo, uint32_t opsetVersion)
    :   DmlOperator(kernelInfo),
        PaddingHelper(kernelInfo, kernelInfo.GetTensorShapeDescription(), opsetVersion)
    {
        const uint32_t inputCount = kernelInfo.GetInputCount();
        ML_CHECK_VALID_ARGUMENT((opsetVersion >= 2 && opsetVersion < 11 && inputCount == 1)
                             || (opsetVersion >= 11 && inputCount >= 2 && inputCount <= 3));
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        std::vector<std::optional<uint32_t>> kernelInputIndices = { 0 }; // Only bind GPU to first 'data' tensor.
        DmlOperator::Initialize(kernelInfo, kernelInputIndices);

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

        // Read the constant value which can come from an attribute or tensor.
        float value = 0.0f;
        if (opsetVersion >= 11)
        {
            if (kernelInfo.IsInputValid(2))
            {
                auto valueTensor = kernelInfo.GetConstantInputTensor(2);
                value = static_cast<float>(ReadScalarTensorCastToFloat64(valueTensor));
            }
        }
        else
        {
             value = kernelInfo.GetOptionalAttribute<float>(AttrName::Value, 0.0f);
        }

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

// A specific type of operation for registration.
template <uint32_t opsetVersion>
class DmlOperatorPaddingTemplate : public DmlOperatorPadding
{
public:
    DmlOperatorPaddingTemplate(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperatorPadding(kernelInfo, opsetVersion)
    {
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Pad7, DmlOperatorPaddingTemplate<7>);
DML_OP_DEFINE_CREATION_FUNCTION(Pad11, DmlOperatorPaddingTemplate<11>);

} // namespace Dml

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorPooling : public DmlOperator, public PoolingHelperBase
{
public:
    using Self = DmlOperatorPooling;

    DmlOperatorPooling(
        const MLOperatorKernelCreationContext& kernelInfo,
        DML_OPERATOR_TYPE function,
        bool useGlobalPooling
        )
    :   DmlOperator(kernelInfo),
        PoolingHelperBase(kernelInfo, kernelInfo.GetTensorShapeDescription(), useGlobalPooling),
        m_function(function)
    {
        const bool hasDilations =
            std::any_of(
                m_kernel.dilations,
                m_kernel.dilations + m_kernel.spatialDimensionCount,
                [](auto d) {return d != 1; }
            );
        const bool hasOutputIndices = (kernelInfo.GetOutputCount() > 1 && kernelInfo.IsOutputValid(1));
        std::vector<std::optional<uint32_t>> kernelOutputIndices = {0};

        if (function == DML_OPERATOR_MAX_POOLING2 && (hasOutputIndices || hasDilations))
        {
            kernelOutputIndices.emplace_back(1);
        }
        DmlOperator::Initialize(kernelInfo, std::nullopt, kernelOutputIndices);
        
        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        ML_CHECK_VALID_ARGUMENT(inputDescs.size() >= 1, "MaxPool input count must be >=1.");
        ML_CHECK_VALID_ARGUMENT(outputDescs.size() >= 1, "MaxPool output count must be >=1.");

        assert(m_kernel.spatialDimensionCount <= ARRAYSIZE(m_kernel.windowSize));

        // The below attributes are temporarily not supported:
        int storageOrder = kernelInfo.GetOptionalAttribute<int>(AttrName::StorageOrder, 0);
        ORT_THROW_HR_IF(E_NOTIMPL, storageOrder != 0);

        // DML requires that DimensionCount be equal to Input.DimCount - 2 for Pooling
        uint32_t expectedSpatialDimCount = m_inputTensorDescs[0].GetDimensionCount() - 2;
        if (m_kernel.spatialDimensionCount < expectedSpatialDimCount)
        {
            size_t shift = expectedSpatialDimCount - m_kernel.spatialDimensionCount;

            for (int i = gsl::narrow_cast<int>(m_kernel.spatialDimensionCount) - 1; i >= 0; i--)
            {
                m_kernel.windowSize[i + shift] = m_kernel.windowSize[i];
                m_kernel.windowSize[i] = 1;

                m_kernel.strides[i + shift] = m_kernel.strides[i];
                m_kernel.strides[i] = 1;

                m_kernel.startPadding[i + shift] = m_kernel.startPadding[i];
                m_kernel.startPadding[i] = 0;

                m_kernel.endPadding[i + shift] = m_kernel.endPadding[i];
                m_kernel.endPadding[i] = 0;

                m_kernel.dilations[i + shift] = m_kernel.dilations[i];
                m_kernel.dilations[i] = 1;
            }

            m_kernel.spatialDimensionCount = expectedSpatialDimCount;
        }

        auto SetOpDesc = [&](auto& poolingDesc)
        {
            poolingDesc.InputTensor = inputDescs.data();
            poolingDesc.OutputTensor = outputDescs.data();
            poolingDesc.DimensionCount = m_kernel.spatialDimensionCount;
            poolingDesc.WindowSize = m_kernel.windowSize;
            poolingDesc.Strides = m_kernel.strides;
            poolingDesc.StartPadding = m_kernel.startPadding;
            poolingDesc.EndPadding = m_kernel.endPadding;

            DML_OPERATOR_DESC opDesc = {};
            opDesc.Type = ApiTraits::OperatorDescTraits<typename std::remove_reference<decltype(poolingDesc)>::type>::Type;
            opDesc.Desc = &poolingDesc;
            SetDmlOperatorDesc(opDesc, kernelInfo);
        };

        switch (m_function)
        {
            case DML_OPERATOR_AVERAGE_POOLING:
            {
                DML_AVERAGE_POOLING_OPERATOR_DESC desc = {};
                desc.IncludePadding = kernelInfo.GetOptionalAttribute<bool>(AttrName::CountIncludePad, false);
                SetOpDesc(desc);
                break;
            }
            case DML_OPERATOR_LP_POOLING:
            {
                DML_LP_POOLING_OPERATOR_DESC desc = {};
                desc.P = kernelInfo.GetOptionalAttribute<int>(AttrName::P, 2);
                ML_CHECK_VALID_ARGUMENT(desc.P > 0);
                SetOpDesc(desc);
                break;
            }
            case DML_OPERATOR_MAX_POOLING:
            case DML_OPERATOR_MAX_POOLING1:
            case DML_OPERATOR_MAX_POOLING2:
            {
                if (hasOutputIndices || hasDilations)
                {
                    DML_MAX_POOLING2_OPERATOR_DESC desc = {};

                    if (hasOutputIndices)
                    {
                        m_outputTensorDescs[1].ForceUnsignedDataType(); // MaxPool accepts uint32_t/uint64_t.
                        desc.OutputIndicesTensor = &outputDescs[1];
                    }

                    desc.Dilations = m_kernel.dilations;
                    SetOpDesc(desc);
                }
                else
                {
                    // Use the old pooling command, which supports potential metacommands.
                    DML_MAX_POOLING_OPERATOR_DESC desc = {};
                    SetOpDesc(desc);
                }
                break;
            }
        }
    }

private:
    DML_OPERATOR_TYPE m_function;
};

// A specific type of operation for registration.
template <DML_OPERATOR_TYPE Function, bool UseGlobalPooling>
class DmlOperatorPoolingTemplate : public DmlOperatorPooling
{
public:
    DmlOperatorPoolingTemplate(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperatorPooling(kernelInfo, Function, UseGlobalPooling)
    {
    }
};

void CALLBACK QueryMaxPool(IMLOperatorSupportQueryContextPrivate* context, bool* isSupported)
{
    *isSupported = false;
    
    MLOperatorAttributes attributes(context);

    int storageOrder = attributes.GetOptionalAttribute<int>(AttrName::StorageOrder, 0);
    if (storageOrder != 0)
    {
        return;
    }

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(AveragePool,           DmlOperatorPoolingTemplate<DML_OPERATOR_AVERAGE_POOLING, false>);
DML_OP_DEFINE_CREATION_FUNCTION(GlobalAveragePool,     DmlOperatorPoolingTemplate<DML_OPERATOR_AVERAGE_POOLING, true>);
DML_OP_DEFINE_CREATION_FUNCTION(MaxPool,               DmlOperatorPoolingTemplate<DML_OPERATOR_MAX_POOLING2, false>);
DML_OP_DEFINE_CREATION_FUNCTION(GlobalMaxPool,         DmlOperatorPoolingTemplate<DML_OPERATOR_MAX_POOLING, true>);
DML_OP_DEFINE_CREATION_FUNCTION(LpPool,                DmlOperatorPoolingTemplate<DML_OPERATOR_LP_POOLING, false>);
DML_OP_DEFINE_CREATION_FUNCTION(GlobalLpPool,          DmlOperatorPoolingTemplate<DML_OPERATOR_LP_POOLING, true>);

} // namespace Dml

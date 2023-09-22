// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorQLinearAveragePooling : public DmlOperator, public PoolingHelperBase
{
    // For QLinear Avg Pool ORT and DML have same indexing order
    enum OrtInputTensors : uint32_t
    {
        ortInput,
        ortInputScale,
        ortInputZeroPoint,
        ortOutputScale,
        ortOutputZeroPoint,
        ortInputCount
    };

public:
    using Self = DmlOperatorQLinearAveragePooling;

    DmlOperatorQLinearAveragePooling(
        const MLOperatorKernelCreationContext& kernelInfo,
        bool useGlobalPooling
        )
    :   DmlOperator(kernelInfo),
        PoolingHelperBase(kernelInfo, kernelInfo.GetTensorShapeDescription(), useGlobalPooling)
    {
        DmlOperator::Initialize(kernelInfo);

        bool isNhwc = m_kernel.channelsLast;
        std::vector<DimensionType> inputShape = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(OrtInputTensors::ortInput);
        std::vector<DimensionType> outputShape = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);

        uint32_t dmlDimSize = m_inputTensorDescs[OrtInputTensors::ortInput].GetDimensionCount();
        ML_CHECK_VALID_ARGUMENT(dmlDimSize >= 2);
        
        // DML requires that DimensionCount be equal to Input.dmlDimSize - 2 for Pooling
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

        // Initialize dimensionMapping for NCHW or NHWC layout
        std::vector<uint32_t> dimensionMapping = {0u, dmlDimSize - 1u};
        dimensionMapping.resize(dmlDimSize);
        if (isNhwc)
        {
            // Form a remapping for dimensions so C is moved before the spatial dimensions.
            // e.g. NWC   -> {0,2,1}     -> NCW
            //      NHWC  -> {0,3,1,2}   -> NCHW
            //      NDHWC -> {0,4,1,2,3} -> NCDHW
            std::iota(dimensionMapping.begin() + 2, dimensionMapping.end(), 1u);
        }
        else
        {
            // Use NCHW {0,1,2,3} format with increasing order of indexs 
            std::iota(dimensionMapping.begin() + 1, dimensionMapping.end(), 1u);
        }
        m_inputTensorDescs[OrtInputTensors::ortInput].PermuteDimensions(dimensionMapping, TensorAxis::LeftAligned);

        // Reshape the Input Scale to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the H channel.
        m_inputTensorDescs[OrtInputTensors::ortInputScale].PermuteDimensions(dimensionMapping, TensorAxis::LeftAligned);

        // Reshape the Input ZeroPoint to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the H channel.
        if (kernelInfo.IsInputValid(OrtInputTensors::ortInputZeroPoint))
        {
            m_inputTensorDescs[OrtInputTensors::ortInputZeroPoint].PermuteDimensions(dimensionMapping, TensorAxis::LeftAligned);
        }

        // Reshape the Output Scale to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the H channel.
        m_inputTensorDescs[OrtInputTensors::ortOutputScale].PermuteDimensions(dimensionMapping, TensorAxis::LeftAligned);

        // Reshape the Input ZeroPoint to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the H channel.
        if (kernelInfo.IsInputValid(OrtInputTensors::ortOutputZeroPoint))
        {
            m_inputTensorDescs[OrtInputTensors::ortOutputZeroPoint].PermuteDimensions(dimensionMapping, TensorAxis::LeftAligned);
        }

        // Initialize the output description while overriding the shape
        m_outputTensorDescs[0].PermuteDimensions(dimensionMapping, TensorAxis::LeftAligned);

        assert(m_kernel.spatialDimensionCount <= ARRAYSIZE(m_kernel.windowSize));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_QUANTIZED_LINEAR_AVERAGE_POOLING_OPERATOR_DESC qLinearAvgPooldesc = {};

        qLinearAvgPooldesc.InputTensor = &inputDescs[OrtInputTensors::ortInput];
        qLinearAvgPooldesc.InputScaleTensor = &inputDescs[OrtInputTensors::ortInputScale];
        qLinearAvgPooldesc.InputZeroPointTensor = &inputDescs[OrtInputTensors::ortInputZeroPoint];
        qLinearAvgPooldesc.OutputScaleTensor = &inputDescs[OrtInputTensors::ortOutputScale];;
        qLinearAvgPooldesc.OutputZeroPointTensor = &inputDescs[OrtInputTensors::ortOutputZeroPoint];;
        qLinearAvgPooldesc.OutputTensor = &outputDescs[0];
        qLinearAvgPooldesc.DimensionCount = m_kernel.spatialDimensionCount;
        qLinearAvgPooldesc.WindowSize = m_kernel.windowSize;
        qLinearAvgPooldesc.Strides = m_kernel.strides;
        qLinearAvgPooldesc.StartPadding = m_kernel.startPadding;
        qLinearAvgPooldesc.EndPadding = m_kernel.endPadding;
        qLinearAvgPooldesc.Dilations = m_kernel.dilations;
        qLinearAvgPooldesc.IncludePadding = kernelInfo.GetOptionalAttribute<bool>(AttrName::CountIncludePad, false);

        DML_OPERATOR_DESC opDesc = { (DML_OPERATOR_TYPE) DML_OPERATOR_QUANTIZED_LINEAR_AVERAGE_POOLING, &qLinearAvgPooldesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

template <bool UseGlobalPooling>
class DmlOperatorQuantizedPoolingTemplate : public DmlOperatorQLinearAveragePooling
{
public:
    DmlOperatorQuantizedPoolingTemplate(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperatorQLinearAveragePooling(kernelInfo, UseGlobalPooling)
    {
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(QLinearAveragePool, DmlOperatorQuantizedPoolingTemplate<false>);
DML_OP_DEFINE_CREATION_FUNCTION(QLinearGlobalAveragePool, DmlOperatorQuantizedPoolingTemplate<true>);

} // namespace Dml

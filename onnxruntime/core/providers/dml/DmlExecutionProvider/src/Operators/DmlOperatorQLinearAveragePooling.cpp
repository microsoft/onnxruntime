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
        // Reshape the Input Scale to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the H channel.
        m_inputTensorDescs[OrtInputTensors::ortInputScale] = CreateTensorDescFromInput(
            kernelInfo, 
            OrtInputTensors::ortInputScale,
            TensorAxis::DoNotCoerce, 
            TensorAxis::H,
            TensorAxis::LeftAligned,
            std::nullopt,
            dmlDimSize
            );

        // Resize the Input ZeroPoint to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the H channel.
        if (kernelInfo.IsInputValid(OrtInputTensors::ortInputZeroPoint))
        {

            m_inputTensorDescs[OrtInputTensors::ortInputZeroPoint] = CreateTensorDescFromInput(
                kernelInfo, 
                OrtInputTensors::ortInputZeroPoint,
                TensorAxis::DoNotCoerce, 
                TensorAxis::H,
                TensorAxis::LeftAligned,
                std::nullopt,
                dmlDimSize
                );
        }

        // Resize the Output Scale to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the H channel.
        m_inputTensorDescs[OrtInputTensors::ortOutputScale] = CreateTensorDescFromInput(
            kernelInfo, 
            OrtInputTensors::ortInputScale,
            TensorAxis::DoNotCoerce, 
            TensorAxis::H,
            TensorAxis::LeftAligned,
            std::nullopt,
            dmlDimSize
            );

        // Resize the Input ZeroPoint to be the same dimension as the input tensor.
        // The 1D tensor needs to be moved to the H channel.
        if (kernelInfo.IsInputValid(OrtInputTensors::ortOutputZeroPoint))
        {

            m_inputTensorDescs[OrtInputTensors::ortOutputZeroPoint] = CreateTensorDescFromInput(
                kernelInfo, 
                OrtInputTensors::ortOutputZeroPoint,
                TensorAxis::DoNotCoerce, 
                TensorAxis::H,
                TensorAxis::LeftAligned,
                std::nullopt,
                dmlDimSize
                );
        }

        // Initialize the output description while overriding the shape
        m_outputTensorDescs[0] = CreateTensorDescFromOutput(kernelInfo, 0, TensorAxis::DoNotCoerce, TensorAxis::W, TensorAxis::RightAligned, outputShape);

        assert(m_kernel.spatialDimensionCount <= ARRAYSIZE(m_kernel.windowSize));

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

        if (isNhwc)
        {
            uint32_t dimCount = m_inputTensorDescs[0].GetDimensionCount();
            const auto inputSizes = m_inputTensorDescs[OrtInputTensors::ortInput].GetSizes();
            std::vector<uint32_t> nchwInputSizes;
            std::vector<uint32_t> nchwInputStrides;
            ConvertNHWCToNCHW(dimCount, inputSizes, nchwInputSizes, nchwInputStrides);
            m_inputTensorDescs[OrtInputTensors::ortInput] = TensorDesc(m_inputTensorDescs[OrtInputTensors::ortInput].GetDmlDataType(), nchwInputSizes, nchwInputStrides);

            gsl::span<const uint32_t> inputScaleSizes = m_inputTensorDescs[OrtInputTensors::ortInputScale].GetSizes();
            std::vector<uint32_t> nchwInputScaleSizes;
            std::vector<uint32_t> nchwInputScaleStrides;
            ConvertNHWCToNCHW(dimCount, inputScaleSizes, nchwInputScaleSizes, nchwInputScaleStrides);
            m_inputTensorDescs[OrtInputTensors::ortInputScale] = TensorDesc(m_inputTensorDescs[OrtInputTensors::ortInputScale].GetDmlDataType(),     nchwInputScaleSizes, nchwInputScaleStrides);

            gsl::span<const uint32_t> inputZeroPointSizes = m_inputTensorDescs[OrtInputTensors::ortInputZeroPoint].GetSizes();
            std::vector<uint32_t> nchwInputZeroPointSizes;
            std::vector<uint32_t> nchwInputZeroPointStrides;
            ConvertNHWCToNCHW(dimCount, inputZeroPointSizes, nchwInputZeroPointSizes, nchwInputZeroPointStrides);
            m_inputTensorDescs[OrtInputTensors::ortInputZeroPoint] = TensorDesc(m_inputTensorDescs[OrtInputTensors::ortInputZeroPoint].GetDmlDataType(), nchwInputZeroPointSizes, nchwInputZeroPointStrides);

            gsl::span<const uint32_t> outputScaleSizes = m_inputTensorDescs[OrtInputTensors::ortOutputScale].GetSizes();
            std::vector<uint32_t> nchwOutputScaleSizes;
            std::vector<uint32_t> nchwOutputScaleStrides;
            ConvertNHWCToNCHW(dimCount, outputScaleSizes, nchwOutputScaleSizes, nchwOutputScaleStrides);
            m_inputTensorDescs[OrtInputTensors::ortOutputScale] = TensorDesc(m_inputTensorDescs[OrtInputTensors::ortOutputScale].GetDmlDataType(), nchwOutputScaleSizes, nchwOutputScaleStrides);

            gsl::span<const uint32_t> outputZeroPointSizes = m_inputTensorDescs[OrtInputTensors::ortOutputZeroPoint].GetSizes();
            std::vector<uint32_t> nchwOutputZeroPointSizes;
            std::vector<uint32_t> nchwOutputZeroPointStrides;
            ConvertNHWCToNCHW(dimCount, outputZeroPointSizes, nchwOutputZeroPointSizes, nchwOutputZeroPointStrides);
            m_inputTensorDescs[OrtInputTensors::ortOutputZeroPoint] = TensorDesc(m_inputTensorDescs[OrtInputTensors::ortOutputZeroPoint].GetDmlDataType(), nchwOutputZeroPointSizes, nchwOutputZeroPointStrides);

            gsl::span<const uint32_t> outputSizes = m_outputTensorDescs[0].GetSizes();
            std::vector<uint32_t> nchwOutputSizes;
            std::vector<uint32_t> nchwOutputStrides;
            ConvertNHWCToNCHW(dimCount, outputSizes, nchwOutputSizes, nchwOutputStrides);
            m_outputTensorDescs[0] = TensorDesc(m_outputTensorDescs[0].GetDmlDataType(), nchwOutputSizes, nchwOutputStrides);
        }

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

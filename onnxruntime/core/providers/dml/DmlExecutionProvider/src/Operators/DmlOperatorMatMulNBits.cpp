// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
class DmlOperatorMatMulNBits : public DmlOperator
{
public:
    DmlOperatorMatMulNBits(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= 3 && kernelInfo.GetInputCount() <= 4);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);
        DmlOperator::Initialize(kernelInfo);

        const bool hasZeroPoint = kernelInfo.IsInputValid(3);
        const uint32_t bRowCount = gsl::narrow_cast<uint32_t>(kernelInfo.GetAttribute<int64_t>(AttrName::UppercaseN));
        const uint32_t bColCount = gsl::narrow_cast<uint32_t>(kernelInfo.GetAttribute<int64_t>(AttrName::UppercaseK));
        const auto bitCount = kernelInfo.GetAttribute<int64_t>(AttrName::Bits);

        MLOperatorTensorDataType mlDataType = kernelInfo.GetInputEdgeDescription(0).tensorDataType;
        const DML_TENSOR_DATA_TYPE quantizedDataType = bitCount == 4 ? DML_TENSOR_DATA_TYPE_UINT4 : DML_TENSOR_DATA_TYPE_UINT8;

        std::vector<DimensionType> inputShape0 = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(0);
        std::vector<DimensionType> broadcastedInputShape1 = inputShape0;

        // The quantized input to MatMulNBits always comes as uint8, but the real shape is provided through the N and K attributes
        broadcastedInputShape1[broadcastedInputShape1.size() - 2] = bRowCount;
        broadcastedInputShape1[broadcastedInputShape1.size() - 1] = bColCount;

        // The B tensor always has a batch size and channel of 1 since it's a weight, but DML requires it to have the same channels
        // and channel count as the A tensor so we need to broadcast it
        std::vector<DimensionType> inputShape1(inputShape0.size(), 1);
        inputShape1[inputShape1.size() - 2] = bRowCount;
        inputShape1[inputShape1.size() - 1] = bColCount;

        std::vector<DimensionType> outputShape = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);

        std::vector<DimensionType> inputShape2 = inputShape1;

        uint32_t scaleElementCount = ComputeElementCountFromDimensions(m_inputTensorDescs[2].GetSizes());
        inputShape2[inputShape2.size() - 1] = scaleElementCount / bRowCount;

        std::vector<DimensionType> broadcastedInputShape2 = broadcastedInputShape1;
        broadcastedInputShape2.back() = inputShape2.back();

        // The quantized input and zero point to MatMulNBits always comes as uint8, but DML will expect the real data type (int4 or int8)
        m_inputTensorDescs[0] = TensorDesc::ConstructDefaultTensorDesc(mlDataType, inputShape0);
        m_inputTensorDescs[1] = TensorDesc::ConstructBroadcastedTensorDesc(GetMlDataTypeFromDmlDataType(quantizedDataType), broadcastedInputShape1, inputShape1);
        m_inputTensorDescs[2] = TensorDesc::ConstructBroadcastedTensorDesc(mlDataType, broadcastedInputShape2, inputShape2);

        if (hasZeroPoint)
        {
            m_inputTensorDescs[3] = TensorDesc::ConstructBroadcastedTensorDesc(GetMlDataTypeFromDmlDataType(quantizedDataType), broadcastedInputShape2, inputShape2);
        }

        auto dequantizedInputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(mlDataType, broadcastedInputShape1);
        auto dequantizedInputDmlTensorDesc = dequantizedInputTensorDesc.GetDmlDesc();

        // Initialize the output description while overriding the shape
        m_outputTensorDescs[0] = TensorDesc::ConstructDefaultTensorDesc(mlDataType, outputShape);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        std::vector<DML_TENSOR_DESC> quantizationParametersTensors;
        quantizationParametersTensors.push_back(inputDescs[2]);

        if (hasZeroPoint)
        {
            quantizationParametersTensors.push_back(inputDescs[3]);
        }

        DML_DEQUANTIZE_OPERATOR_DESC dequantizeDesc = {};
        dequantizeDesc.InputTensor = &inputDescs[1];
        dequantizeDesc.QuantizationParametersType = hasZeroPoint ? DML_QUANTIZATION_PARAMETERS_TYPE_SCALE_ZEROPOINT : DML_QUANTIZATION_PARAMETERS_TYPE_SCALE;
        dequantizeDesc.QuantizationParametersTensorCount = gsl::narrow_cast<uint32_t>(quantizationParametersTensors.size());
        dequantizeDesc.QuantizationParametersTensors = quantizationParametersTensors.data();
        dequantizeDesc.OutputTensor = &dequantizedInputDmlTensorDesc;
        DML_OPERATOR_DESC dequantizeOpDesc = { DML_OPERATOR_DEQUANTIZE, &dequantizeDesc };

        DML_GEMM_OPERATOR_DESC gemmDesc = {};
        gemmDesc.ATensor = &inputDescs[0];
        gemmDesc.BTensor = &dequantizedInputDmlTensorDesc;
        gemmDesc.CTensor = nullptr;
        gemmDesc.OutputTensor = &outputDescs[0];
        gemmDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
        gemmDesc.TransB = DML_MATRIX_TRANSFORM_TRANSPOSE;
        gemmDesc.Alpha = 1.0f;
        gemmDesc.Beta = 0.0f;
        DML_OPERATOR_DESC gemmOpDesc = { DML_OPERATOR_GEMM, &gemmDesc };

        // Construct the graph
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

        std::vector<const DML_OPERATOR_DESC*> opDescs = {
            &dequantizeOpDesc,
            &gemmOpDesc,
        };

        DML_INPUT_GRAPH_EDGE_DESC secondInputToDequantizeEdge = {};
        secondInputToDequantizeEdge.GraphInputIndex = 1;
        secondInputToDequantizeEdge.ToNodeIndex = 0;
        secondInputToDequantizeEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(secondInputToDequantizeEdge);

        DML_INPUT_GRAPH_EDGE_DESC scaleToDequantizeEdge = {};
        scaleToDequantizeEdge.GraphInputIndex = 2;
        scaleToDequantizeEdge.ToNodeIndex = 0;
        scaleToDequantizeEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(scaleToDequantizeEdge);

        if (hasZeroPoint)
        {
            DML_INPUT_GRAPH_EDGE_DESC zeroPointToDequantizeEdge = {};
            zeroPointToDequantizeEdge.GraphInputIndex = 3;
            zeroPointToDequantizeEdge.ToNodeIndex = 0;
            zeroPointToDequantizeEdge.ToNodeInputIndex = 2;
            inputEdges.push_back(zeroPointToDequantizeEdge);
        }

        DML_INPUT_GRAPH_EDGE_DESC firstInputToGemmEdge = {};
        firstInputToGemmEdge.GraphInputIndex = 0;
        firstInputToGemmEdge.ToNodeIndex = 1;
        firstInputToGemmEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(firstInputToGemmEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC dequantizeToGemmEdge = {};
        dequantizeToGemmEdge.FromNodeIndex = 0;
        dequantizeToGemmEdge.FromNodeOutputIndex = 0;
        dequantizeToGemmEdge.ToNodeIndex = 1;
        dequantizeToGemmEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(dequantizeToGemmEdge);

        DML_OUTPUT_GRAPH_EDGE_DESC gemmToOutputEdge = {};
        gemmToOutputEdge.FromNodeIndex = 1;
        gemmToOutputEdge.FromNodeOutputIndex = 0;
        gemmToOutputEdge.GraphOutputIndex = 0;
        outputEdges.push_back(gemmToOutputEdge);

        MLOperatorGraphDesc operatorGraphDesc = {};
        operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
        operatorGraphDesc.inputEdges = inputEdges.data();
        operatorGraphDesc.intermediateEdgeCount = gsl::narrow_cast<uint32_t>(intermediateEdges.size());
        operatorGraphDesc.intermediateEdges = intermediateEdges.data();
        operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
        operatorGraphDesc.outputEdges = outputEdges.data();
        operatorGraphDesc.nodeCount = gsl::narrow_cast<uint32_t>(opDescs.size());
        operatorGraphDesc.nodes = opDescs.data();

        SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelInfo);
    }
};

void CALLBACK QueryMatMulNBits(IMLOperatorSupportQueryContextPrivate* context, /*out*/ bool* isSupported)
{
    *isSupported = false;

    MLOperatorAttributes attributes(context);
    const auto bitCount = attributes.GetAttribute<int64_t>(AttrName::Bits);
    if (bitCount != 4 && bitCount != 8)
    {
        return;
    }

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(MatMulNBits, DmlOperatorMatMulNBits);
} // namespace Dml

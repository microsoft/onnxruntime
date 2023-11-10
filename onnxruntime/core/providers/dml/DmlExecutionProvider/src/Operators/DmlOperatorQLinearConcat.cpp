// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
// QLinearConcat = Dequantize + Join + Quantize
class DmlOperatorQLinearConcat : public DmlOperator, public QLinearConcatHelper
{
    // This order matches the ONNX schema.
    enum OnnxInputIndex
    {
        YScale,
        YZeroPoint,
        Count,
    };

public:
    DmlOperatorQLinearConcat(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        QLinearConcatHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        DmlOperator::Initialize(kernelCreationContext);

        auto outputShape = kernelCreationContext.GetTensorShapeDescription().GetOutputTensorShape(0);

        // inputs: {y_scale, y_zero_point, tuple(x_tensor, x_scale, x_zero_point)}
        uint32_t inputDefinitionCount = kernelCreationContext.GetInputCount();
        ML_CHECK_VALID_ARGUMENT(inputDefinitionCount >= 5, "Require at least 5 inputs.");
        ML_CHECK_VALID_ARGUMENT((inputDefinitionCount - 2) % 3 == 0, "Each input must be (tensor, scale, zero_point) tuple!");

        uint32_t inputCount = (inputDefinitionCount - 2) / 3;

        auto yScaleDataType = kernelCreationContext.GetInputEdgeDescription(OnnxInputIndex::YScale).tensorDataType;
        auto yZeroPointDataType = kernelCreationContext.GetInputEdgeDescription(OnnxInputIndex::YZeroPoint).tensorDataType;

        // broadcast y_scale and y_zero_point to output shape
        m_inputTensorDescs[OnnxInputIndex::YScale] = TensorDesc(
            yScaleDataType,
            outputShape,
            kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(OnnxInputIndex::YScale),
            TensorAxis::DoNotCoerce,
            TensorAxis::W,
            TensorAxis::RightAligned,
            NchwDimensionCount, // minDimensionCount
            0 // guaranteedBaseOffsetAlignment
        );

        m_inputTensorDescs[OnnxInputIndex::YZeroPoint] = TensorDesc(
            yZeroPointDataType,
            outputShape,
            kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(OnnxInputIndex::YZeroPoint),
            TensorAxis::DoNotCoerce,
            TensorAxis::W,
            TensorAxis::RightAligned,
            NchwDimensionCount, // minDimensionCount
            0 // guaranteedBaseOffsetAlignment
        );

        // Validate input tensors
        for (uint32_t inputIndex = 0; inputIndex < inputCount; ++inputIndex)
        {
            // Inputs(input tensor, scale, zero_point) are in tuple and starting from index 2
            auto tupleStartIndex = 2 + inputIndex * 3;
            auto xScaleDataType = kernelCreationContext.GetInputEdgeDescription(tupleStartIndex + 1).tensorDataType;
            auto xZeroPointDataType = kernelCreationContext.GetInputEdgeDescription(tupleStartIndex + 2).tensorDataType;
            ML_CHECK_VALID_ARGUMENT(xScaleDataType == yScaleDataType, "Wrong input type encountered for scale");
            ML_CHECK_VALID_ARGUMENT(xZeroPointDataType == yZeroPointDataType, "Wrong input type encountered for zero point");

            // broadcast x_scale and x_zero_point to shape of corresponding x
            m_inputTensorDescs[tupleStartIndex + 1] = TensorDesc(
                xScaleDataType,
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tupleStartIndex),
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tupleStartIndex + 1),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );

            m_inputTensorDescs[tupleStartIndex + 2] = TensorDesc(
                xZeroPointDataType,
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tupleStartIndex),
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tupleStartIndex + 2),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );
        }

        uint32_t dmlAxis = GetDmlAdjustedAxis(m_axis, kernelCreationContext, m_inputTensorDescs.front().GetDimensionCount(), 2);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        // 1. output edges between Dequantize and Join node
        // 2. input edge between Join and Quantize node
        std::vector<TensorDesc> intermediateOutputTensorDescs(inputCount);
        std::vector<DML_TENSOR_DESC> namedDequantizeOperatorDescs(inputCount);
        std::vector<DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC> dequantizeOperatorDescs(inputCount);
        std::vector<DML_OPERATOR_DESC> dmlOpDesc(inputCount);
        std::vector<const DML_OPERATOR_DESC*> opDescs;
        for (uint32_t inputIndex = 0; inputIndex < inputCount; ++inputIndex)
        {
            auto tupleStartIndex = 2 + inputIndex * 3;
            intermediateOutputTensorDescs[inputIndex] = TensorDesc(
                MLOperatorTensorDataType::Float,
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tupleStartIndex),
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tupleStartIndex),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment)
            );
            namedDequantizeOperatorDescs[inputIndex] = intermediateOutputTensorDescs[inputIndex].GetDmlDesc();

            dequantizeOperatorDescs[inputIndex].InputTensor = &inputDescs[tupleStartIndex];
            dequantizeOperatorDescs[inputIndex].ScaleTensor = &inputDescs[tupleStartIndex + 1];
            dequantizeOperatorDescs[inputIndex].ZeroPointTensor = &inputDescs[tupleStartIndex + 2];
            dequantizeOperatorDescs[inputIndex].OutputTensor = &namedDequantizeOperatorDescs[inputIndex];

            dmlOpDesc[inputIndex] = {DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR, &dequantizeOperatorDescs[inputIndex]};
            opDescs.push_back(&dmlOpDesc[inputIndex]);
        }

        TensorDesc joinOutputTensorDesc = TensorDesc(
            MLOperatorTensorDataType::Float,
            outputShape,
            outputShape,
            TensorAxis::DoNotCoerce,
            TensorAxis::W,
            TensorAxis::RightAligned,
            NchwDimensionCount, // minDimensionCount
            0 // guaranteedBaseOffsetAlignment
            );
        DML_TENSOR_DESC namedJoinOutputTensorDesc = joinOutputTensorDesc.GetDmlDesc();

        DML_JOIN_OPERATOR_DESC joinDesc = {};
        joinDesc.InputCount = gsl::narrow_cast<uint32_t>(namedDequantizeOperatorDescs.size());
        joinDesc.InputTensors = namedDequantizeOperatorDescs.data();
        joinDesc.OutputTensor = &namedJoinOutputTensorDesc;
        joinDesc.Axis = dmlAxis;

        const DML_OPERATOR_DESC opJoinDesc = {DML_OPERATOR_JOIN, &joinDesc};
        opDescs.push_back(&opJoinDesc);

        DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC quantizeOperatorDesc = {};
        quantizeOperatorDesc.InputTensor = joinDesc.OutputTensor;
        quantizeOperatorDesc.ScaleTensor = &inputDescs[OnnxInputIndex::YScale];
        quantizeOperatorDesc.ZeroPointTensor = &inputDescs[OnnxInputIndex::YZeroPoint];
        quantizeOperatorDesc.OutputTensor = &outputDescs[0];
        const DML_OPERATOR_DESC opQuantizeDesc = {DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR, &quantizeOperatorDesc};
        opDescs.push_back(&opQuantizeDesc);

        MLOperatorGraphDesc operatorGraphDesc = {};
        operatorGraphDesc.nodeCount = static_cast<uint32_t>(opDescs.size());
        operatorGraphDesc.nodesAsOpDesc = opDescs.data();

        uint32_t joinNodeIndex = operatorGraphDesc.nodeCount - 2;
        uint32_t quantizeNodeIndex = operatorGraphDesc.nodeCount - 1;

        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        // Input edges to Dequantize nodes
        for (uint32_t inputIndex = 0; inputIndex < inputCount; ++inputIndex)
        {
            auto tupleStartIndex = 2 + inputIndex * 3;
            for (auto edge_index = 0; edge_index < 3; ++edge_index)
            {
                DML_INPUT_GRAPH_EDGE_DESC inputEdge = {};
                inputEdge.GraphInputIndex = tupleStartIndex + edge_index;
                inputEdge.ToNodeIndex = inputIndex;
                inputEdge.ToNodeInputIndex = edge_index;
                inputEdges.push_back(inputEdge);
            }
        }

        // Input edge from y_scale to quantize node
        DML_INPUT_GRAPH_EDGE_DESC yScaleInputEdge = {};
        yScaleInputEdge.GraphInputIndex = 0; // Y_scale
        yScaleInputEdge.ToNodeIndex = quantizeNodeIndex;
        yScaleInputEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(yScaleInputEdge);

        // Input edge from y_zero_point to quantize node
        DML_INPUT_GRAPH_EDGE_DESC yZeroPointInputEdge = {};
        yZeroPointInputEdge.GraphInputIndex = 1; // Y_zero_point
        yZeroPointInputEdge.ToNodeIndex = quantizeNodeIndex;
        yZeroPointInputEdge.ToNodeInputIndex = 2;
        inputEdges.push_back(yZeroPointInputEdge);

        operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
        operatorGraphDesc.inputEdges = inputEdges.data();

        // set intermediate edges
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        for (uint32_t inputIndex = 0; inputIndex < inputCount; ++inputIndex)
        {
            DML_INTERMEDIATE_GRAPH_EDGE_DESC dequantizeToJoinEdge = {};
            dequantizeToJoinEdge.FromNodeIndex = inputIndex;
            dequantizeToJoinEdge.FromNodeOutputIndex = 0;
            dequantizeToJoinEdge.ToNodeIndex = joinNodeIndex; // The second last node Join
            dequantizeToJoinEdge.ToNodeInputIndex = inputIndex;
            intermediateEdges.push_back(dequantizeToJoinEdge);
        }

        DML_INTERMEDIATE_GRAPH_EDGE_DESC joinToQuantizeEdge = {};
        joinToQuantizeEdge.FromNodeIndex = joinNodeIndex;
        joinToQuantizeEdge.FromNodeOutputIndex = 0;
        joinToQuantizeEdge.ToNodeIndex = quantizeNodeIndex; // The second last node Join
        joinToQuantizeEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(joinToQuantizeEdge);

        operatorGraphDesc.intermediateEdgeCount = gsl::narrow_cast<uint32_t>(intermediateEdges.size());
        operatorGraphDesc.intermediateEdges = intermediateEdges.data();

        // set the output edges
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
        outputEdge.FromNodeIndex = quantizeNodeIndex;
        outputEdge.FromNodeOutputIndex = 0;
        outputEdge.GraphOutputIndex = 0;
        outputEdges.push_back(outputEdge);
        operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
        operatorGraphDesc.outputEdges = outputEdges.data();

        SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelCreationContext);
    };
};

DML_OP_DEFINE_CREATION_FUNCTION(QLinearConcat, DmlOperatorQLinearConcat);
} // namespace Dml

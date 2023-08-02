// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
// QLinearConcat = Dequantize + Join + Quantize
// This kernel is the first usage of graph based implementation
class DmlOperatorQLinearConcat : public DmlOperator, public QLinearConcatHelper
{

public:
    DmlOperatorQLinearConcat(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        QLinearConcatHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {

        DmlOperator::Initialize(kernelCreationContext);

        auto outputShape = kernelCreationContext.GetTensorShapeDescription().GetOutputTensorShape(0);

        // inputs: {y_scale, y_zero_point, tuple(x_tensor, x_scale, x_zero_point)}
        uint32_t input_def_count = kernelCreationContext.GetInputCount();
        ML_CHECK_VALID_ARGUMENT(input_def_count >= 5 && (input_def_count - 2) % 3 == 0,
              "Each input must be (tensor, scale, zero_point) tuple!");
        uint32_t input_count = (input_def_count - 2) / 3;
        auto yScaleDataType = kernelCreationContext.GetInputEdgeDescription(0).tensorDataType;
        auto yZeroPointDataType = kernelCreationContext.GetInputEdgeDescription(1).tensorDataType;

        // broadcast y_scale and y_zero_point to output shape
        m_inputTensorDescs[0] = TensorDesc(
            kernelCreationContext.GetInputEdgeDescription(0).tensorDataType,
            outputShape,
            kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0),
            TensorAxis::DoNotCoerce,
            TensorAxis::W,
            TensorAxis::RightAligned,
            NchwDimensionCount, // minDimensionCount
            0 // guaranteedBaseOffsetAlignment
        );

        m_inputTensorDescs[1] = TensorDesc(
            kernelCreationContext.GetInputEdgeDescription(1).tensorDataType,
            outputShape,
            kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(1),
            TensorAxis::DoNotCoerce,
            TensorAxis::W,
            TensorAxis::RightAligned,
            NchwDimensionCount, // minDimensionCount
            0 // guaranteedBaseOffsetAlignment
        );

        // Validate input tensors
        for (uint32_t input_index = 0; input_index < input_count; ++input_index)
        {
            auto tuple_start = 2 + input_index * 3;
            // broadcast x_scale and x_zero_point to shape of corresponding x
            auto xScaleDataType = kernelCreationContext.GetInputEdgeDescription(tuple_start + 1).tensorDataType;
            auto xZeroPointDataType = kernelCreationContext.GetInputEdgeDescription(tuple_start + 2).tensorDataType;
            ML_CHECK_VALID_ARGUMENT(xScaleDataType == yScaleDataType, "Input scale is not float");
            m_inputTensorDescs[tuple_start + 1] = TensorDesc(
                kernelCreationContext.GetInputEdgeDescription(tuple_start + 1).tensorDataType,
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tuple_start),
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tuple_start + 1),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );

            ML_CHECK_VALID_ARGUMENT(xZeroPointDataType == yZeroPointDataType, "Wrong input type encountered for zero point");
            m_inputTensorDescs[tuple_start + 2] = TensorDesc(
                kernelCreationContext.GetInputEdgeDescription(tuple_start + 2).tensorDataType,
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tuple_start),
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tuple_start + 2),
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
        std::vector<TensorDesc> intermediateOutputTensorDescs(input_count);
        std::vector<DML_TENSOR_DESC> namedDequantizeOperatorDescs(input_count);
        std::vector<DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC> dequantizeOperatorDescs(input_count);
        std::vector<DML_OPERATOR_DESC> dmlOpDesc(input_count);
        std::vector<const DML_OPERATOR_DESC*> opDescs = {};
        for (uint32_t input_index = 0; input_index < input_count; ++input_index)
        {
            auto tuple_start = 2 + input_index * 3;
            intermediateOutputTensorDescs[input_index] = TensorDesc(
                MLOperatorTensorDataType::Float,
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tuple_start),
                kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(tuple_start),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment)
            );
            namedDequantizeOperatorDescs[input_index] = intermediateOutputTensorDescs[input_index].GetDmlDesc();

            dequantizeOperatorDescs[input_index].InputTensor = &inputDescs[tuple_start];
            dequantizeOperatorDescs[input_index].ScaleTensor = &inputDescs[tuple_start + 1];
            dequantizeOperatorDescs[input_index].ZeroPointTensor = &inputDescs[tuple_start + 2];
            dequantizeOperatorDescs[input_index].OutputTensor = &namedDequantizeOperatorDescs[input_index];

            dmlOpDesc[input_index] = {DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR, &dequantizeOperatorDescs[input_index]};
            opDescs.push_back(&dmlOpDesc[input_index]);
        }

        TensorDesc joinOutputTensorDesc = TensorDesc(
                MLOperatorTensorDataType::Float,
                outputShape,
                outputShape,
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment)
            );
        DML_TENSOR_DESC namedJoinOutputTensorDesc = joinOutputTensorDesc.GetDmlDesc();

        DML_JOIN_OPERATOR_DESC joinDesc = {};
        joinDesc.InputCount = gsl::narrow_cast<uint32_t>(namedDequantizeOperatorDescs.size());
        joinDesc.InputTensors = namedDequantizeOperatorDescs.data();
        joinDesc.OutputTensor = &namedJoinOutputTensorDesc;
        joinDesc.Axis = dmlAxis;

        const DML_OPERATOR_DESC opJoinDesc{DML_OPERATOR_JOIN, &joinDesc};
        opDescs.push_back(&opJoinDesc);

        DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC quantizeOperatorDesc = {};
        quantizeOperatorDesc.InputTensor = joinDesc.OutputTensor;
        quantizeOperatorDesc.ScaleTensor = &inputDescs[0];
        quantizeOperatorDesc.ZeroPointTensor = &inputDescs[1];
        quantizeOperatorDesc.OutputTensor = &outputDescs[0];
        const DML_OPERATOR_DESC opQuantizeDesc{DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR, &quantizeOperatorDesc};
        opDescs.push_back(&opQuantizeDesc);

        MLOperatorGraphDesc operatorGraphDesc = {};
        operatorGraphDesc.nodeCount = static_cast<uint32_t>(opDescs.size());
        operatorGraphDesc.nodesAsOpDesc = opDescs.data();


        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        for (uint32_t input_index = 0; input_index < input_count; ++input_index)
        {
            auto tuple_start = 2 + input_index * 3;
            for (auto edge_index = 0; edge_index < 3; ++edge_index)
            {
                DML_INPUT_GRAPH_EDGE_DESC inputEdge = {};
                inputEdge.GraphInputIndex = tuple_start + edge_index; // OnnxInputIndex and DmlInputIndex are identity for QLinearSigmoid
                inputEdge.ToNodeIndex = input_index;
                inputEdge.ToNodeInputIndex = edge_index;
                inputEdges.push_back(inputEdge);
            }
        }

        DML_INPUT_GRAPH_EDGE_DESC yScaleInputEdge = {};
        yScaleInputEdge.GraphInputIndex = 0; // Y_scale
        yScaleInputEdge.ToNodeIndex = operatorGraphDesc.nodeCount - 1; // To the last node Quantize
        yScaleInputEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(yScaleInputEdge);

        DML_INPUT_GRAPH_EDGE_DESC yZeroPointInputEdge = {};
        yZeroPointInputEdge.GraphInputIndex = 1; // Y_scale
        yZeroPointInputEdge.ToNodeIndex = operatorGraphDesc.nodeCount - 1; // To the last node Quantize
        yZeroPointInputEdge.ToNodeInputIndex = 2;
        inputEdges.push_back(yZeroPointInputEdge);

        operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
        operatorGraphDesc.inputEdges = inputEdges.data();

        // set intermediate edges
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        for (uint32_t input_index = 0; input_index < input_count; ++input_index)
        {
            DML_INTERMEDIATE_GRAPH_EDGE_DESC dequantizeToJoinEdge = {};
            dequantizeToJoinEdge.FromNodeIndex = input_index;
            dequantizeToJoinEdge.FromNodeOutputIndex = 0;
            dequantizeToJoinEdge.ToNodeIndex = operatorGraphDesc.nodeCount - 2; // The second last node Join
            dequantizeToJoinEdge.ToNodeInputIndex = input_index;
            intermediateEdges.push_back(dequantizeToJoinEdge);
        }

        DML_INTERMEDIATE_GRAPH_EDGE_DESC joinToQuantizeEdge = {};
        joinToQuantizeEdge.FromNodeIndex = operatorGraphDesc.nodeCount - 2;
        joinToQuantizeEdge.FromNodeOutputIndex = 0;
        joinToQuantizeEdge.ToNodeIndex = operatorGraphDesc.nodeCount - 1; // The second last node Join
        joinToQuantizeEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(joinToQuantizeEdge);

        operatorGraphDesc.intermediateEdgeCount = gsl::narrow_cast<uint32_t>(intermediateEdges.size());
        operatorGraphDesc.intermediateEdges = intermediateEdges.data();

        // set the output edges
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
        outputEdge.FromNodeIndex = operatorGraphDesc.nodeCount - 1;
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

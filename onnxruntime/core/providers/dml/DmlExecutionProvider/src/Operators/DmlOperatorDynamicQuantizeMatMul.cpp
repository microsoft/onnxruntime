// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
// DynamicQuantizeMatMul = MatrixMultiplyIntegerToFloat(DynamicQuantizeLinear(A), B)
class DmlOperatorDynamicQuantizeMatMul : public DmlOperator
{
    // This order matches the ONNX schema.
    enum OnnxInputIndex
    {
        A, // Input
        B,
        B_scale,
        B_zero_point,
        Bias,
        Count,
    };

public:
    DmlOperatorDynamicQuantizeMatMul(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        DmlOperator::Initialize(kernelCreationContext);

        const bool hasBias = kernelCreationContext.IsInputValid(OnnxInputIndex::Bias);
        const bool hasBZP = kernelCreationContext.IsInputValid(OnnxInputIndex::B_zero_point);

        // Broadcast Bias tensor to the shape of the output tensor.
        if (hasBias)
        {
            m_inputTensorDescs[OnnxInputIndex::Bias] = CreateTensorDescFromInput(
                kernelCreationContext,
                OnnxInputIndex::Bias,
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                kernelCreationContext.GetTensorShapeDescription().GetOutputTensorShape(0)
            );
        }
        MLOperatorTensorDataType BDatatype = kernelCreationContext.GetInputEdgeDescription(OnnxInputIndex::B).tensorDataType;

        std::vector<uint32_t> ATensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(OnnxInputIndex::A);
        std::vector<uint32_t> ExpectedAScaleTensorShape = {1, 1, 1, 1};
        std::vector<uint32_t> ExpectedAZeroPointTensorShape = {1, 1, 1, 1};

        //  output edges between DynQL and MMItoFloat node
        TensorDesc intermediateQuantizedATensorDesc = TensorDesc(
                BDatatype,
                gsl::make_span(ATensorShape),
                gsl::make_span(ATensorShape),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );

        TensorDesc intermediateQuantizedAScaleTensorDesc = TensorDesc(
                MLOperatorTensorDataType::Float,
                gsl::make_span(ExpectedAScaleTensorShape),
                gsl::make_span(ExpectedAScaleTensorShape),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );

        TensorDesc intermediateQuantizedAZeroPointTensorDesc = TensorDesc(
                BDatatype,
                gsl::make_span(ExpectedAZeroPointTensorShape),
                gsl::make_span(ExpectedAZeroPointTensorShape),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );

        DML_TENSOR_DESC namedIntermediateQuantizedATensorDesc = intermediateQuantizedATensorDesc.GetDmlDesc();
        DML_TENSOR_DESC namedIntermediateQuantizedAScaleTensorDesc = intermediateQuantizedAScaleTensorDesc.GetDmlDesc();
        DML_TENSOR_DESC namedIntermediateQuantizedAZeroPointTensorDesc = intermediateQuantizedAZeroPointTensorDesc.GetDmlDesc();

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_DYNAMIC_QUANTIZE_LINEAR_OPERATOR_DESC dynamicQuantizeLinearOperatorDesc = {};
        dynamicQuantizeLinearOperatorDesc.InputTensor = &inputDescs[OnnxInputIndex::A];
        dynamicQuantizeLinearOperatorDesc.OutputTensor = &namedIntermediateQuantizedATensorDesc;
        dynamicQuantizeLinearOperatorDesc.OutputScaleTensor = &namedIntermediateQuantizedAScaleTensorDesc;
        dynamicQuantizeLinearOperatorDesc.OutputZeroPointTensor = &namedIntermediateQuantizedAZeroPointTensorDesc;

        const DML_OPERATOR_DESC opDesc1{DML_OPERATOR_DYNAMIC_QUANTIZE_LINEAR, &dynamicQuantizeLinearOperatorDesc};

        DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC matrixMultiplyIntergerToFloatOperatorDesc = {};
        matrixMultiplyIntergerToFloatOperatorDesc.ATensor = dynamicQuantizeLinearOperatorDesc.OutputTensor;
        matrixMultiplyIntergerToFloatOperatorDesc.AScaleTensor = dynamicQuantizeLinearOperatorDesc.OutputScaleTensor;
        matrixMultiplyIntergerToFloatOperatorDesc.AZeroPointTensor = dynamicQuantizeLinearOperatorDesc.OutputZeroPointTensor;
        matrixMultiplyIntergerToFloatOperatorDesc.BTensor = &inputDescs[OnnxInputIndex::B];
        matrixMultiplyIntergerToFloatOperatorDesc.BScaleTensor = &inputDescs[OnnxInputIndex::B_scale];
        matrixMultiplyIntergerToFloatOperatorDesc.BZeroPointTensor = hasBZP? &inputDescs[OnnxInputIndex::B_zero_point] : nullptr;
        matrixMultiplyIntergerToFloatOperatorDesc.BiasTensor = hasBias? &inputDescs[OnnxInputIndex::Bias] : nullptr;
        matrixMultiplyIntergerToFloatOperatorDesc.OutputTensor = &outputDescs[0];

        const DML_OPERATOR_DESC opDesc2{ static_cast<DML_OPERATOR_TYPE>(DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT), &matrixMultiplyIntergerToFloatOperatorDesc};

        MLOperatorGraphDesc operatorGraphDesc = {};
        std::vector<const DML_OPERATOR_DESC*> opDescs{&opDesc1, &opDesc2};
        operatorGraphDesc.nodeCount = static_cast<uint32_t>(opDescs.size());
        operatorGraphDesc.nodesAsOpDesc = opDescs.data();

        // set input edges
        std::pair<uint32_t, uint32_t> nodeToNodeInputIndex[OnnxInputIndex::Count] {{0, 0}, {1, 3}, {1, 4}, {1, 5}, {1, 6}};
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        for (uint32_t inputIndex = 0; inputIndex < OnnxInputIndex::Count; inputIndex++)
        {
            if (inputIndex == OnnxInputIndex::B_zero_point && !hasBZP) continue;
            if (inputIndex == OnnxInputIndex::Bias && !hasBias) continue;
            DML_INPUT_GRAPH_EDGE_DESC inputEdge = {};
            inputEdge.GraphInputIndex = inputIndex; // OnnxInputIndex and DmlInputIndex are identity for QLinearSigmoid
            inputEdge.ToNodeIndex = nodeToNodeInputIndex[inputIndex].first;
            inputEdge.ToNodeInputIndex = nodeToNodeInputIndex[inputIndex].second;
            inputEdges.push_back(inputEdge);
        }
        operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
        operatorGraphDesc.inputEdges = inputEdges.data();

        // set intermediate edges
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;

        DML_INTERMEDIATE_GRAPH_EDGE_DESC dynQLToMMItofloatEdge1 = {};
        dynQLToMMItofloatEdge1.FromNodeIndex = 0;
        dynQLToMMItofloatEdge1.FromNodeOutputIndex = 0;
        dynQLToMMItofloatEdge1.ToNodeIndex = 1;
        dynQLToMMItofloatEdge1.ToNodeInputIndex = 0;
        intermediateEdges.push_back(dynQLToMMItofloatEdge1);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC dynQLToMMItofloatEdge2 = {};
        dynQLToMMItofloatEdge2.FromNodeIndex = 0;
        dynQLToMMItofloatEdge2.FromNodeOutputIndex = 1;
        dynQLToMMItofloatEdge2.ToNodeIndex = 1;
        dynQLToMMItofloatEdge2.ToNodeInputIndex = 1;
        intermediateEdges.push_back(dynQLToMMItofloatEdge2);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC dynQLToMMItofloatEdge3 = {};
        dynQLToMMItofloatEdge3.FromNodeIndex = 0;
        dynQLToMMItofloatEdge3.FromNodeOutputIndex = 2;
        dynQLToMMItofloatEdge3.ToNodeIndex = 1;
        dynQLToMMItofloatEdge3.ToNodeInputIndex = 2;
        intermediateEdges.push_back(dynQLToMMItofloatEdge3);

        operatorGraphDesc.intermediateEdgeCount = gsl::narrow_cast<uint32_t>(intermediateEdges.size());
        operatorGraphDesc.intermediateEdges = intermediateEdges.data();

        // set the output edges
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
        outputEdge.FromNodeIndex = 1;
        outputEdge.FromNodeOutputIndex = 0;
        outputEdge.GraphOutputIndex = 0;
        outputEdges.push_back(outputEdge);
        operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
        operatorGraphDesc.outputEdges = outputEdges.data();

        SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(DynamicQuantizeMatMul, DmlOperatorDynamicQuantizeMatMul);
} // namespace Dml

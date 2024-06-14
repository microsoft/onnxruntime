// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
// QLinearSigmoid = Dequantize + Sigmoid + Quantize
// This kernel is the first usage of graph based implementation
class DmlOperatorQLinearSigmoid : public DmlOperator
{
    // This order matches the ONNX schema.
    enum OnnxInputIndex
    {
        X, // Input
        X_scale,
        X_zero_point,
        Y_scale,
        Y_zero_point,
        Count,
    };

public:
    DmlOperatorQLinearSigmoid(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        DmlOperator::Initialize(kernelCreationContext);

        std::vector<uint32_t> outputShape = kernelCreationContext.GetTensorShapeDescription().GetOutputTensorShape(0);
        const uint32_t outputShapeDimCount = gsl::narrow_cast<uint32_t>(outputShape.size());

        uint32_t axis = 0;
        // Explicitly reshape each of the inputs after the first input (scale and zero point tensors).
        for (uint32_t index = OnnxInputIndex::X_scale, inputCount = gsl::narrow_cast<uint32_t>(OnnxInputIndex::Count); index < inputCount; ++index)
        {
            auto edgeDesc = kernelCreationContext.GetInputEdgeDescription(index);
            assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);

            // Fix up the the tensor shape by filling with trailing ones. So input[2,3] with axis=0 and scale[2]
            // becomes scale[2,1], so that broadcasting works correctly.
            std::vector<uint32_t> inputTensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(index);

            // If the input tensor is a 1D vector, then extra massaging is needed to project their
            // 1D vectors back to the full shape for broadcasting along the given axis.
            // The 1D vector should have a length equal to the output tensor's dimension on that axis.
            if (inputTensorShape.size() == 1 && inputTensorShape != outputShape)
            {
                ML_CHECK_VALID_ARGUMENT(axis < outputShapeDimCount);
                uint32_t broadcastAxisLength = outputShape[axis];
                ML_CHECK_VALID_ARGUMENT(inputTensorShape[0] == broadcastAxisLength);
                inputTensorShape.insert(inputTensorShape.begin(), axis, 1);
                inputTensorShape.insert(inputTensorShape.end(), outputShapeDimCount - 1 - axis, 1);
            }
            // For any other shape (scalar/ND), leave it alone, and the TensorDesc constructor
            // will apply broadcasting with standard elementwise alignment.

            m_inputTensorDescs[index] = TensorDesc(
                edgeDesc.tensorDataType,
                gsl::make_span(outputShape),
                gsl::make_span(inputTensorShape),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );
        }

        //  1. output edge between Dequantize and Sigmoid node
        //  2. input edge between Sigmoid and Quantize node
        TensorDesc intermediateOutputTensorDesc = TensorDesc(
                MLOperatorTensorDataType::Float,
                gsl::make_span(outputShape),
                gsl::make_span(outputShape),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );
        DML_TENSOR_DESC namedIntermediateOutputTensorDesc = intermediateOutputTensorDesc.GetDmlDesc();

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC dequantizeOperatorDesc = {};
        dequantizeOperatorDesc.InputTensor = &inputDescs[OnnxInputIndex::X];
        dequantizeOperatorDesc.ScaleTensor = &inputDescs[OnnxInputIndex::X_scale];
        dequantizeOperatorDesc.ZeroPointTensor = &inputDescs[OnnxInputIndex::X_zero_point];
        dequantizeOperatorDesc.OutputTensor = &namedIntermediateOutputTensorDesc;

        TryConvertTensorToBroadcastScalar(kernelCreationContext, dequantizeOperatorDesc.ScaleTensor,     OnnxInputIndex::X_scale);
        TryConvertTensorToBroadcastScalar(kernelCreationContext, dequantizeOperatorDesc.ZeroPointTensor, OnnxInputIndex::X_zero_point);
        
        const DML_OPERATOR_DESC opDesc1{DML_OPERATOR_ELEMENT_WISE_DEQUANTIZE_LINEAR, &dequantizeOperatorDesc};

        DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoidOperatorDesc = {};
        sigmoidOperatorDesc.InputTensor = dequantizeOperatorDesc.OutputTensor;
        sigmoidOperatorDesc.OutputTensor = dequantizeOperatorDesc.OutputTensor;
        const DML_OPERATOR_DESC opDesc2{DML_OPERATOR_ACTIVATION_SIGMOID, &sigmoidOperatorDesc};

        DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC quantizeOperatorDesc = {};
        quantizeOperatorDesc.InputTensor = sigmoidOperatorDesc.OutputTensor;
        quantizeOperatorDesc.ScaleTensor = &inputDescs[OnnxInputIndex::Y_scale];
        quantizeOperatorDesc.ZeroPointTensor = &inputDescs[OnnxInputIndex::Y_zero_point];
        quantizeOperatorDesc.OutputTensor = &outputDescs[0];
        
        TryConvertTensorToBroadcastScalar(kernelCreationContext, quantizeOperatorDesc.ScaleTensor,     OnnxInputIndex::Y_scale);
        TryConvertTensorToBroadcastScalar(kernelCreationContext, quantizeOperatorDesc.ZeroPointTensor, OnnxInputIndex::Y_zero_point);

        const DML_OPERATOR_DESC opDesc3{DML_OPERATOR_ELEMENT_WISE_QUANTIZE_LINEAR, &quantizeOperatorDesc};

        MLOperatorGraphDesc operatorGraphDesc = {};
        operatorGraphDesc.nodeCount = 3;
        std::vector<const DML_OPERATOR_DESC*> opDescs{&opDesc1, &opDesc2, &opDesc3};
        operatorGraphDesc.nodes = opDescs.data();

        // set input edges
        std::pair<uint32_t, uint32_t> nodeToNodeInputIndex[5] {{0, 0}, {0, 1}, {0, 2}, {2, 1}, {2, 2}};
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges(OnnxInputIndex::Count);
        for (uint32_t inputIndex = 0; inputIndex < OnnxInputIndex::Count; inputIndex++)
        {
            DML_INPUT_GRAPH_EDGE_DESC inputEdge = {};
            inputEdge.GraphInputIndex = inputIndex; // OnnxInputIndex and DmlInputIndex are identity for QLinearSigmoid
            inputEdge.ToNodeIndex = nodeToNodeInputIndex[inputIndex].first;
            inputEdge.ToNodeInputIndex = nodeToNodeInputIndex[inputIndex].second;
            inputEdges[inputIndex] = inputEdge;
        }
        operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
        operatorGraphDesc.inputEdges = inputEdges.data();

        // set intermediate edges
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        
        DML_INTERMEDIATE_GRAPH_EDGE_DESC dequantizeToSigmoidEdge = {};
        dequantizeToSigmoidEdge.FromNodeIndex = 0;
        dequantizeToSigmoidEdge.FromNodeOutputIndex = 0;
        dequantizeToSigmoidEdge.ToNodeIndex = 1;
        dequantizeToSigmoidEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(dequantizeToSigmoidEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC sigmoidToQuantizeEdge = {};
        sigmoidToQuantizeEdge.FromNodeIndex = 1;
        sigmoidToQuantizeEdge.FromNodeOutputIndex = 0;
        sigmoidToQuantizeEdge.ToNodeIndex = 2;
        sigmoidToQuantizeEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(sigmoidToQuantizeEdge);

        operatorGraphDesc.intermediateEdgeCount = gsl::narrow_cast<uint32_t>(intermediateEdges.size());
        operatorGraphDesc.intermediateEdges = intermediateEdges.data();

        // set the output edges
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
        outputEdge.FromNodeIndex = 2;
        outputEdge.FromNodeOutputIndex = 0;
        outputEdge.GraphOutputIndex = 0;
        outputEdges.push_back(outputEdge);
        operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
        operatorGraphDesc.outputEdges = outputEdges.data();

        SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelCreationContext);
    }
};

void CALLBACK QueryQLinearSigmoid(IMLOperatorSupportQueryContextPrivate* context, /*out*/ bool* isSupported)
{
    *isSupported = false;
    // Right now the contract is if optional input tensors (like x_zero_point, y_zero_point) are
    // not present, then fallback to CPU because DML Quantize_Linear and Dequantize_Linear does not support 
    // optionality of the zero_point tensor. BUG: https://microsoft.visualstudio.com/OS/_queries/edit/41599005
    if (context->GetInputCount() < 5)
    {
        return;
    }

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(QLinearSigmoid, DmlOperatorQLinearSigmoid);
} // namespace Dml
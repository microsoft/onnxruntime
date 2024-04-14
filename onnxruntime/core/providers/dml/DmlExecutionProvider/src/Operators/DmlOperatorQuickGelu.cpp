// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorQuickGelu : public DmlOperator
{
public:
    DmlOperatorQuickGelu(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);
        DmlOperator::Initialize(kernelCreationContext);

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs.size() == 1);
        ML_CHECK_VALID_ARGUMENT(m_outputTensorDescs.size() == 1);
        const float alpha = kernelCreationContext.GetAttribute<float>(AttrName::Alpha);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        // 1. Apply the alpha if needed
        DML_SCALE_BIAS scaleBias{alpha, 0.0f};
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC mulAlphaDesc{};
        if (alpha != 1.0f)
        {
            mulAlphaDesc.InputTensor = &inputDescs[0];
            mulAlphaDesc.OutputTensor = &inputDescs[0];
            mulAlphaDesc.ScaleBias = &scaleBias;
        }
        DML_OPERATOR_DESC dmlMulAlphaDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &mulAlphaDesc };

        // 2. Apply the sigmoid activation function
        DML_ACTIVATION_SIGMOID_OPERATOR_DESC sigmoidDesc{};
        sigmoidDesc.InputTensor = &inputDescs[0];
        sigmoidDesc.OutputTensor = &inputDescs[0];
        DML_OPERATOR_DESC dmlSigmoidDesc = { DML_OPERATOR_ACTIVATION_SIGMOID, &sigmoidDesc };

        // 3. Multiply the sigmoid result with the original input
        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC multiplyDesc{};
        multiplyDesc.ATensor = &inputDescs[0];
        multiplyDesc.BTensor = &inputDescs[0];
        multiplyDesc.OutputTensor = &inputDescs[0];
        DML_OPERATOR_DESC dmlMultiplyDesc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &multiplyDesc };

        enum NodeIndex
        {
            sigmoidNodeIndex,
            multiplyNodeIndex,
            mulAlphaNodeIndex,
            nodeCount,
        };

        // Construct the graph
        std::vector<const DML_OPERATOR_DESC*> opDescs;
        opDescs.reserve(3);
        opDescs.push_back(&dmlSigmoidDesc);
        opDescs.push_back(&dmlMultiplyDesc);

        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        inputEdges.reserve(2);

        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        intermediateEdges.reserve(2);

        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        outputEdges.reserve(1);

        if (alpha != 1.0f)
        {
            opDescs.push_back(&dmlMulAlphaDesc);

            DML_INPUT_GRAPH_EDGE_DESC inputToMulAlphaEdge{};
            inputToMulAlphaEdge.GraphInputIndex = 0;
            inputToMulAlphaEdge.ToNodeIndex = mulAlphaNodeIndex;
            inputToMulAlphaEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(inputToMulAlphaEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC mulAlphaToSigmoidEdge{};
            mulAlphaToSigmoidEdge.FromNodeIndex = mulAlphaNodeIndex;
            mulAlphaToSigmoidEdge.FromNodeOutputIndex = 0;
            mulAlphaToSigmoidEdge.ToNodeIndex = sigmoidNodeIndex;
            mulAlphaToSigmoidEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(mulAlphaToSigmoidEdge);
        }
        else
        {
            DML_INPUT_GRAPH_EDGE_DESC inputToSigmoidEdge{};
            inputToSigmoidEdge.GraphInputIndex = 0;
            inputToSigmoidEdge.ToNodeIndex = sigmoidNodeIndex;
            inputToSigmoidEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(inputToSigmoidEdge);
        }

        DML_INPUT_GRAPH_EDGE_DESC inputToMultiplyEdge{};
        inputToMultiplyEdge.GraphInputIndex = 0;
        inputToMultiplyEdge.ToNodeIndex = multiplyNodeIndex;
        inputToMultiplyEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(inputToMultiplyEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC sigmoidToMultiplyEdge{};
        sigmoidToMultiplyEdge.FromNodeIndex = sigmoidNodeIndex;
        sigmoidToMultiplyEdge.FromNodeOutputIndex = 0;
        sigmoidToMultiplyEdge.ToNodeIndex = multiplyNodeIndex;
        sigmoidToMultiplyEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(sigmoidToMultiplyEdge);

        DML_OUTPUT_GRAPH_EDGE_DESC multiplyToOutputEdge{};
        multiplyToOutputEdge.FromNodeIndex = multiplyNodeIndex;
        multiplyToOutputEdge.FromNodeOutputIndex = 0;
        multiplyToOutputEdge.GraphOutputIndex = 0;
        outputEdges.push_back(multiplyToOutputEdge);

        MLOperatorGraphDesc operatorGraphDesc = {};
        operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
        operatorGraphDesc.inputEdges = inputEdges.data();
        operatorGraphDesc.intermediateEdgeCount = gsl::narrow_cast<uint32_t>(intermediateEdges.size());
        operatorGraphDesc.intermediateEdges = intermediateEdges.data();
        operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
        operatorGraphDesc.outputEdges = outputEdges.data();
        operatorGraphDesc.nodeCount = gsl::narrow_cast<uint32_t>(opDescs.size());
        operatorGraphDesc.nodes = opDescs.data();
        SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(QuickGelu, DmlOperatorQuickGelu);

} // namespace Dml

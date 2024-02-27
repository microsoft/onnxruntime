// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorBiasSplitGelu : public DmlOperator
{
public:
    DmlOperatorBiasSplitGelu(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 2);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);

        // Broadcast bias to have the same dimensions as the input
        std::vector<uint32_t> inputTensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        DmlOperator::Initialize(kernelCreationContext, std::nullopt, std::nullopt, inputTensorShape);

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs.size() == 2);
        ML_CHECK_VALID_ARGUMENT(m_outputTensorDescs.size() == 1);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[0].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[1].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[0].GetSizes().back() == m_inputTensorDescs[1].GetSizes().back());
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[0].GetSizes().back() % 2 == 0);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        // 1. Add the bias
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC addDesc{};
        addDesc.ATensor = &inputDescs[0];
        addDesc.BTensor = &inputDescs[1];
        addDesc.OutputTensor = &inputDescs[0];
        DML_OPERATOR_DESC dmlAddDesc = { DML_OPERATOR_ELEMENT_WISE_ADD, &addDesc };

        // 2. Split the tensor in 2
        const std::array<DML_TENSOR_DESC, 2> splitOutputTensors = {outputDescs[0], outputDescs[0]};
        DML_SPLIT_OPERATOR_DESC splitDesc{};
        splitDesc.InputTensor = &inputDescs[0];
        splitDesc.OutputTensors = splitOutputTensors.data();
        splitDesc.OutputCount = gsl::narrow_cast<uint32_t>(splitOutputTensors.size());
        splitDesc.Axis = m_inputTensorDescs[0].GetDimensionCount() - 1;
        DML_OPERATOR_DESC dmlSplitDesc = { DML_OPERATOR_SPLIT, &splitDesc };

        // 3. Apply Gelu to the second tensor
        DML_ACTIVATION_GELU_OPERATOR_DESC geluDesc{};
        geluDesc.InputTensor = &outputDescs[0];
        geluDesc.OutputTensor = &outputDescs[0];
        DML_OPERATOR_DESC dmlGeluDesc = { DML_OPERATOR_ACTIVATION_GELU, &geluDesc };

        // 4. Multiply the 2 tensors together
        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC multiplyDesc{};
        multiplyDesc.ATensor = &outputDescs[0];
        multiplyDesc.BTensor = &outputDescs[0];
        multiplyDesc.OutputTensor = &outputDescs[0];
        DML_OPERATOR_DESC dmlMultiplyDesc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &multiplyDesc };

        enum NodeIndex
        {
            addNodeIndex,
            splitNodeIndex,
            geluNodeIndex,
            multiplyNodeIndex,
            nodeCount,
        };

        // Construct the graph
        std::vector<const DML_OPERATOR_DESC*> opDescs(nodeCount);
        opDescs[addNodeIndex] = &dmlAddDesc;
        opDescs[splitNodeIndex] = &dmlSplitDesc;
        opDescs[geluNodeIndex] = &dmlGeluDesc;
        opDescs[multiplyNodeIndex] = &dmlMultiplyDesc;

        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        inputEdges.reserve(2);

        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        intermediateEdges.reserve(4);

        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        outputEdges.reserve(1);

        DML_INPUT_GRAPH_EDGE_DESC inputToAddEdge{};
        inputToAddEdge.GraphInputIndex = 0;
        inputToAddEdge.ToNodeIndex = addNodeIndex;
        inputToAddEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(inputToAddEdge);

        DML_INPUT_GRAPH_EDGE_DESC biasToAddEdge{};
        biasToAddEdge.GraphInputIndex = 1;
        biasToAddEdge.ToNodeIndex = addNodeIndex;
        biasToAddEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(biasToAddEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC addToSplitEdge{};
        addToSplitEdge.FromNodeIndex = addNodeIndex;
        addToSplitEdge.FromNodeOutputIndex = 0;
        addToSplitEdge.ToNodeIndex = splitNodeIndex;
        addToSplitEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(addToSplitEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC splitToGeluEdge{};
        splitToGeluEdge.FromNodeIndex = splitNodeIndex;
        splitToGeluEdge.FromNodeOutputIndex = 1;
        splitToGeluEdge.ToNodeIndex = geluNodeIndex;
        splitToGeluEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(splitToGeluEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC splitToMultiplyEdge{};
        splitToMultiplyEdge.FromNodeIndex = splitNodeIndex;
        splitToMultiplyEdge.FromNodeOutputIndex = 0;
        splitToMultiplyEdge.ToNodeIndex = multiplyNodeIndex;
        splitToMultiplyEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(splitToMultiplyEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC geluToMultiplyEdge{};
        geluToMultiplyEdge.FromNodeIndex = geluNodeIndex;
        geluToMultiplyEdge.FromNodeOutputIndex = 0;
        geluToMultiplyEdge.ToNodeIndex = multiplyNodeIndex;
        geluToMultiplyEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(geluToMultiplyEdge);

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

DML_OP_DEFINE_CREATION_FUNCTION(BiasSplitGelu, DmlOperatorBiasSplitGelu);

} // namespace Dml

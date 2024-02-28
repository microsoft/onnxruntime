// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorBiasAdd : public DmlOperator
{
public:
    DmlOperatorBiasAdd(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 3);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);

        // Broadcast bias to have the same dimensions as the input
        std::vector<uint32_t> inputTensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        DmlOperator::Initialize(kernelCreationContext, std::nullopt, std::nullopt, inputTensorShape);

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs.size() == 3);
        ML_CHECK_VALID_ARGUMENT(m_outputTensorDescs.size() == 1);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[0].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[1].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[2].GetDimensionCount() == 4);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[0].GetSizes() == m_inputTensorDescs[2].GetSizes());

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        // 1. Add the bias
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC addBiasDesc{};
        addBiasDesc.ATensor = &inputDescs[0];
        addBiasDesc.BTensor = &inputDescs[1];
        addBiasDesc.OutputTensor = &inputDescs[0];
        DML_OPERATOR_DESC dmlAddBiasDesc = { DML_OPERATOR_ELEMENT_WISE_ADD, &addBiasDesc };

        // 2. Add the residual inputs
        DML_ELEMENT_WISE_ADD_OPERATOR_DESC addSkipDesc{};
        addSkipDesc.ATensor = &inputDescs[0];
        addSkipDesc.BTensor = &inputDescs[2];
        addSkipDesc.OutputTensor = &inputDescs[0];
        DML_OPERATOR_DESC dmlAddSkipDesc = { DML_OPERATOR_ELEMENT_WISE_ADD, &addSkipDesc };

        enum NodeIndex
        {
            addBiasNodeIndex,
            addSkipNodeIndex,
            nodeCount,
        };

        // Construct the graph
        std::vector<const DML_OPERATOR_DESC*> opDescs(nodeCount);
        opDescs[addBiasNodeIndex] = &dmlAddBiasDesc;
        opDescs[addSkipNodeIndex] = &dmlAddSkipDesc;

        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        inputEdges.reserve(3);

        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        intermediateEdges.reserve(1);

        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        outputEdges.reserve(1);

        DML_INPUT_GRAPH_EDGE_DESC inputToAddBiasEdge{};
        inputToAddBiasEdge.GraphInputIndex = 0;
        inputToAddBiasEdge.ToNodeIndex = addBiasNodeIndex;
        inputToAddBiasEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(inputToAddBiasEdge);

        DML_INPUT_GRAPH_EDGE_DESC biasToAddBiasEdge{};
        biasToAddBiasEdge.GraphInputIndex = 1;
        biasToAddBiasEdge.ToNodeIndex = addBiasNodeIndex;
        biasToAddBiasEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(biasToAddBiasEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC addBiasToAddSkipEdge{};
        addBiasToAddSkipEdge.FromNodeIndex = addBiasNodeIndex;
        addBiasToAddSkipEdge.FromNodeOutputIndex = 0;
        addBiasToAddSkipEdge.ToNodeIndex = addSkipNodeIndex;
        addBiasToAddSkipEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(addBiasToAddSkipEdge);

        DML_INPUT_GRAPH_EDGE_DESC skipToAddSkipEdge{};
        skipToAddSkipEdge.GraphInputIndex = 2;
        skipToAddSkipEdge.ToNodeIndex = addSkipNodeIndex;
        skipToAddSkipEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(skipToAddSkipEdge);

        DML_OUTPUT_GRAPH_EDGE_DESC addSkipToOutputEdge{};
        addSkipToOutputEdge.FromNodeIndex = addSkipNodeIndex;
        addSkipToOutputEdge.FromNodeOutputIndex = 0;
        addSkipToOutputEdge.GraphOutputIndex = 0;
        outputEdges.push_back(addSkipToOutputEdge);

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

DML_OP_DEFINE_CREATION_FUNCTION(BiasAdd, DmlOperatorBiasAdd);

} // namespace Dml

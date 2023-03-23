// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorGroupNorm : public DmlOperator
{
public:
    DmlOperatorGroupNorm(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 3);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);

        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1, 2};
        DmlOperator::Initialize(kernelCreationContext, kernelInputIndices);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);
        const int activation = kernelCreationContext.GetAttribute<int>(AttrName::Activation);
        const int groups = kernelCreationContext.GetAttribute<int>(AttrName::Activation);

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs.size() == 3);

        const auto inputTensorShape = m_inputTensorDescs[0].GetSizes();
        ML_CHECK_VALID_ARGUMENT(inputTensorShape.size() == 4);

        const auto gammaTensorShape = m_inputTensorDescs[1].GetSizes();
        ML_CHECK_VALID_ARGUMENT(gammaTensorShape.size() == 1);

        const auto betaTensorShape = m_inputTensorDescs[2].GetSizes();
        ML_CHECK_VALID_ARGUMENT(betaTensorShape.size() == 1);

        // Data is in NHWC format
        const uint32_t batch = inputTensorShape[0];
        const uint32_t height = inputTensorShape[1];
        const uint32_t width = inputTensorShape[2];
        const uint32_t channels = inputTensorShape[3];
        ML_CHECK_VALID_ARGUMENT(gammaTensorShape[0] == channels);
        ML_CHECK_VALID_ARGUMENT(betaTensorShape[0] == channels);
        ML_CHECK_VALID_ARGUMENT(channels % groups == 0);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[1].GetDmlDataType() == m_inputTensorDescs[2].GetDmlDataType());

        // DML doesn't support grouped MVN, so split the data and perform MVN separately on each one of them
        const uint32_t splitChannels = channels / groups;

        // Use NHWC strides with NCHW sizes to satisfy DML's requirements
        const std::array<uint32_t, 4> inputShape = {batch, channels, height, width};
        const std::array<uint32_t, 4> inputStrides = {channels * height * width, 1, channels * width, channels};
        TensorDesc inputTensorDesc = TensorDesc(m_inputTensorDescs[0].GetDmlDataType(), inputShape, inputStrides);
        const DML_TENSOR_DESC inputDmlTensorDesc = inputTensorDesc.GetDmlDesc();

        const std::array<uint32_t, 4> splitInputShape = {batch, splitChannels, height, width};
        const std::array<uint32_t, 4> splitInputStrides = {splitChannels * height * width, 1, splitChannels * width, splitChannels};
        TensorDesc splitInputTensorDesc = TensorDesc(m_inputTensorDescs[0].GetDmlDataType(), splitInputShape, splitInputStrides);
        const DML_TENSOR_DESC splitInputDmlTensorDesc = splitInputTensorDesc.GetDmlDesc();
        const std::vector<DML_TENSOR_DESC> splitInputTensors(groups, splitInputDmlTensorDesc);

        const std::array<uint32_t, 4> gammaBetaShape = {1, channels, 1, 1};
        const std::array<uint32_t, 4> gammaBetaStrides = {0, 1, 0, 0};
        TensorDesc gammaBetaTensorDesc = TensorDesc(m_inputTensorDescs[1].GetDmlDataType(), gammaBetaShape, gammaBetaStrides);
        const DML_TENSOR_DESC gammaBetaDmlTensorDesc = gammaBetaTensorDesc.GetDmlDesc();

        const std::array<uint32_t, 4> splitGammaBetaShape = {1, splitChannels, 1, 1};
        const std::array<uint32_t, 4> splitGammaBetaStrides = {0, 1, 0, 0};
        TensorDesc splitGammaBetaTensorDesc = TensorDesc(m_inputTensorDescs[1].GetDmlDataType(), splitGammaBetaShape, splitGammaBetaStrides);
        const DML_TENSOR_DESC splitGammaBetaDmlTensorDesc = splitGammaBetaTensorDesc.GetDmlDesc();
        const std::vector<DML_TENSOR_DESC> splitGammaBetaTensors(groups, splitGammaBetaDmlTensorDesc);

        // Split the input data into tensors of `splitChannels` channels
        DML_SPLIT_OPERATOR_DESC splitInputDesc{};
        splitInputDesc.InputTensor = &inputDmlTensorDesc;
        splitInputDesc.OutputCount = gsl::narrow_cast<uint32_t>(splitInputTensors.size());
        splitInputDesc.OutputTensors = splitInputTensors.data();
        splitInputDesc.Axis = 1;
        DML_OPERATOR_DESC dmlSplitInputDesc = { DML_OPERATOR_SPLIT, &splitInputDesc };

        // Split the gamma and beta into tensors of `splitChannels` channels
        DML_SPLIT_OPERATOR_DESC splitGammaBetaDesc{};
        splitGammaBetaDesc.InputTensor = &gammaBetaDmlTensorDesc;
        splitGammaBetaDesc.OutputCount = gsl::narrow_cast<uint32_t>(splitGammaBetaTensors.size());
        splitGammaBetaDesc.OutputTensors = splitGammaBetaTensors.data();
        splitGammaBetaDesc.Axis = 1;
        DML_OPERATOR_DESC dmlSplitGammaBetaDesc = { DML_OPERATOR_SPLIT, &splitGammaBetaDesc };

        // Perform the MVN on the split data
        DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC mvnDesc{};
        mvnDesc.InputTensor = &splitInputDmlTensorDesc;
        mvnDesc.ScaleTensor = &splitGammaBetaDmlTensorDesc;
        mvnDesc.BiasTensor = &splitGammaBetaDmlTensorDesc;
        mvnDesc.OutputTensor = &splitInputDmlTensorDesc;
        mvnDesc.CrossChannel = true;
        mvnDesc.NormalizeVariance = true;
        mvnDesc.Epsilon = epsilon;
        DML_OPERATOR_DESC dmlMvnDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION, &mvnDesc };

        // Finally, join the split MVN results together
        DML_JOIN_OPERATOR_DESC joinDesc{};
        joinDesc.InputCount = gsl::narrow_cast<uint32_t>(splitInputTensors.size());
        joinDesc.InputTensors = splitInputTensors.data();
        joinDesc.OutputTensor = &inputDmlTensorDesc;
        joinDesc.Axis = 1;

        // Construct the graph
        std::vector<const DML_OPERATOR_DESC*> opDescs;
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

        uint32_t currentNodeIndex = 0;

        const uint32_t splitInputNodeIndex = currentNodeIndex++;
        opDescs.push_back(&dmlSplitInputDesc);

        const uint32_t splitGammaNodeIndex = currentNodeIndex++;
        opDescs.push_back(&dmlSplitGammaBetaDesc);

        const uint32_t splitBetaNodeIndex = currentNodeIndex++;
        opDescs.push_back(&dmlSplitGammaBetaDesc);

        DML_INPUT_GRAPH_EDGE_DESC inputEdge{};
        inputEdge.GraphInputIndex = 0;
        inputEdge.ToNodeIndex = splitInputNodeIndex;
        inputEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(inputEdge);

        DML_INPUT_GRAPH_EDGE_DESC gammaEdge{};
        gammaEdge.GraphInputIndex = 1;
        gammaEdge.ToNodeIndex = splitGammaNodeIndex;
        gammaEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(gammaEdge);

        DML_INPUT_GRAPH_EDGE_DESC betaEdge{};
        betaEdge.GraphInputIndex = 2;
        betaEdge.ToNodeIndex = splitBetaNodeIndex;
        betaEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(betaEdge);

        std::vector<uint32_t> mvnNodeIndices;
        mvnNodeIndices.reserve(splitInputTensors.size());
        for (uint32_t splitTensorIndex = 0; splitTensorIndex < splitInputTensors.size(); ++splitTensorIndex)
        {
            const uint32_t mvnNodeIndex = currentNodeIndex++;
            opDescs.push_back(&dmlMvnDesc);
            mvnNodeIndices.push_back(mvnNodeIndex);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC splitInputToMvnEdge = {};
            splitInputToMvnEdge.FromNodeIndex = splitInputNodeIndex;
            splitInputToMvnEdge.FromNodeOutputIndex = splitTensorIndex;
            splitInputToMvnEdge.ToNodeIndex = mvnNodeIndex;
            splitInputToMvnEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(splitInputToMvnEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC splitGammaToMvnEdge = {};
            splitGammaToMvnEdge.FromNodeIndex = splitGammaNodeIndex;
            splitGammaToMvnEdge.FromNodeOutputIndex = splitTensorIndex;
            splitGammaToMvnEdge.ToNodeIndex = mvnNodeIndex;
            splitGammaToMvnEdge.ToNodeInputIndex = 1;
            intermediateEdges.push_back(splitGammaToMvnEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC splitBetaToMvnEdge = {};
            splitBetaToMvnEdge.FromNodeIndex = splitBetaNodeIndex;
            splitBetaToMvnEdge.FromNodeOutputIndex = splitTensorIndex;
            splitBetaToMvnEdge.ToNodeIndex = mvnNodeIndex;
            splitBetaToMvnEdge.ToNodeInputIndex = 2;
            intermediateEdges.push_back(splitBetaToMvnEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC mvnToJoinEdge = {};
            mvnToJoinEdge.FromNodeIndex = splitBetaNodeIndex;
            mvnToJoinEdge.FromNodeOutputIndex = splitTensorIndex;
            mvnToJoinEdge.ToNodeIndex = mvnNodeIndex;
            mvnToJoinEdge.ToNodeInputIndex = 2;
            intermediateEdges.push_back(mvnToJoinEdge);
        }

         opDescs.push_back(&dmlJoinDesc);

        MLOperatorGraphDesc operatorGraphDesc = {};
        operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
        operatorGraphDesc.inputEdges = inputEdges.data();
        operatorGraphDesc.intermediateEdgeCount = gsl::narrow_cast<uint32_t>(intermediateEdges.size());
        operatorGraphDesc.intermediateEdges = intermediateEdges.data();
        operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
        operatorGraphDesc.outputEdges = outputEdges.data();
        operatorGraphDesc.nodeCount = gsl::narrow_cast<uint32_t>(opDescs.size());
        operatorGraphDesc.nodesAsOpDesc = opDescs.data();

        SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(GroupNorm, DmlOperatorGroupNorm);

} // namespace Dml

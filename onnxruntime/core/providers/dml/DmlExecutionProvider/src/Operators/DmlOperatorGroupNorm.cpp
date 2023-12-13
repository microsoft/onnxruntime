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
        DmlOperator::Initialize(kernelCreationContext, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 1);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        const bool channelsLast = kernelCreationContext.GetOptionalAttribute<bool>(AttrName::ChannelsLast, true);
        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);
        const bool activation = gsl::narrow_cast<bool>(kernelCreationContext.GetAttribute<int64_t>(AttrName::Activation));
        const uint32_t groups = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::Groups));

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs.size() == 3);

        const auto inputTensorShape = m_inputTensorDescs[0].GetSizes();
        ML_CHECK_VALID_ARGUMENT(inputTensorShape.size() == 4);

        const auto gammaTensorShape = m_inputTensorDescs[1].GetSizes();
        ML_CHECK_VALID_ARGUMENT(gammaTensorShape.size() == 1);

        const auto betaTensorShape = m_inputTensorDescs[2].GetSizes();
        ML_CHECK_VALID_ARGUMENT(betaTensorShape.size() == 1);

        // Data is in NHWC format
        const uint32_t batch = inputTensorShape[0];
        const uint32_t height = channelsLast ? inputTensorShape[1] : inputTensorShape[2];
        const uint32_t width = channelsLast ? inputTensorShape[2] : inputTensorShape[3];
        const uint32_t channels = channelsLast ? inputTensorShape[3] : inputTensorShape[1];
        ML_CHECK_VALID_ARGUMENT(gammaTensorShape[0] == channels);
        ML_CHECK_VALID_ARGUMENT(betaTensorShape[0] == channels);
        ML_CHECK_VALID_ARGUMENT(channels % groups == 0);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[1].GetDmlDataType() == m_inputTensorDescs[2].GetDmlDataType());
        const uint32_t channelsPerGroup = channels / groups;

        std::array<uint32_t, 4> inputShape;
        std::array<uint32_t, 4> inputStrides;

        if (channelsLast)
        {
            // 1. Reshape the input from [batch, height, width, channels] to [batch, height * width, groups, channelsPerGroup]
            // 2. Stride the reshaped input from [batch, height * width, groups, channelsPerGroup] to [batch, groups, channelsPerGroup, height * width]
            inputShape = {batch, groups, channelsPerGroup, height * width};
            inputStrides = {channelsPerGroup * height * width * groups, channelsPerGroup, 1, groups * channelsPerGroup};
        }
        else
        {
            // Reshape the input from [batch, channels, height, width] to [batch, groups, channelsPerGroup, height * width]
            inputShape = {batch, groups, channelsPerGroup, height * width};
            inputStrides = {groups * channelsPerGroup * height * width, channelsPerGroup * height * width, height * width, 1};
        }

        TensorDesc inputTensorDesc = TensorDesc(m_inputTensorDescs[0].GetDmlDataType(), inputShape, inputStrides);
        const DML_TENSOR_DESC inputDmlTensorDesc = inputTensorDesc.GetDmlDesc();

        TensorDesc transposedInputTensorDesc = TensorDesc(m_inputTensorDescs[0].GetDmlDataType(), inputShape);
        const DML_TENSOR_DESC transposedInputDmlTensorDesc = transposedInputTensorDesc.GetDmlDesc();

        // 1. Reshape the gamma and beta from [channels] to [groups, channelsPerGroup]
        // 2. Broadcast the gamma and beta from [groups, channelsPerGroup] to [batch, height * width, groups, channelsPerGroup]
        // 3. Stride the brodcasted gamma and beta from [batch, height * width, groups, channelsPerGroup] to [batch, groups, channelsPerGroup, height * width]
        const std::array<uint32_t, 4> gammaBetaStrides = {0, channelsPerGroup, 1, 0};
        TensorDesc gammaBetaTensorDesc = TensorDesc(m_inputTensorDescs[1].GetDmlDataType(), inputShape, gammaBetaStrides);
        const DML_TENSOR_DESC gammaBetaDmlTensorDesc = gammaBetaTensorDesc.GetDmlDesc();

        // Cast Gamma and Beta if their datatype differ from the input's datatype
        const bool gammaBetaCastNeeded = m_inputTensorDescs[0].GetDmlDataType() != m_inputTensorDescs[1].GetDmlDataType();
        DML_CAST_OPERATOR_DESC castGammaBetaDesc{};
        if (gammaBetaCastNeeded)
        {
            castGammaBetaDesc.InputTensor = &gammaBetaDmlTensorDesc;
            castGammaBetaDesc.OutputTensor = &inputDmlTensorDesc;
        }
        DML_OPERATOR_DESC dmlCastGammaBetaDesc = { DML_OPERATOR_CAST, &castGammaBetaDesc };

        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC transposeInputDesc{};
        transposeInputDesc.InputTensor = &inputDmlTensorDesc;
        transposeInputDesc.OutputTensor = &transposedInputDmlTensorDesc;
        DML_OPERATOR_DESC dmlTransposeInputDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &transposeInputDesc };

        // Then, perform MVN
        DML_MEAN_VARIANCE_NORMALIZATION_OPERATOR_DESC mvnDesc{};
        mvnDesc.InputTensor = &transposedInputDmlTensorDesc;
        mvnDesc.ScaleTensor = gammaBetaCastNeeded ? &inputDmlTensorDesc : &gammaBetaDmlTensorDesc;
        mvnDesc.BiasTensor = gammaBetaCastNeeded ? &inputDmlTensorDesc : &gammaBetaDmlTensorDesc;
        mvnDesc.OutputTensor = &transposedInputDmlTensorDesc;
        mvnDesc.NormalizeVariance = true;
        mvnDesc.CrossChannel = false;
        mvnDesc.Epsilon = epsilon;
        DML_OPERATOR_DESC dmlMvnDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION, &mvnDesc };

        // Finally, execute the Swish activation function (x * sigmoid(x)) if provided
        DML_ACTIVATION_SIGMOID_OPERATOR_DESC swishSigmoidDesc{};
        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC swishMulDesc{};
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC transposeOutputDesc{};
        if (activation)
        {
            swishSigmoidDesc.InputTensor = &transposedInputDmlTensorDesc;
            swishSigmoidDesc.OutputTensor = &inputDmlTensorDesc;

            swishMulDesc.ATensor = &transposedInputDmlTensorDesc;
            swishMulDesc.BTensor = &inputDmlTensorDesc;
            swishMulDesc.OutputTensor = &inputDmlTensorDesc;
        }
        else
        {
            // Transpose the output to the layout that ORT expects
            transposeOutputDesc.InputTensor = &transposedInputDmlTensorDesc;
            transposeOutputDesc.OutputTensor = &inputDmlTensorDesc;
        }
        DML_OPERATOR_DESC dmlSwishSigmoidDesc = { DML_OPERATOR_ACTIVATION_SIGMOID, &swishSigmoidDesc };
        DML_OPERATOR_DESC dmlSwishMulDesc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &swishMulDesc };
        DML_OPERATOR_DESC dmlTransposeOutputDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &transposeOutputDesc };

        // Construct the graph
        std::vector<const DML_OPERATOR_DESC*> opDescs;
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

        uint32_t currentNodeIndex = 0;

        const uint32_t mvnNodeIndex = currentNodeIndex++;
        opDescs.push_back(&dmlMvnDesc);

        // We only need a transpose the input when the layout is NHWC
        if (channelsLast)
        {
            const uint32_t transposeInputIndex = currentNodeIndex++;
            opDescs.push_back(&dmlTransposeInputDesc);

            DML_INPUT_GRAPH_EDGE_DESC inputEdge{};
            inputEdge.GraphInputIndex = 0;
            inputEdge.ToNodeIndex = transposeInputIndex;
            inputEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(inputEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC transposeInputToMvnEdge = {};
            transposeInputToMvnEdge.FromNodeIndex = transposeInputIndex;
            transposeInputToMvnEdge.FromNodeOutputIndex = 0;
            transposeInputToMvnEdge.ToNodeIndex = mvnNodeIndex;
            transposeInputToMvnEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(transposeInputToMvnEdge);
        }
        else
        {
            DML_INPUT_GRAPH_EDGE_DESC inputEdge{};
            inputEdge.GraphInputIndex = 0;
            inputEdge.ToNodeIndex = mvnNodeIndex;
            inputEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(inputEdge);
        }

        if (gammaBetaCastNeeded)
        {
            const uint32_t gammaCastNodeIndex = currentNodeIndex++;
            opDescs.push_back(&dmlCastGammaBetaDesc);

            const uint32_t betaCastNodeIndex = currentNodeIndex++;
            opDescs.push_back(&dmlCastGammaBetaDesc);

            DML_INPUT_GRAPH_EDGE_DESC gammaEdge{};
            gammaEdge.GraphInputIndex = 1;
            gammaEdge.ToNodeIndex = gammaCastNodeIndex;
            gammaEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(gammaEdge);

            DML_INPUT_GRAPH_EDGE_DESC betaEdge{};
            betaEdge.GraphInputIndex = 2;
            betaEdge.ToNodeIndex = betaCastNodeIndex;
            betaEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(betaEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC gammaCastToMvnEdge = {};
            gammaCastToMvnEdge.FromNodeIndex = gammaCastNodeIndex;
            gammaCastToMvnEdge.FromNodeOutputIndex = 0;
            gammaCastToMvnEdge.ToNodeIndex = mvnNodeIndex;
            gammaCastToMvnEdge.ToNodeInputIndex = 1;
            intermediateEdges.push_back(gammaCastToMvnEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC betaCastToMvnEdge = {};
            betaCastToMvnEdge.FromNodeIndex = betaCastNodeIndex;
            betaCastToMvnEdge.FromNodeOutputIndex = 0;
            betaCastToMvnEdge.ToNodeIndex = mvnNodeIndex;
            betaCastToMvnEdge.ToNodeInputIndex = 2;
            intermediateEdges.push_back(betaCastToMvnEdge);
        }
        else
        {
            DML_INPUT_GRAPH_EDGE_DESC gammaEdge{};
            gammaEdge.GraphInputIndex = 1;
            gammaEdge.ToNodeIndex = mvnNodeIndex;
            gammaEdge.ToNodeInputIndex = 1;
            inputEdges.push_back(gammaEdge);

            DML_INPUT_GRAPH_EDGE_DESC betaEdge{};
            betaEdge.GraphInputIndex = 2;
            betaEdge.ToNodeIndex = mvnNodeIndex;
            betaEdge.ToNodeInputIndex = 2;
            inputEdges.push_back(betaEdge);
        }

        if (activation)
        {
            const uint32_t swishSigmoidNodeIndex = currentNodeIndex++;
            opDescs.push_back(&dmlSwishSigmoidDesc);

            const uint32_t swishMulNodeIndex = currentNodeIndex++;
            opDescs.push_back(&dmlSwishMulDesc);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC mvnToSwishSigmoidEdge = {};
            mvnToSwishSigmoidEdge.FromNodeIndex = mvnNodeIndex;
            mvnToSwishSigmoidEdge.FromNodeOutputIndex = 0;
            mvnToSwishSigmoidEdge.ToNodeIndex = swishSigmoidNodeIndex;
            mvnToSwishSigmoidEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(mvnToSwishSigmoidEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC mvnToSwishMulEdge = {};
            mvnToSwishMulEdge.FromNodeIndex = mvnNodeIndex;
            mvnToSwishMulEdge.FromNodeOutputIndex = 0;
            mvnToSwishMulEdge.ToNodeIndex = swishMulNodeIndex;
            mvnToSwishMulEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(mvnToSwishMulEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC swishSigmoidToSwishMulEdge = {};
            swishSigmoidToSwishMulEdge.FromNodeIndex = swishSigmoidNodeIndex;
            swishSigmoidToSwishMulEdge.FromNodeOutputIndex = 0;
            swishSigmoidToSwishMulEdge.ToNodeIndex = swishMulNodeIndex;
            swishSigmoidToSwishMulEdge.ToNodeInputIndex = 1;
            intermediateEdges.push_back(swishSigmoidToSwishMulEdge);

            DML_OUTPUT_GRAPH_EDGE_DESC swishMulToOutputEdge{};
            swishMulToOutputEdge.FromNodeIndex = swishMulNodeIndex;
            swishMulToOutputEdge.FromNodeOutputIndex = 0;
            swishMulToOutputEdge.GraphOutputIndex = 0;
            outputEdges.push_back(swishMulToOutputEdge);
        }
        else
        {
            if (channelsLast)
            {
                // We only need a transpose the output when the layout is NHWC
                const uint32_t transposeOutputNodeIndex = currentNodeIndex++;
                opDescs.push_back(&dmlTransposeOutputDesc);

                DML_INTERMEDIATE_GRAPH_EDGE_DESC mvnToTransposeOutputEdge = {};
                mvnToTransposeOutputEdge.FromNodeIndex = mvnNodeIndex;
                mvnToTransposeOutputEdge.FromNodeOutputIndex = 0;
                mvnToTransposeOutputEdge.ToNodeIndex = transposeOutputNodeIndex;
                mvnToTransposeOutputEdge.ToNodeInputIndex = 0;
                intermediateEdges.push_back(mvnToTransposeOutputEdge);

                DML_OUTPUT_GRAPH_EDGE_DESC transposeOutputToOutputEdge{};
                transposeOutputToOutputEdge.FromNodeIndex = transposeOutputNodeIndex;
                transposeOutputToOutputEdge.FromNodeOutputIndex = 0;
                transposeOutputToOutputEdge.GraphOutputIndex = 0;
                outputEdges.push_back(transposeOutputToOutputEdge);
            }
            else
            {
                DML_OUTPUT_GRAPH_EDGE_DESC mvnToOutputEdge{};
                mvnToOutputEdge.FromNodeIndex = mvnNodeIndex;
                mvnToOutputEdge.FromNodeOutputIndex = 0;
                mvnToOutputEdge.GraphOutputIndex = 0;
                outputEdges.push_back(mvnToOutputEdge);
            }
        }

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

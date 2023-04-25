// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorSkipLayerNormalization : public DmlOperator
{
public:
    DmlOperatorSkipLayerNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1, 2, 3, 4};
        std::vector<std::optional<uint32_t>> kernelOutputIndices = {0, 1, 2, 3};

        DmlOperator::Initialize(
            kernelCreationContext,
            kernelInputIndices,
            kernelOutputIndices,
            kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0),
            std::nullopt,
            kernelCreationContext.GetTensorShapeDescription().GetInputTensorDimensionCount(0));

        constexpr static uint32_t minimumDimensionCount = 4;

        // Pad the input and the output with trailing 1's until they are at least 4D
        for (uint32_t i = 0; i < kernelCreationContext.GetInputCount(); ++i)
        {
            if (m_inputTensorDescs[i].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
            {
                auto sizes = m_inputTensorDescs[i].GetSizes();
                std::vector<uint32_t> tensorShape(sizes.begin(), sizes.end());
                tensorShape.resize(std::max<size_t>(tensorShape.size(), minimumDimensionCount), 1);

                std::optional<std::vector<uint32_t>> optionalStrides;
                if (!m_inputTensorDescs[i].GetStrides().empty())
                {
                    auto strides = m_inputTensorDescs[i].GetStrides();
                    std::vector<uint32_t> tensorStrides(strides.begin(), strides.end());
                    tensorStrides.resize(std::max<size_t>(tensorStrides.size(), minimumDimensionCount), 0);
                    optionalStrides = std::move(tensorStrides);
                }

                m_inputTensorDescs[i] = TensorDesc(
                    m_inputTensorDescs[i].GetDmlDataType(),
                    tensorShape,
                    std::move(optionalStrides));
            }
        }

        m_outputTensorDescs[0] = TensorDesc(
            m_outputTensorDescs[0].GetDmlDataType(),
            m_inputTensorDescs[0].GetSizes());

        if (m_outputTensorDescs[3].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
        {
            m_outputTensorDescs[3] = TensorDesc(
                m_outputTensorDescs[3].GetDmlDataType(),
                m_inputTensorDescs[0].GetSizes());
        }

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);
        int32_t onnxAxis = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::Axis, -1);
        uint32_t onnxDimCount = kernelCreationContext.GetTensorShapeDescription().GetInputTensorDimensionCount(0);
        uint32_t dmlDimCount = m_inputTensorDescs[0].GetDimensionCount();
        onnxAxis = OperatorHelper::HandleNegativeAxis(onnxAxis, onnxDimCount);
        std::vector<uint32_t> onnxAxes(static_cast<size_t>(dmlDimCount) - static_cast<size_t>(onnxAxis));
        std::iota(onnxAxes.begin(), onnxAxes.end(), onnxAxis);

        assert(m_inputTensorDescs.size() == 5);
        assert(m_outputTensorDescs.size() == 4);

        auto inputDesc = m_inputTensorDescs[0].GetDmlDesc();
        auto skipDesc = m_inputTensorDescs[1].GetDmlDesc();
        auto gammaDesc = m_inputTensorDescs[2].GetDmlDesc();
        auto betaDesc = m_inputTensorDescs[3].GetDmlDesc();
        auto biasDesc = m_inputTensorDescs[4].GetDmlDesc();
        auto outputDesc = m_outputTensorDescs[0].GetDmlDesc();
        auto inputSkipBiasSum = m_outputTensorDescs[3].GetDmlDesc();

        TensorDesc inputSkipBiasTensorDesc(m_inputTensorDescs[0].GetDmlDataType(), m_inputTensorDescs[0].GetSizes());
        DML_TENSOR_DESC inputSkipBiasDmlTensorDesc = inputSkipBiasTensorDesc.GetDmlDesc();

        DML_ELEMENT_WISE_ADD_OPERATOR_DESC inputSkipAddDesc = {};
        inputSkipAddDesc.ATensor = &inputDesc;
        inputSkipAddDesc.BTensor = &skipDesc;
        inputSkipAddDesc.OutputTensor = &inputSkipBiasDmlTensorDesc;
        DML_OPERATOR_DESC inputSkipAddOpDesc = { DML_OPERATOR_ELEMENT_WISE_ADD, &inputSkipAddDesc };

        DML_ELEMENT_WISE_ADD_OPERATOR_DESC inputSkipBiasAddDesc = {};
        inputSkipBiasAddDesc.ATensor = &inputSkipBiasDmlTensorDesc;
        inputSkipBiasAddDesc.BTensor = &biasDesc;
        inputSkipBiasAddDesc.OutputTensor = &inputSkipBiasDmlTensorDesc;
        DML_OPERATOR_DESC inputSkipBiasAddOpDesc = { DML_OPERATOR_ELEMENT_WISE_ADD, &inputSkipBiasAddDesc };

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC mvnDesc = {};
        mvnDesc.InputTensor = &inputSkipBiasDmlTensorDesc;
        mvnDesc.ScaleTensor = &gammaDesc;
        mvnDesc.BiasTensor = betaDesc.Desc ? &betaDesc : nullptr;
        mvnDesc.OutputTensor = &outputDesc;
        mvnDesc.Axes = onnxAxes.data();
        mvnDesc.AxisCount = gsl::narrow_cast<uint32_t>(onnxAxes.size());
        mvnDesc.NormalizeVariance = true;
        mvnDesc.Epsilon = epsilon;
        mvnDesc.FusedActivation = nullptr;
        DML_OPERATOR_DESC mvnOpDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &mvnDesc };

        // Construct the graph
        std::vector<const DML_OPERATOR_DESC*> opDescs;
        opDescs.reserve(3);

        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        inputEdges.reserve(5);

        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        intermediateEdges.reserve(2);

        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        outputEdges.reserve(1);

        // Insert the Input + Skip operation into the graph
        opDescs.push_back(&inputSkipAddOpDesc);

        DML_INPUT_GRAPH_EDGE_DESC dataInputEdge = {};
        dataInputEdge.GraphInputIndex = 0;
        dataInputEdge.ToNodeIndex = 0;
        dataInputEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(std::move(dataInputEdge));

        DML_INPUT_GRAPH_EDGE_DESC skipInputEdge = {};
        skipInputEdge.GraphInputIndex = 1;
        skipInputEdge.ToNodeIndex = 0;
        skipInputEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(std::move(skipInputEdge));

        // Insert the InputSkip + Bias operation into the graph
        if (biasDesc.Desc)
        {
            opDescs.push_back(&inputSkipBiasAddOpDesc);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdge = {};
            intermediateEdge.FromNodeIndex = 0;
            intermediateEdge.FromNodeOutputIndex = 0;
            intermediateEdge.ToNodeIndex = 1;
            intermediateEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(std::move(intermediateEdge));

            DML_INPUT_GRAPH_EDGE_DESC biasInputEdge = {};
            biasInputEdge.GraphInputIndex = 4;
            biasInputEdge.ToNodeIndex = 1;
            biasInputEdge.ToNodeInputIndex = 1;
            inputEdges.push_back(std::move(biasInputEdge));

            if (inputSkipBiasSum.Desc)
            {
                DML_OUTPUT_GRAPH_EDGE_DESC inputSkipBiasSumEdge = {};
                inputSkipBiasSumEdge.FromNodeIndex = 1;
                inputSkipBiasSumEdge.FromNodeOutputIndex = 0;
                inputSkipBiasSumEdge.GraphOutputIndex = 3;
                outputEdges.push_back(std::move(inputSkipBiasSumEdge));
            }
        }
        else if (inputSkipBiasSum.Desc)
        {
            DML_OUTPUT_GRAPH_EDGE_DESC inputSkipBiasSumEdge = {};
            inputSkipBiasSumEdge.FromNodeIndex = 0;
            inputSkipBiasSumEdge.FromNodeOutputIndex = 0;
            inputSkipBiasSumEdge.GraphOutputIndex = 3;
            outputEdges.push_back(std::move(inputSkipBiasSumEdge));
        }

        // Insert the MVN operation into the graph
        opDescs.push_back(&mvnOpDesc);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdge = {};
        intermediateEdge.FromNodeIndex = biasDesc.Desc ? 1 : 0;
        intermediateEdge.FromNodeOutputIndex = 0;
        intermediateEdge.ToNodeIndex = biasDesc.Desc ? 2 : 1;
        intermediateEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(std::move(intermediateEdge));

        DML_INPUT_GRAPH_EDGE_DESC gammaInputEdge = {};
        gammaInputEdge.GraphInputIndex = 2;
        gammaInputEdge.ToNodeIndex = biasDesc.Desc ? 2 : 1;
        gammaInputEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(std::move(gammaInputEdge));

        if (betaDesc.Desc)
        {
            DML_INPUT_GRAPH_EDGE_DESC betaInputEdge = {};
            betaInputEdge.GraphInputIndex = 3;
            betaInputEdge.ToNodeIndex = biasDesc.Desc ? 2 : 1;
            betaInputEdge.ToNodeInputIndex = 2;
            inputEdges.push_back(std::move(betaInputEdge));
        }

        DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
        outputEdge.GraphOutputIndex = 0;
        outputEdge.FromNodeIndex = biasDesc.Desc ? 2 : 1;
        outputEdge.FromNodeOutputIndex = 0;
        outputEdges.push_back(std::move(outputEdge));

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

void CALLBACK QuerySkipLayerNormalization(IMLOperatorSupportQueryContextPrivate* context, /*out*/ bool* isSupported)
{
    *isSupported = false;

    // `mean` output tensor is not supported yet
    if (context->IsOutputValid(1))
    {
        return;
    }

    // `inv_std_var` output tensor is not supported yet
    if (context->IsOutputValid(2))
    {
        return;
    }

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(SkipLayerNormalization, DmlOperatorSkipLayerNormalization);

} // namespace Dml

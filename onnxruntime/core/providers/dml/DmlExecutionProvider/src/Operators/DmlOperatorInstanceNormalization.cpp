// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorInstanceNormalization : public DmlOperator
{
    enum InputTensors
    {
        IN_X,
        IN_SCALE,
        IN_BIAS
    };

    void Shift1DInputsTensorDesc(
        const MLOperatorKernelCreationContext& kernelCreationContext,
        int index,
        int count,
        uint32_t minimumDimensionCount
        )
    {
        for (int i = index; i != index + count; ++i)
        {
            // Shift a single dimension size to the C channel.
            // e.g. Given a 4D input (X), then a 1D scale of [7] becomes [1,7,1,1].
            TensorDesc& tensorDesc = m_inputTensorDescs[i];
            gsl::span<const uint32_t> sizes = tensorDesc.GetSizes();
            gsl::span<const uint32_t> lastDimension = sizes.last(1);
            m_inputTensorDescs[i] = CreateTensorDescFromInput(kernelCreationContext, i, TensorAxis::DoNotCoerce, TensorAxis::C, TensorAxis::LeftAligned, lastDimension, minimumDimensionCount);
        }
    }

public:
    DmlOperatorInstanceNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1, 2};
        DmlOperator::Initialize(
            kernelCreationContext,
            kernelInputIndices,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            /*minimumDimensionCount*/ 1);

        constexpr static uint32_t minimumDimensionCount = 4;

        // Shift IN_SCALE and IN_BIAS input tensor descs {1, C, 1, 1} out of 1D tensors.
        Shift1DInputsTensorDesc(kernelCreationContext, IN_SCALE, 2, minimumDimensionCount);

        // Pad the input and the output with trailing 1's until they are at least 4D
        auto sizes = m_inputTensorDescs[0].GetSizes();
        std::vector<uint32_t> tensorShape(sizes.begin(), sizes.end());
        tensorShape.resize(std::max<size_t>(tensorShape.size(), minimumDimensionCount), 1);

        m_inputTensorDescs[0] = TensorDesc(
            m_inputTensorDescs[0].GetDmlDataType(),
            tensorShape);

        m_inputTensorDescs[1] = TensorDesc(
            m_inputTensorDescs[1].GetDmlDataType(),
            m_inputTensorDescs[1].GetSizes(),
            std::vector<uint32_t>({0, 1, 0, 0}));

        m_inputTensorDescs[2] = TensorDesc(
            m_inputTensorDescs[2].GetDmlDataType(),
            m_inputTensorDescs[2].GetSizes(),
            std::vector<uint32_t>({0, 1, 0, 0}));

        m_outputTensorDescs[0] = m_inputTensorDescs[0];

        // "Instance" normalization is really spatial normalization,
        // where the spatial channels are reduced and normalized, while
        // batch and channel remain independent. So pass a list of axes
        // just beyond the leading batch and channel dimensions (starting
        // at axis 2 up to the last spatial dimension).
        const uint32_t inputDimensionCount = m_inputTensorDescs.front().GetDimensionCount();
        std::vector<uint32_t> axes(inputDimensionCount - 2);
        std::iota(axes.begin(), axes.end(), 2u);

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);
        const std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelCreationContext);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        std::vector<uint32_t> strides(4);
        GetDescendingPackedStrides(m_inputTensorDescs[0].GetSizes(), strides);

        std::vector<uint32_t> transposedStrides(4);
        transposedStrides[1] = 1;
        transposedStrides[3] = m_inputTensorDescs[0].GetSizes()[1];
        transposedStrides[2] = m_inputTensorDescs[0].GetSizes()[1] * m_inputTensorDescs[0].GetSizes()[3];
        transposedStrides[0] = m_inputTensorDescs[0].GetSizes()[1] * m_inputTensorDescs[0].GetSizes()[3] * m_inputTensorDescs[0].GetSizes()[2];

        auto transposedInputDesc = TensorDesc(m_inputTensorDescs[0].GetDmlDataType(), m_inputTensorDescs[0].GetSizes(), transposedStrides);
        auto transposedInputDmlDesc = transposedInputDesc.GetDmlDesc();

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &transposedInputDmlDesc;
        operatorDesc.ScaleTensor = &inputDescs[1];
        operatorDesc.BiasTensor = &inputDescs[2];
        operatorDesc.OutputTensor = &transposedInputDmlDesc;
        operatorDesc.Axes = axes.data();
        operatorDesc.AxisCount = static_cast<uint32_t>(axes.size());
        operatorDesc.NormalizeVariance = true;
        operatorDesc.Epsilon = epsilon;
        operatorDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;
        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &operatorDesc };

        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC transpose1Desc = {};
        transpose1Desc.InputTensor = &inputDescs[0];
        transpose1Desc.OutputTensor = &transposedInputDmlDesc;
        DML_OPERATOR_DESC transpose1OpDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &transpose1Desc };

        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC transpose2Desc = {};
        transpose2Desc.InputTensor = &transposedInputDmlDesc;
        transpose2Desc.OutputTensor = &outputDescs[0];
        DML_OPERATOR_DESC transpose2OpDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &transpose2Desc };

        // Construct the graph
        std::vector<const DML_OPERATOR_DESC*> opDescs({&transpose1OpDesc, &opDesc, &transpose2OpDesc});

        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        inputEdges.reserve(3);

        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        intermediateEdges.reserve(4);

        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        outputEdges.reserve(1);

        DML_INPUT_GRAPH_EDGE_DESC dataInputEdge1 = {};
        dataInputEdge1.GraphInputIndex = 0;
        dataInputEdge1.ToNodeIndex = 0;
        dataInputEdge1.ToNodeInputIndex = 0;
        inputEdges.push_back(std::move(dataInputEdge1));

        DML_INTERMEDIATE_GRAPH_EDGE_DESC dataIntermediateEdge1 = {};
        dataIntermediateEdge1.FromNodeIndex = 0;
        dataIntermediateEdge1.FromNodeOutputIndex = 0;
        dataIntermediateEdge1.ToNodeIndex = 1;
        dataIntermediateEdge1.ToNodeInputIndex = 0;
        intermediateEdges.push_back(std::move(dataIntermediateEdge1));

        DML_INPUT_GRAPH_EDGE_DESC dataInputEdge2 = {};
        dataInputEdge2.GraphInputIndex = 1;
        dataInputEdge2.ToNodeIndex = 1;
        dataInputEdge2.ToNodeInputIndex = 1;
        inputEdges.push_back(std::move(dataInputEdge2));

        DML_INPUT_GRAPH_EDGE_DESC dataInputEdge3 = {};
        dataInputEdge3.GraphInputIndex = 2;
        dataInputEdge3.ToNodeIndex = 1;
        dataInputEdge3.ToNodeInputIndex = 2;
        inputEdges.push_back(std::move(dataInputEdge3));

        DML_INTERMEDIATE_GRAPH_EDGE_DESC dataIntermediateEdge2 = {};
        dataIntermediateEdge2.FromNodeIndex = 1;
        dataIntermediateEdge2.FromNodeOutputIndex = 0;
        dataIntermediateEdge2.ToNodeIndex = 2;
        dataIntermediateEdge2.ToNodeInputIndex = 0;
        intermediateEdges.push_back(std::move(dataIntermediateEdge2));

        DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
        outputEdge.FromNodeIndex = 2;
        outputEdge.FromNodeOutputIndex = 0;
        outputEdge.GraphOutputIndex = 0;
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

DML_OP_DEFINE_CREATION_FUNCTION(InstanceNormalization, DmlOperatorInstanceNormalization);
DML_OP_DEFINE_CREATION_FUNCTION(DmlFusedInstanceNormalization, DmlOperatorInstanceNormalization);

} // namespace Dml

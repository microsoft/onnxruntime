// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorSimplifiedLayerNormalization : public DmlOperator
{
public:
    DmlOperatorSimplifiedLayerNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1};

        DmlOperator::Initialize(
            kernelCreationContext,
            kernelInputIndices,
            std::nullopt,
            std::nullopt,
            std::nullopt,
            kernelCreationContext.GetTensorShapeDescription().GetInputTensorDimensionCount(0));

        constexpr static uint32_t minimumDimensionCount = 4;

        // Pad the input and the output with trailing 1's until they are at least 4D
        for (uint32_t i = 0; i < kernelCreationContext.GetInputCount(); ++i)
        {
            auto sizes = m_inputTensorDescs[i].GetSizes();
            std::vector<uint32_t> tensorShape(sizes.begin(), sizes.end());
            tensorShape.resize(std::max<size_t>(tensorShape.size(), minimumDimensionCount), 1);

            if (m_inputTensorDescs[i].GetDmlDataType() != DML_TENSOR_TYPE_INVALID)
            {
                m_inputTensorDescs[i] = TensorDesc(
                    m_inputTensorDescs[i].GetDmlDataType(),
                    tensorShape);
            }
        }

        m_outputTensorDescs[0] = TensorDesc(
            m_outputTensorDescs[0].GetDmlDataType(),
            m_inputTensorDescs[0].GetSizes());

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, DefaultEpsilon);
        int32_t onnxAxis = kernelCreationContext.GetOptionalAttribute<int32_t>(AttrName::Axis, -1);
        uint32_t onnxDimCount = kernelCreationContext.GetTensorShapeDescription().GetInputTensorDimensionCount(0);
        uint32_t dmlDimCount = m_inputTensorDescs[0].GetDimensionCount();
        onnxAxis = OperatorHelper::HandleNegativeAxis(onnxAxis, onnxDimCount);
        std::vector<uint32_t> onnxAxes(static_cast<size_t>(dmlDimCount) - static_cast<size_t>(onnxAxis));
        std::iota(onnxAxes.begin(), onnxAxes.end(), onnxAxis);

        assert(m_inputTensorDescs.size() == 2);
        assert(m_outputTensorDescs.size() == 1);

        auto inputDataType = m_inputTensorDescs[0].GetDmlDataType();
        ORT_THROW_HR_IF(E_INVALIDARG, inputDataType != DML_TENSOR_DATA_TYPE_FLOAT16 && inputDataType != DML_TENSOR_DATA_TYPE_FLOAT32);

        auto scaleDataType = m_inputTensorDescs[1].GetDmlDataType();
        ORT_THROW_HR_IF(E_INVALIDARG, scaleDataType != DML_TENSOR_DATA_TYPE_FLOAT16 && scaleDataType != DML_TENSOR_DATA_TYPE_FLOAT32);

        // Scale and Output always have the same data type
        ORT_THROW_HR_IF(E_INVALIDARG, m_outputTensorDescs[0].GetDmlDataType() != scaleDataType);

        auto inputDesc = m_inputTensorDescs[0].GetDmlDesc();
        auto scaleDesc = m_inputTensorDescs[1].GetDmlDesc();
        auto outputDesc = m_outputTensorDescs[0].GetDmlDesc();

        DML_CAST_OPERATOR_DESC inputCastDesc = {};
        DML_OPERATOR_DESC inputCastOpDesc = { DML_OPERATOR_CAST, nullptr };

        DML_CAST_OPERATOR_DESC scaleCastDesc = {};
        DML_OPERATOR_DESC scaleCastOpDesc = { DML_OPERATOR_CAST, nullptr };

        // When data types mismatch, we cast to the highest precision to respect DML's requirement that all datatypes must match
        TensorDesc inputCastOutputTensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_inputTensorDescs[0].GetSizes());
        DML_TENSOR_DESC inputCastOutputDmlTensorDesc = inputCastOutputTensorDesc.GetDmlDesc();

        TensorDesc scaleCastOutputTensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_inputTensorDescs[1].GetSizes());
        DML_TENSOR_DESC scaleCastOutputDmlTensorDesc = scaleCastOutputTensorDesc.GetDmlDesc();

        // Cast all tensors to the highest common precision
        if (inputDataType == DML_TENSOR_DATA_TYPE_FLOAT16 && scaleDataType == DML_TENSOR_DATA_TYPE_FLOAT32)
        {
            inputCastDesc.InputTensor = &inputDesc;
            inputCastDesc.OutputTensor = &inputCastOutputDmlTensorDesc;
            inputCastOpDesc.Desc = &inputCastDesc;
        }
        else if (inputDataType == DML_TENSOR_DATA_TYPE_FLOAT32 && scaleDataType == DML_TENSOR_DATA_TYPE_FLOAT16)
        {
            scaleCastDesc.InputTensor = &scaleDesc;
            scaleCastDesc.OutputTensor = &scaleCastOutputDmlTensorDesc;
            scaleCastOpDesc.Desc = &scaleCastDesc;
        }

        // Make sure that the output is the same type as the input
        DML_CAST_OPERATOR_DESC outputCastDesc = {};
        DML_OPERATOR_DESC outputCastOpDesc = { DML_OPERATOR_CAST, nullptr };

        auto realInputDataType = inputCastOpDesc.Desc ? inputCastOutputTensorDesc.GetDmlDataType() : m_inputTensorDescs[0].GetDmlDataType();
        TensorDesc outputCastOutputTensorDesc(realInputDataType, m_outputTensorDescs[0].GetSizes());
        DML_TENSOR_DESC outputCastOutputDmlTensorDesc = outputCastOutputTensorDesc.GetDmlDesc();

        if (realInputDataType != m_outputTensorDescs[0].GetDmlDataType())
        {
            // After the operator has been executed, we need to cast the "casted" output tensor to the original output tensor that TF expects
            outputCastDesc.InputTensor = &outputCastOutputDmlTensorDesc;
            outputCastDesc.OutputTensor = &outputDesc;
            outputCastOpDesc.Desc = &outputCastDesc;
        }

        // Simplified Layer Norm doesn't use Mean or Scale, so broadcast an empty tensor
        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC zeroDesc = {};
        zeroDesc.ValueDataType = inputCastOpDesc.Desc ? inputCastOutputTensorDesc.GetDmlDataType() : m_inputTensorDescs[0].GetDmlDataType();

        std::vector<uint32_t> scalarTensorSizes(m_inputTensorDescs[0].GetDimensionCount(), 1);
        TensorDesc scalarTensorDesc(m_outputTensorDescs[0].GetDmlDataType(), scalarTensorSizes);
        auto scalarTensorDmlDesc = scalarTensorDesc.GetDmlDesc();
        zeroDesc.OutputTensor = &scalarTensorDmlDesc;

        std::vector<uint32_t> broadcastedScalarTensorStrides(m_inputTensorDescs[0].GetDimensionCount(), 0);
        TensorDesc broadcastedScalarTensorDesc(m_outputTensorDescs[0].GetDmlDataType(), m_inputTensorDescs[0].GetSizes(), broadcastedScalarTensorStrides);
        auto broadcastedScalarTensorDmlDesc = broadcastedScalarTensorDesc.GetDmlDesc();

        DML_BATCH_NORMALIZATION_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = inputCastOpDesc.Desc ? &inputCastOutputDmlTensorDesc : &inputDesc;
        operatorDesc.ScaleTensor = scaleCastOpDesc.Desc ? &scaleCastOutputDmlTensorDesc : &scaleDesc;
        operatorDesc.MeanTensor = &broadcastedScalarTensorDmlDesc;
        operatorDesc.BiasTensor = &broadcastedScalarTensorDmlDesc;

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.OutputTensor = outputCastOpDesc.Desc ? &outputCastOutputDmlTensorDesc : &outputDesc;
        operatorDesc.Axes = onnxAxes.data();
        operatorDesc.AxisCount = gsl::narrow_cast<uint32_t>(onnxAxes.size());
        operatorDesc.NormalizeVariance = true;
        operatorDesc.Epsilon = epsilon;
        operatorDesc.FusedActivation = nullptr;
        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &operatorDesc };

        // Construct the graph
        std::vector<const DML_OPERATOR_DESC*> opDescs;
        opDescs.reserve(5);

        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        inputEdges.reserve(3);

        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        intermediateEdges.reserve(4);

        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        outputEdges.reserve(1);

        opDescs.push_back(&opDesc);
        uint32_t currentNodeIndex = 1;

        DML_INPUT_GRAPH_EDGE_DESC dataInputEdge = {};
        dataInputEdge.GraphInputIndex = 0;
        dataInputEdge.ToNodeIndex = inputCastOpDesc.Desc ? currentNodeIndex : 0;
        dataInputEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(std::move(dataInputEdge));

        if (inputCastOpDesc.Desc)
        {
            opDescs.push_back(&inputCastOpDesc);

            // Link the cast op to the MVN op
            DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdge = {};
            intermediateEdge.FromNodeIndex = currentNodeIndex;
            intermediateEdge.FromNodeOutputIndex = 0;
            intermediateEdge.ToNodeIndex = 0;
            intermediateEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(std::move(intermediateEdge));
            ++currentNodeIndex;
        }

        DML_INPUT_GRAPH_EDGE_DESC scaleInputEdge = {};
        scaleInputEdge.GraphInputIndex = 1;
        scaleInputEdge.ToNodeIndex = scaleCastOpDesc.Desc ? currentNodeIndex : 0;
        scaleInputEdge.ToNodeInputIndex = scaleCastOpDesc.Desc ? 0 : 1;
        inputEdges.push_back(std::move(scaleInputEdge));

        if (scaleCastOpDesc.Desc)
        {
            opDescs.push_back(&scaleCastOpDesc);

            // Link the cast op to the MVN op
            DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdge = {};
            intermediateEdge.FromNodeIndex = currentNodeIndex;
            intermediateEdge.FromNodeOutputIndex = 0;
            intermediateEdge.ToNodeIndex = 0;
            intermediateEdge.ToNodeInputIndex = 1;
            intermediateEdges.push_back(std::move(intermediateEdge));
            ++currentNodeIndex;
        }

        DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
        outputEdge.GraphOutputIndex = 0;
        outputEdge.FromNodeIndex = outputCastOpDesc.Desc ? currentNodeIndex : 0;
        outputEdge.FromNodeOutputIndex = 0;
        outputEdges.push_back(std::move(outputEdge));

        if (outputCastOpDesc.Desc)
        {
            opDescs.push_back(&outputCastOpDesc);

            // Link the MVN op to the cast op
            DML_INTERMEDIATE_GRAPH_EDGE_DESC intermediateEdge = {};
            intermediateEdge.FromNodeIndex = 0;
            intermediateEdge.FromNodeOutputIndex = 0;
            intermediateEdge.ToNodeIndex = currentNodeIndex;
            intermediateEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(std::move(intermediateEdge));
            ++currentNodeIndex;
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

DML_OP_DEFINE_CREATION_FUNCTION(SimplifiedLayerNormalization, DmlOperatorSimplifiedLayerNormalization);

} // namespace Dml

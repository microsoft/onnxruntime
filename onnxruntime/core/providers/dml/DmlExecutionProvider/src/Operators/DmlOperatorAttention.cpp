// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
class DmlOperatorAttention : public DmlOperator
{
    // This order matches the ONNX schema.
    //enum OnnxInputIndex
    //{
    //    X, // Input
    //    X_scale,
    //    X_zero_point,
    //    Y_scale,
    //    Y_zero_point,
    //    Count,
    //};

public:
    DmlOperatorAttention(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() >= 3);
        DmlOperator::Initialize(kernelCreationContext, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 1);

        std::vector<uint32_t> inputTensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        std::vector<uint32_t> weightTensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(1);
        std::vector<uint32_t> biasTensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(2);
        std::vector<uint32_t> maskIndexTensorShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(3);
        ML_CHECK_VALID_ARGUMENT(inputTensorShape.size() == 3);
        ML_CHECK_VALID_ARGUMENT(weightTensorShape.size() == 2);
        ML_CHECK_VALID_ARGUMENT(biasTensorShape.size() == 1);
        ML_CHECK_VALID_ARGUMENT(weightTensorShape[1] == biasTensorShape[0]);
        ML_CHECK_VALID_ARGUMENT(biasTensorShape[0] % 3 == 0);
        ML_CHECK_VALID_ARGUMENT(inputTensorShape[2] == weightTensorShape[0]);
        ML_CHECK_VALID_ARGUMENT(maskIndexTensorShape.size() > 1); // TODO fix Attention kernel when maskIndexTensorShape is 1
        const uint32_t batchSize = inputTensorShape[0];
        const uint32_t sequenceLength = inputTensorShape[1];
        const uint32_t hiddenSize = biasTensorShape[0] / 3;
        const uint32_t numHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::NumHeads));
        ML_CHECK_VALID_ARGUMENT(hiddenSize % numHeads == 0);
        const uint32_t headSize = hiddenSize / numHeads;

        std::vector<uint32_t> desiredWeightTensorShape{batchSize, weightTensorShape[0], 3 * hiddenSize};
        std::vector<uint32_t> desiredBiasTensorShape{batchSize, sequenceLength, 3 * hiddenSize};
        auto inputEdgeDesc = kernelCreationContext.GetInputEdgeDescription(0);
        MLOperatorTensorDataType dataType = inputEdgeDesc.tensorDataType;

        // overwrite weightTensorDesc
        m_inputTensorDescs[1] = TensorDesc(
                dataType,
                gsl::make_span(desiredWeightTensorShape),  // desired shape
                gsl::make_span(weightTensorShape),  // actual shape
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                1, // minDimensionCount
                0
            );

        // overwrite weightTensorDesc
        m_inputTensorDescs[2] = TensorDesc(
                dataType,
                gsl::make_span(desiredBiasTensorShape),  // desired shape
                gsl::make_span(biasTensorShape),  // actual shape
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                1, // minDimensionCount
                0
            );

        // overwrite maskIndexTensorDesc
        uint32_t maskIndexDimensionCount = gsl::narrow_cast<uint32_t>(maskIndexTensorShape.size());
        maskIndexTensorShape.insert(maskIndexTensorShape.begin() + 1, 4 - maskIndexDimensionCount, 1);
        std::vector<uint32_t> desiredMaskIndexShape {batchSize, numHeads, sequenceLength, sequenceLength};
        m_inputTensorDescs[3] = TensorDesc(
                kernelCreationContext.GetInputEdgeDescription(3).tensorDataType,
                gsl::make_span(desiredMaskIndexShape),
                gsl::make_span(maskIndexTensorShape),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                1,
                0
            );

        // overwrite output tensor desc
        std::vector<uint32_t> outputTensorShape {batchSize, sequenceLength, numHeads, headSize};
        std::vector<uint32_t> outputTensorStrides {sequenceLength * numHeads * headSize, headSize, headSize * sequenceLength, 1};
        m_outputTensorDescs[0] = TensorDesc(
                GetDmlDataTypeFromMlDataType(dataType),
                gsl::make_span(outputTensorShape),
                gsl::make_span(outputTensorStrides),
                0
            );

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        TensorDesc firstGemmOutputTensorDesc = TensorDesc(
                dataType,
                m_inputTensorDescs[2].GetSizes(),
                m_inputTensorDescs[2].GetSizes(),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                1,
                0
            );
        DML_GEMM_OPERATOR_DESC xWeightOperatorDesc = {};
        xWeightOperatorDesc.ATensor = &inputDescs[0];
        xWeightOperatorDesc.BTensor = &inputDescs[1];
        xWeightOperatorDesc.CTensor = &inputDescs[2];
        xWeightOperatorDesc.OutputTensor = &firstGemmOutputTensorDesc.GetDmlDesc();
        xWeightOperatorDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
        xWeightOperatorDesc.TransB = DML_MATRIX_TRANSFORM_NONE;
        xWeightOperatorDesc.Alpha = 1.0f;
        xWeightOperatorDesc.Beta = 1.0f;
        xWeightOperatorDesc.FusedActivation = nullptr;
        const DML_OPERATOR_DESC xWeightDesc {DML_OPERATOR_GEMM, &xWeightOperatorDesc};


        std::vector<uint32_t> querySlicedTensorShape {batchSize, sequenceLength, hiddenSize};
        TensorDesc querySlicedInputTensorDesc = TensorDesc(
                dataType,
                gsl::make_span(querySlicedTensorShape),
                gsl::make_span(querySlicedTensorShape),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                1, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );

        std::vector<uint32_t> querySliceOffset{0, 0, 0}, keySliceOffset{0, 0, hiddenSize}, valueSliceOffset{0, 0, 2 * hiddenSize};
        std::vector<uint32_t> sliceSize{batchSize, sequenceLength, hiddenSize};
        std::vector<int32_t> strides{1, 1, 1};
        DML_SLICE1_OPERATOR_DESC querySlicedOperatorDesc = {};
        querySlicedOperatorDesc.InputTensor = &firstGemmOutputTensorDesc.GetDmlDesc();
        querySlicedOperatorDesc.OutputTensor = &querySlicedInputTensorDesc.GetDmlDesc();
        querySlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(querySlicedTensorShape.size());
        querySlicedOperatorDesc.InputWindowOffsets = querySliceOffset.data();
        querySlicedOperatorDesc.InputWindowSizes = sliceSize.data();
        querySlicedOperatorDesc.InputWindowStrides = strides.data();
        const DML_OPERATOR_DESC querySlicedDesc = { DML_OPERATOR_SLICE1, &querySlicedOperatorDesc };

        DML_SLICE1_OPERATOR_DESC keySlicedOperatorDesc = {};
        keySlicedOperatorDesc.InputTensor = &firstGemmOutputTensorDesc.GetDmlDesc();
        keySlicedOperatorDesc.OutputTensor = &querySlicedInputTensorDesc.GetDmlDesc();
        keySlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(querySlicedTensorShape.size());
        keySlicedOperatorDesc.InputWindowOffsets = keySliceOffset.data();
        keySlicedOperatorDesc.InputWindowSizes = sliceSize.data();
        keySlicedOperatorDesc.InputWindowStrides = strides.data();
        const DML_OPERATOR_DESC keySlicedDesc = { DML_OPERATOR_SLICE1, &keySlicedOperatorDesc };

        DML_SLICE1_OPERATOR_DESC valueSlicedOperatorDesc = {};
        valueSlicedOperatorDesc.InputTensor = &firstGemmOutputTensorDesc.GetDmlDesc();
        valueSlicedOperatorDesc.OutputTensor = &querySlicedInputTensorDesc.GetDmlDesc();
        valueSlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(querySlicedTensorShape.size());
        valueSlicedOperatorDesc.InputWindowOffsets = valueSliceOffset.data();
        valueSlicedOperatorDesc.InputWindowSizes = sliceSize.data();
        valueSlicedOperatorDesc.InputWindowStrides = strides.data();
        const DML_OPERATOR_DESC valueSlicedDesc = { DML_OPERATOR_SLICE1, &valueSlicedOperatorDesc};

        TensorDesc castedMaskIndexTensorDesc = TensorDesc(
                MLOperatorTensorDataType::Float,
                gsl::make_span(desiredMaskIndexShape),
                gsl::make_span(desiredMaskIndexShape),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                1, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );
        DML_CAST_OPERATOR_DESC castMaskIndexOperatorDesc = {};
        castMaskIndexOperatorDesc.InputTensor = &inputDescs[3];
        castMaskIndexOperatorDesc.OutputTensor = &castedMaskIndexTensorDesc.GetDmlDesc();
        const DML_OPERATOR_DESC castMaskIndexDesc = {DML_OPERATOR_CAST, &castMaskIndexOperatorDesc};

        DML_SCALE_BIAS scaleBias = {};
        scaleBias.Scale = -10000.0f;
        scaleBias.Bias = 10000.0f;
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC maskOperatorDesc = {};
        maskOperatorDesc.InputTensor = &castedMaskIndexTensorDesc.GetDmlDesc();
        maskOperatorDesc.OutputTensor = &castedMaskIndexTensorDesc.GetDmlDesc();
        maskOperatorDesc.ScaleBias = &scaleBias;
        const DML_OPERATOR_DESC maskDesc = {DML_OPERATOR_ELEMENT_WISE_IDENTITY, &maskOperatorDesc};

        // original reshaped shape: [batchSize, seqenceLength, numHeads, headSize]
        // transposed shape to [0, 2, 1, 3] -> [batchSize, numHeads, sequenceLength, headSize]
        std::vector<uint32_t> reshapedTransposedQueryTensorShape {batchSize, numHeads, sequenceLength, headSize};
        std::vector<uint32_t> reshapedTransposedQueryTensorStride {sequenceLength * numHeads * headSize, headSize, numHeads * headSize, 1};
        TensorDesc reshapedTransposedQueryTensorDesc = TensorDesc(
                GetDmlDataTypeFromMlDataType(dataType),
                gsl::make_span(reshapedTransposedQueryTensorShape),
                gsl::make_span(reshapedTransposedQueryTensorStride),
                0 // guaranteedBaseOffsetAlignment
            );

        std::vector<uint32_t> reshapedTransposedKeyTensorShape {batchSize, numHeads, headSize, sequenceLength};
        std::vector<uint32_t> reshapedTransposedKeyTensorStride {sequenceLength * numHeads * headSize, headSize, 1, numHeads * headSize};
        TensorDesc reshapedTransposedKeyTensorDesc = TensorDesc(
                GetDmlDataTypeFromMlDataType(dataType),
                gsl::make_span(reshapedTransposedKeyTensorShape),
                gsl::make_span(reshapedTransposedKeyTensorStride),
                0 // guaranteedBaseOffsetAlignment
            );
        
        std::vector<uint32_t> queryKeyTensorShape {batchSize, numHeads, sequenceLength, sequenceLength};
        TensorDesc queryKeyTensorDesc = TensorDesc(
                dataType,
                gsl::make_span(queryKeyTensorShape),
                gsl::make_span(queryKeyTensorShape),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                1, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );

        float alpha = static_cast<float>(1 / sqrt(headSize));
        DML_GEMM_OPERATOR_DESC attentionScoreOperatorDesc = {};
        attentionScoreOperatorDesc.ATensor = &reshapedTransposedQueryTensorDesc.GetDmlDesc();
        attentionScoreOperatorDesc.BTensor = &reshapedTransposedKeyTensorDesc.GetDmlDesc();
        attentionScoreOperatorDesc.CTensor = &castedMaskIndexTensorDesc.GetDmlDesc();
        attentionScoreOperatorDesc.OutputTensor = &queryKeyTensorDesc.GetDmlDesc();
        attentionScoreOperatorDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
        attentionScoreOperatorDesc.TransB = DML_MATRIX_TRANSFORM_NONE;
        attentionScoreOperatorDesc.Alpha = alpha;
        attentionScoreOperatorDesc.Beta = 0.0f;
        attentionScoreOperatorDesc.FusedActivation = nullptr;
        const DML_OPERATOR_DESC attentionScoreDesc {DML_OPERATOR_GEMM, &attentionScoreOperatorDesc};

        std::vector<uint32_t> axes {3};
        DML_ACTIVATION_SOFTMAX1_OPERATOR_DESC softmaxOperatorDesc = {};
        softmaxOperatorDesc.InputTensor = &queryKeyTensorDesc.GetDmlDesc();
        softmaxOperatorDesc.OutputTensor = &queryKeyTensorDesc.GetDmlDesc();
        softmaxOperatorDesc.AxisCount = 1;
        softmaxOperatorDesc.Axes = axes.data();
        const DML_OPERATOR_DESC softmaxDesc = {DML_OPERATOR_ACTIVATION_SOFTMAX1, &softmaxOperatorDesc};

        std::vector<uint32_t> reshapedTransposedOutputTensorShape {batchSize, numHeads, sequenceLength, headSize};
        std::vector<uint32_t> reshapedTransposedOutputTensorStride {sequenceLength * numHeads * headSize, headSize * sequenceLength, headSize, 1};
        TensorDesc reshapedTransposedOutputTensorDesc = TensorDesc(
                GetDmlDataTypeFromMlDataType(dataType),
                gsl::make_span(reshapedTransposedOutputTensorShape),
                gsl::make_span(reshapedTransposedOutputTensorStride),
                0 // guaranteedBaseOffsetAlignment
            );

        DML_GEMM_OPERATOR_DESC attentionWeightOperatorDesc = {};
        attentionWeightOperatorDesc.ATensor = &queryKeyTensorDesc.GetDmlDesc();
        attentionWeightOperatorDesc.BTensor = &reshapedTransposedQueryTensorDesc.GetDmlDesc();
        attentionWeightOperatorDesc.CTensor = nullptr;
        attentionWeightOperatorDesc.OutputTensor = &reshapedTransposedOutputTensorDesc.GetDmlDesc();
        attentionWeightOperatorDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
        attentionWeightOperatorDesc.TransB = DML_MATRIX_TRANSFORM_NONE;
        attentionWeightOperatorDesc.Alpha = 1.0f;
        attentionWeightOperatorDesc.Beta = 0.0f;
        attentionWeightOperatorDesc.FusedActivation = nullptr;
        const DML_OPERATOR_DESC attentionWeightDesc {DML_OPERATOR_GEMM, &attentionWeightOperatorDesc};

        TensorDesc finalOutputTensorDesc = TensorDesc(
                m_outputTensorDescs[0].GetDmlDataType(),
                m_outputTensorDescs[0].GetSizes(),
                std::nullopt,
                0 // guaranteedBaseOffsetAlignment
            );
        DML_ACTIVATION_LINEAR_OPERATOR_DESC outputOperatorDesc = {};
        outputOperatorDesc.Alpha = 1.0f;
        outputOperatorDesc.Beta = 0.0f;
        outputOperatorDesc.InputTensor = &outputDescs[0];
        outputOperatorDesc.OutputTensor = &finalOutputTensorDesc.GetDmlDesc();
        const DML_OPERATOR_DESC outputDesc {DML_OPERATOR_ACTIVATION_LINEAR, &outputOperatorDesc};


        MLOperatorGraphDesc operatorGraphDesc = {};
        operatorGraphDesc.nodeCount = 10;
        std::vector<const DML_OPERATOR_DESC*> opDescs{&xWeightDesc, &querySlicedDesc, &keySlicedDesc, &valueSlicedDesc, &attentionScoreDesc, &softmaxDesc, &attentionWeightDesc, &castMaskIndexDesc, &maskDesc, &outputDesc};
        operatorGraphDesc.nodesAsOpDesc = opDescs.data();

        // set input edges
        std::pair<uint32_t, uint32_t> nodeToNodeInputIndex[4] {{0, 0}, {0, 1}, {0, 2}, {7, 0}};
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges(4);
        for (uint32_t inputIndex = 0; inputIndex < 4; inputIndex++)
        {
            DML_INPUT_GRAPH_EDGE_DESC inputEdge = {};
            inputEdge.GraphInputIndex = inputIndex;
            inputEdge.ToNodeIndex = nodeToNodeInputIndex[inputIndex].first;
            inputEdge.ToNodeInputIndex = nodeToNodeInputIndex[inputIndex].second;
            inputEdges[inputIndex] = inputEdge;
        }
        operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
        operatorGraphDesc.inputEdges = inputEdges.data();

        // set intermediate edges
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        
        DML_INTERMEDIATE_GRAPH_EDGE_DESC gemmToQuerySliceEdge = {};
        gemmToQuerySliceEdge.FromNodeIndex = 0;
        gemmToQuerySliceEdge.FromNodeOutputIndex = 0;
        gemmToQuerySliceEdge.ToNodeIndex = 1;
        gemmToQuerySliceEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(gemmToQuerySliceEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC gemmToKeySliceEdge = {};
        gemmToKeySliceEdge.FromNodeIndex = 0;
        gemmToKeySliceEdge.FromNodeOutputIndex = 0;
        gemmToKeySliceEdge.ToNodeIndex = 2;
        gemmToKeySliceEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(gemmToKeySliceEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC gemmToValueSliceEdge = {};
        gemmToValueSliceEdge.FromNodeIndex = 0;
        gemmToValueSliceEdge.FromNodeOutputIndex = 0;
        gemmToValueSliceEdge.ToNodeIndex = 3;
        gemmToValueSliceEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(gemmToValueSliceEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC querySliceToGemm = {};
        querySliceToGemm.FromNodeIndex = 1;
        querySliceToGemm.FromNodeOutputIndex = 0;
        querySliceToGemm.ToNodeIndex = 4;
        querySliceToGemm.ToNodeInputIndex = 0;
        intermediateEdges.push_back(querySliceToGemm);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC keySliceToGemm = {};
        keySliceToGemm.FromNodeIndex = 2;
        keySliceToGemm.FromNodeOutputIndex = 0;
        keySliceToGemm.ToNodeIndex = 4;
        keySliceToGemm.ToNodeInputIndex = 1;
        intermediateEdges.push_back(keySliceToGemm);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC castedMaskIndexToIdentity = {};
        castedMaskIndexToIdentity.FromNodeIndex = 7;
        castedMaskIndexToIdentity.FromNodeOutputIndex = 0;
        castedMaskIndexToIdentity.ToNodeIndex = 8;
        castedMaskIndexToIdentity.ToNodeInputIndex = 0;
        intermediateEdges.push_back(castedMaskIndexToIdentity);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC maskToGemm = {};
        maskToGemm.FromNodeIndex = 8;
        maskToGemm.FromNodeOutputIndex = 0;
        maskToGemm.ToNodeIndex = 4;
        maskToGemm.ToNodeInputIndex = 2;
        intermediateEdges.push_back(maskToGemm);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC attentionScoreToSoftmax = {};
        attentionScoreToSoftmax.FromNodeIndex = 4;
        attentionScoreToSoftmax.FromNodeOutputIndex = 0;
        attentionScoreToSoftmax.ToNodeIndex = 5;
        attentionScoreToSoftmax.ToNodeInputIndex = 0;
        intermediateEdges.push_back(attentionScoreToSoftmax);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC softmaxToGemm = {};
        softmaxToGemm.FromNodeIndex = 5;
        softmaxToGemm.FromNodeOutputIndex = 0;
        softmaxToGemm.ToNodeIndex = 6;
        softmaxToGemm.ToNodeInputIndex = 0;
        intermediateEdges.push_back(softmaxToGemm);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC valueSliceToGemm = {};
        valueSliceToGemm.FromNodeIndex = 3;
        valueSliceToGemm.FromNodeOutputIndex = 0;
        valueSliceToGemm.ToNodeIndex = 6;
        valueSliceToGemm.ToNodeInputIndex = 1;
        intermediateEdges.push_back(valueSliceToGemm);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC gemmToIdentity = {};
        gemmToIdentity.FromNodeIndex = 6;
        gemmToIdentity.FromNodeOutputIndex = 0;
        gemmToIdentity.ToNodeIndex = 9;
        gemmToIdentity.ToNodeInputIndex = 0;
        intermediateEdges.push_back(gemmToIdentity);

        operatorGraphDesc.intermediateEdgeCount = gsl::narrow_cast<uint32_t>(intermediateEdges.size());
        operatorGraphDesc.intermediateEdges = intermediateEdges.data();

        // set the output edges
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
        outputEdge.FromNodeIndex = 9;
        outputEdge.FromNodeOutputIndex = 0;
        outputEdge.GraphOutputIndex = 0;
        outputEdges.push_back(outputEdge);
        operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
        operatorGraphDesc.outputEdges = outputEdges.data();

        SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelCreationContext);
    }
};

void CALLBACK QueryAttention(IMLOperatorSupportQueryContextPrivate* context, /*out*/ bool* isSupported)
{
    *isSupported = false;
    if (context->GetInputCount() > 4 || context->GetOutputCount() > 1)
    {
        return;
    }

    if (context->IsInputValid(4) || context->IsInputValid(5))
    {
        return;
    }

    MLOperatorAttributes attributes(context);
    if (attributes.HasAttribute(AttrName::QkvHiddenSize, MLOperatorAttributeType::IntArray))
    {
        return;
    }

    if (attributes.GetOptionalAttribute<int32_t>(AttrName::Unidirectional, 0) != 0)
    {
        return;
    }

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(Attention, DmlOperatorAttention);
} // namespace Dml

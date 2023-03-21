// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

/*
Abbreviations: B is batch_size, S is sequence_length, W is hidden_size
               N is number of attention heads, H is head size, and W=N*H
               M is mask_index tensor

     M               A  B  C    // M, A, B, and C are Inputs
     |                \ |  /
    Cast               Gemm
     |                / |   \
     |               /  |    \
     |              /   |     \
     |          Slice  Slice  Slice
  Identity        |     |       |
     |            |     |       |
     |      Identity Identity Identity // The identities are used to transpose NCHW -> NHCW while
     |            |     |       |      // keeping the GEMM strides as NCHW to better target metacommands
     |            |     |       |
     |             -----        |
     -----------    |           |
                 \  |           |
                  Gemm          |
                    |           |
                    |           |
                Softmax         |
                    |          /
                    |         /
                     \       /
                       \    /
                        Gemm
                          |
                   ActivationLinear
                          |
                        Output  // Final output

 This kernel creates a DML_GRAPH, as mentioned above.
 For reference, refer to this Doc:
 https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#com.microsoft.Attention
 */
namespace Dml
{
class DmlOperatorAttention : public DmlOperator
{
public:
    DmlOperatorAttention(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        enum InputIndex : uint32_t
        {
            inputIndex,
            weightsIndex,
            biasIndex,
            maskIndex,
            pastIndex,
            relativePositionBiasIndex,
            pastSequenceLengthIndex,
            inputCount,
        };

        enum OutputIndex : uint32_t
        {
            outputIndex,
            outputCount,
        };

        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() >= 2);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() >= 1);

        bool maskIsUnpaddedBounds =
            kernelCreationContext.IsInputValid(maskIndex) &&
            kernelCreationContext.GetInputTensorDimensionCount(maskIndex) == 1;

        uint32_t dmlInputIndex = inputIndex;
        uint32_t dmlWeightsIndex = weightsIndex;
        uint32_t dmlBiasIndex = biasIndex;
        uint32_t dmlKeyPaddingMaskIndex = maskIndex;
        uint32_t dmlUnpaddedKeyBoundsIndex = maskIndex;
        uint32_t dmlRelativePositionBiasIndex = relativePositionBiasIndex;

        const bool hasBias = kernelCreationContext.IsInputValid(biasIndex);
        const bool hasMask = kernelCreationContext.IsInputValid(maskIndex) && !maskIsUnpaddedBounds;
        const bool hasUnpaddedBounds = kernelCreationContext.IsInputValid(maskIndex) && maskIsUnpaddedBounds;
        const bool hasRelativePositionBias = kernelCreationContext.IsInputValid(relativePositionBiasIndex);

        DmlOperator::Initialize(kernelCreationContext, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 1);

        const uint32_t numHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::NumHeads));
        ML_CHECK_VALID_ARGUMENT(numHeads > 0); // to avoid process crash because of division by zero.

        auto inputTensorShape = m_inputTensorDescs[dmlInputIndex].GetSizes();
        ML_CHECK_VALID_ARGUMENT(inputTensorShape.size() == 3);

        auto weightTensorShape = m_inputTensorDescs[dmlWeightsIndex].GetSizes();
        ML_CHECK_VALID_ARGUMENT(weightTensorShape.size() == 2);
        ML_CHECK_VALID_ARGUMENT(weightTensorShape[0] == inputTensorShape[2]);

        const auto qkvHiddenSizes = kernelCreationContext.GetOptionalAttributeVectorInt32(AttrName::QkvHiddenSizes);
        if (hasBias)
        {
            auto biasTensorShape = m_inputTensorDescs[dmlBiasIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(biasTensorShape.size() == 1);
            ML_CHECK_VALID_ARGUMENT(weightTensorShape[1] == biasTensorShape[0]);

            if (qkvHiddenSizes.empty())
            {
                ML_CHECK_VALID_ARGUMENT(biasTensorShape[0] % 3 == 0);
            }
        }

        if (!qkvHiddenSizes.empty())
        {
            ML_CHECK_VALID_ARGUMENT(qkvHiddenSizes.size() == 3);
            ML_CHECK_VALID_ARGUMENT(qkvHiddenSizes[0] == qkvHiddenSizes[1]);
        }
        else
        {
            ML_CHECK_VALID_ARGUMENT(weightTensorShape[1] % 3 == 0);
        }

        const uint32_t hiddenSize = qkvHiddenSizes.empty() ? weightTensorShape[1] / 3 : qkvHiddenSizes[0];
        const uint32_t vHiddenSize = qkvHiddenSizes.empty() ? weightTensorShape[1] / 3 : qkvHiddenSizes[2];
        const uint32_t headSize = hiddenSize / numHeads;
        const uint32_t vHeadSize = vHiddenSize / numHeads;
        const uint32_t batchSize = inputTensorShape[0];
        const uint32_t sequenceLength = inputTensorShape[1];

        uint32_t desiredWeightTensorShape[3] = {batchSize, weightTensorShape[0], hiddenSize + hiddenSize + vHiddenSize};
        MLOperatorTensorDataType dataType = kernelCreationContext.GetInputEdgeDescription(inputIndex).tensorDataType;

        m_inputTensorDescs[dmlWeightsIndex] = TensorDesc::ConstructBroadcastedTensorDesc(dataType, desiredWeightTensorShape, weightTensorShape);

        uint32_t desiredBiasTensorShape[3] = {batchSize, sequenceLength, hiddenSize + hiddenSize + vHiddenSize};
        if (hasBias)
        {
            auto biasTensorShape = m_inputTensorDescs[dmlBiasIndex].GetSizes();
            m_inputTensorDescs[dmlBiasIndex] = TensorDesc::ConstructBroadcastedTensorDesc(dataType, desiredBiasTensorShape, biasTensorShape);
        }

        MLOperatorTensorDataType maskTensorDataType = MLOperatorTensorDataType::Undefined;
        bool hasMaxSequenceMask = false;
        if (hasMask)
        {
            auto maskIndexTensorShape = m_inputTensorDescs[dmlKeyPaddingMaskIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(maskIndexTensorShape.size() > 1 && maskIndexTensorShape.size() <= 4);

            std::vector<uint32_t> reshapedMaskIndexTensorShape(maskIndexTensorShape.begin(), maskIndexTensorShape.end());
            if (maskIndexTensorShape.size() == 4 && maskIndexTensorShape[2] != sequenceLength)
            {
                hasMaxSequenceMask = true;
                ML_CHECK_VALID_ARGUMENT(maskIndexTensorShape[2] == maskIndexTensorShape[3]);
                const uint32_t maxSequenceLength = maskIndexTensorShape[2];
                uint32_t desiredMaskIndexShape[4] {batchSize, numHeads, maxSequenceLength, maxSequenceLength};
                maskTensorDataType = kernelCreationContext.GetInputEdgeDescription(maskIndex).tensorDataType;
                m_inputTensorDescs[dmlKeyPaddingMaskIndex] = TensorDesc::ConstructBroadcastedTensorDesc(maskTensorDataType, desiredMaskIndexShape, reshapedMaskIndexTensorShape);
            }
            else
            {
                uint32_t maskIndexDimensionCount = gsl::narrow_cast<uint32_t>(maskIndexTensorShape.size());
                reshapedMaskIndexTensorShape.insert(reshapedMaskIndexTensorShape.begin() + 1, 4 - maskIndexDimensionCount, 1);
                uint32_t desiredMaskIndexShape[4] {batchSize, numHeads, sequenceLength, sequenceLength};
                maskTensorDataType = kernelCreationContext.GetInputEdgeDescription(maskIndex).tensorDataType;
                m_inputTensorDescs[dmlKeyPaddingMaskIndex] = TensorDesc::ConstructBroadcastedTensorDesc(maskTensorDataType, desiredMaskIndexShape, reshapedMaskIndexTensorShape);
            }

        }

        if (hasUnpaddedBounds)
        {
            auto unpaddedKeyBoundsShape = m_inputTensorDescs[dmlUnpaddedKeyBoundsIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(unpaddedKeyBoundsShape.size() == 1);
            ML_CHECK_VALID_ARGUMENT(unpaddedKeyBoundsShape[0] % batchSize == 0);

            uint32_t desiredShape[2] = {unpaddedKeyBoundsShape[0] / batchSize, batchSize};
            m_inputTensorDescs[dmlUnpaddedKeyBoundsIndex] = TensorDesc(
                m_inputTensorDescs[dmlUnpaddedKeyBoundsIndex].GetDmlDataType(),
                desiredShape);
        }

        if (hasRelativePositionBias)
        {
            auto relativePositionBiasTensorShape = m_inputTensorDescs[dmlRelativePositionBiasIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(relativePositionBiasTensorShape.size() == 4);
            ML_CHECK_VALID_ARGUMENT(relativePositionBiasTensorShape[0] == inputTensorShape[0]);
            ML_CHECK_VALID_ARGUMENT(relativePositionBiasTensorShape[1] == numHeads);
            ML_CHECK_VALID_ARGUMENT(relativePositionBiasTensorShape[2] == inputTensorShape[1]);
        }

        TensorDesc firstGemmOutputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, desiredBiasTensorShape);
        DML_TENSOR_DESC namedFirstGemmOutputTensorDesc = firstGemmOutputTensorDesc.GetDmlDesc();

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_GEMM_OPERATOR_DESC gemmOperatorDesc = {};
        gemmOperatorDesc.ATensor = &inputDescs[0];
        gemmOperatorDesc.BTensor = &inputDescs[1];

        if (hasBias)
        {
            gemmOperatorDesc.CTensor = &inputDescs[2];
        }

        gemmOperatorDesc.OutputTensor = &namedFirstGemmOutputTensorDesc;
        gemmOperatorDesc.TransA = DML_MATRIX_TRANSFORM_NONE;
        gemmOperatorDesc.TransB = DML_MATRIX_TRANSFORM_NONE;
        gemmOperatorDesc.Alpha = 1.0f;
        gemmOperatorDesc.Beta = 1.0f;
        gemmOperatorDesc.FusedActivation = nullptr;
        const DML_OPERATOR_DESC gemmDesc {DML_OPERATOR_GEMM, &gemmOperatorDesc};

        std::array<uint32_t, 3> queryKeySlicedTensorShape {batchSize, sequenceLength, hiddenSize};
        TensorDesc queryKeySlicedInputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, queryKeySlicedTensorShape);
        DML_TENSOR_DESC namedQueryKeySlicedInputTensorDesc = queryKeySlicedInputTensorDesc.GetDmlDesc();

        std::array<uint32_t, 3> valueSlicedTensorShape {batchSize, sequenceLength, vHiddenSize};
        TensorDesc valueSlicedInputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, valueSlicedTensorShape);
        DML_TENSOR_DESC namedValueSlicedInputTensorDesc = valueSlicedInputTensorDesc.GetDmlDesc();

        std::array<uint32_t, 3> querySliceOffset = {0, 0, 0};
        std::array<uint32_t, 3> keySliceOffset = {0, 0, hiddenSize};
        std::array<uint32_t, 3> valueSliceOffset = {0, 0, 2 * hiddenSize};
        std::array<uint32_t, 3> queryKeySliceSize = {batchSize, sequenceLength, hiddenSize};
        std::array<uint32_t, 3> valueSliceSize = {batchSize, sequenceLength, vHiddenSize};
        std::array<int32_t, 3> strides = {1, 1, 1};
        DML_SLICE1_OPERATOR_DESC querySlicedOperatorDesc = {};
        querySlicedOperatorDesc.InputTensor = &namedFirstGemmOutputTensorDesc;
        querySlicedOperatorDesc.OutputTensor = &namedQueryKeySlicedInputTensorDesc;
        querySlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(queryKeySlicedTensorShape.size());
        querySlicedOperatorDesc.InputWindowOffsets = querySliceOffset.data();
        querySlicedOperatorDesc.InputWindowSizes = queryKeySliceSize.data();
        querySlicedOperatorDesc.InputWindowStrides = strides.data();
        const DML_OPERATOR_DESC querySlicedDesc = { DML_OPERATOR_SLICE1, &querySlicedOperatorDesc };

        DML_SLICE1_OPERATOR_DESC keySlicedOperatorDesc = {};
        keySlicedOperatorDesc.InputTensor = &namedFirstGemmOutputTensorDesc;
        keySlicedOperatorDesc.OutputTensor = &namedQueryKeySlicedInputTensorDesc;
        keySlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(queryKeySlicedTensorShape.size());
        keySlicedOperatorDesc.InputWindowOffsets = keySliceOffset.data();
        keySlicedOperatorDesc.InputWindowSizes = queryKeySliceSize.data();
        keySlicedOperatorDesc.InputWindowStrides = strides.data();
        const DML_OPERATOR_DESC keySlicedDesc = { DML_OPERATOR_SLICE1, &keySlicedOperatorDesc };

        DML_SLICE1_OPERATOR_DESC valueSlicedOperatorDesc = {};
        valueSlicedOperatorDesc.InputTensor = &namedFirstGemmOutputTensorDesc;
        valueSlicedOperatorDesc.OutputTensor = &namedValueSlicedInputTensorDesc;
        valueSlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(valueSlicedTensorShape.size());
        valueSlicedOperatorDesc.InputWindowOffsets = valueSliceOffset.data();
        valueSlicedOperatorDesc.InputWindowSizes = valueSliceSize.data();
        valueSlicedOperatorDesc.InputWindowStrides = strides.data();
        const DML_OPERATOR_DESC valueSlicedDesc = { DML_OPERATOR_SLICE1, &valueSlicedOperatorDesc};

        std::array<uint32_t, 4> maskSliceOutputShape {batchSize, numHeads, sequenceLength, sequenceLength};
        std::array<int32_t, 4> maskSliceStrides = {1, 1, 1, 1};
        std::array<uint32_t, 4> maskSliceOffsets = {0, 0, 0, 0};
        TensorDesc maskSliceOutputTensorDesc;
        DML_TENSOR_DESC namedMaskSliceOutputTensorDesc;

        DML_SLICE1_OPERATOR_DESC maskSlicedOperatorDesc = {};
        if (hasMaxSequenceMask)
        {
            maskSliceOutputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(maskTensorDataType, maskSliceOutputShape);
            namedMaskSliceOutputTensorDesc = maskSliceOutputTensorDesc.GetDmlDesc();
            maskSlicedOperatorDesc.InputTensor = &inputDescs[dmlKeyPaddingMaskIndex];
            maskSlicedOperatorDesc.OutputTensor = &namedMaskSliceOutputTensorDesc;
            maskSlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(maskSliceOutputShape.size());
            maskSlicedOperatorDesc.InputWindowOffsets = maskSliceOffsets.data();
            maskSlicedOperatorDesc.InputWindowSizes = maskSliceOutputShape.data();
            maskSlicedOperatorDesc.InputWindowStrides = maskSliceStrides.data();
        }
        const DML_OPERATOR_DESC maskSlicedDesc = { DML_OPERATOR_SLICE1, &maskSlicedOperatorDesc};

        DML_MULTI_HEAD_ATTENTION_OPERATOR_DESC mhaOperatorDesc = {};
        mhaOperatorDesc.QueryTensor = &namedQueryKeySlicedInputTensorDesc;
        mhaOperatorDesc.KeyTensor = &namedQueryKeySlicedInputTensorDesc;
        mhaOperatorDesc.ValueTensor = &namedValueSlicedInputTensorDesc;

        if (hasMaxSequenceMask)
        {
            mhaOperatorDesc.MaskTensor = &namedMaskSliceOutputTensorDesc;
        }
        else
        {
            mhaOperatorDesc.MaskTensor = hasMask ? &inputDescs[dmlKeyPaddingMaskIndex] : nullptr;
        }

        mhaOperatorDesc.UnpaddedKeyBoundsTensor = hasUnpaddedBounds ? &inputDescs[dmlUnpaddedKeyBoundsIndex] : nullptr;
        mhaOperatorDesc.RelativePositionBiasTensor = hasRelativePositionBias ? &inputDescs[dmlRelativePositionBiasIndex] : nullptr;
        mhaOperatorDesc.OutputTensor = &outputDescs[outputIndex];
        mhaOperatorDesc.Scale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Scale, gsl::narrow_cast<float>(1.0f / std::sqrt(headSize)));
        mhaOperatorDesc.MaskFilterValue = kernelCreationContext.GetOptionalAttribute<float>(AttrName::MaskFilterValue, -10'000.0f);
        mhaOperatorDesc.HeadCount = numHeads;
        const DML_OPERATOR_DESC mhaDesc = { DML_OPERATOR_MULTI_HEAD_ATTENTION, &mhaOperatorDesc };

        // Construct the graph
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

        enum NodeIndex : uint32_t
        {
            gemm,
            querySlice,
            keySlice,
            valueSlice,
            mha,
            maskSlice,
            count,
        };

        std::vector<const DML_OPERATOR_DESC*> opDescs = {
            &gemmDesc,
            &querySlicedDesc,
            &keySlicedDesc,
            &valueSlicedDesc,
            &mhaDesc,
        };

        if (hasMaxSequenceMask)
        {
            opDescs.push_back(&maskSlicedDesc);
        }

        DML_INPUT_GRAPH_EDGE_DESC inputToGemmEdge = {};
        inputToGemmEdge.GraphInputIndex = dmlInputIndex;
        inputToGemmEdge.ToNodeIndex = NodeIndex::gemm;
        inputToGemmEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(inputToGemmEdge);

        DML_INPUT_GRAPH_EDGE_DESC weightToGemmEdge = {};
        weightToGemmEdge.GraphInputIndex = dmlWeightsIndex;
        weightToGemmEdge.ToNodeIndex = NodeIndex::gemm;
        weightToGemmEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(weightToGemmEdge);

        if (hasBias)
        {
            DML_INPUT_GRAPH_EDGE_DESC biasToGemmEdge = {};
            biasToGemmEdge.GraphInputIndex = dmlBiasIndex;
            biasToGemmEdge.ToNodeIndex = NodeIndex::gemm;
            biasToGemmEdge.ToNodeInputIndex = 2;
            inputEdges.push_back(biasToGemmEdge);
        }

        if (hasMask)
        {
            if (hasMaxSequenceMask)
            {
                DML_INPUT_GRAPH_EDGE_DESC maskToMaskSliceEdge = {};
                maskToMaskSliceEdge.GraphInputIndex = dmlKeyPaddingMaskIndex;
                maskToMaskSliceEdge.ToNodeIndex = NodeIndex::maskSlice;
                maskToMaskSliceEdge.ToNodeInputIndex = 0;
                inputEdges.push_back(maskToMaskSliceEdge);

                DML_INTERMEDIATE_GRAPH_EDGE_DESC maskSliceToMhaEdge = {};
                maskSliceToMhaEdge.FromNodeIndex = NodeIndex::maskSlice;
                maskSliceToMhaEdge.FromNodeOutputIndex = 0;
                maskSliceToMhaEdge.ToNodeIndex = NodeIndex::mha;
                maskSliceToMhaEdge.ToNodeInputIndex = 4;
                intermediateEdges.push_back(maskSliceToMhaEdge);
            }
            else
            {
                DML_INPUT_GRAPH_EDGE_DESC maskToMhaEdge = {};
                maskToMhaEdge.GraphInputIndex = dmlKeyPaddingMaskIndex;
                maskToMhaEdge.ToNodeIndex = NodeIndex::mha;
                maskToMhaEdge.ToNodeInputIndex = 4;
                inputEdges.push_back(maskToMhaEdge);
            }
        }

        if (hasUnpaddedBounds)
        {
            DML_INPUT_GRAPH_EDGE_DESC maskToMhaEdge = {};
            maskToMhaEdge.GraphInputIndex = dmlUnpaddedKeyBoundsIndex;
            maskToMhaEdge.ToNodeIndex = NodeIndex::mha;
            maskToMhaEdge.ToNodeInputIndex = 5;
            inputEdges.push_back(maskToMhaEdge);
        }

        if (hasRelativePositionBias)
        {
            DML_INPUT_GRAPH_EDGE_DESC relativePositionBiasToMhaEdge = {};
            relativePositionBiasToMhaEdge.GraphInputIndex = dmlRelativePositionBiasIndex;
            relativePositionBiasToMhaEdge.ToNodeIndex = NodeIndex::mha;
            relativePositionBiasToMhaEdge.ToNodeInputIndex = 6;
            inputEdges.push_back(relativePositionBiasToMhaEdge);
        }

        DML_INTERMEDIATE_GRAPH_EDGE_DESC gemmToQuerySliceEdge = {};
        gemmToQuerySliceEdge.FromNodeIndex = NodeIndex::gemm;
        gemmToQuerySliceEdge.FromNodeOutputIndex = 0;
        gemmToQuerySliceEdge.ToNodeIndex = NodeIndex::querySlice;
        gemmToQuerySliceEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(gemmToQuerySliceEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC gemmToKeySliceEdge = {};
        gemmToKeySliceEdge.FromNodeIndex = NodeIndex::gemm;
        gemmToKeySliceEdge.FromNodeOutputIndex = 0;
        gemmToKeySliceEdge.ToNodeIndex = NodeIndex::keySlice;
        gemmToKeySliceEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(gemmToKeySliceEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC gemmToValueSliceEdge = {};
        gemmToValueSliceEdge.FromNodeIndex = NodeIndex::gemm;
        gemmToValueSliceEdge.FromNodeOutputIndex = 0;
        gemmToValueSliceEdge.ToNodeIndex = NodeIndex::valueSlice;
        gemmToValueSliceEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(gemmToValueSliceEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC querySliceEdgeToMhaEdge = {};
        querySliceEdgeToMhaEdge.FromNodeIndex = NodeIndex::querySlice;
        querySliceEdgeToMhaEdge.FromNodeOutputIndex = 0;
        querySliceEdgeToMhaEdge.ToNodeIndex = NodeIndex::mha;
        querySliceEdgeToMhaEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(querySliceEdgeToMhaEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC keySliceEdgeToMhaEdge = {};
        keySliceEdgeToMhaEdge.FromNodeIndex = NodeIndex::keySlice;
        keySliceEdgeToMhaEdge.FromNodeOutputIndex = 0;
        keySliceEdgeToMhaEdge.ToNodeIndex = NodeIndex::mha;
        keySliceEdgeToMhaEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(keySliceEdgeToMhaEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC valueSliceEdgeToMhaEdge = {};
        valueSliceEdgeToMhaEdge.FromNodeIndex = NodeIndex::valueSlice;
        valueSliceEdgeToMhaEdge.FromNodeOutputIndex = 0;
        valueSliceEdgeToMhaEdge.ToNodeIndex = NodeIndex::mha;
        valueSliceEdgeToMhaEdge.ToNodeInputIndex = 2;
        intermediateEdges.push_back(valueSliceEdgeToMhaEdge);

        DML_OUTPUT_GRAPH_EDGE_DESC mhaToOutputEdge = {};
        mhaToOutputEdge.FromNodeIndex = NodeIndex::mha;
        mhaToOutputEdge.FromNodeOutputIndex = 0;
        mhaToOutputEdge.GraphOutputIndex = 0;
        outputEdges.push_back(mhaToOutputEdge);

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

void CALLBACK QueryAttention(IMLOperatorSupportQueryContextPrivate* context, /*out*/ bool* isSupported)
{
    *isSupported = false;
    // `past` input tensor is not supported yet
    if (context->IsInputValid(4))
    {
        return;
    }

    // `past_sequence_length` input tensor is not supported yet
    if (context->IsInputValid(6))
    {
        return;
    }

    // `present` output tensor is not supported yet
    if (context->IsOutputValid(1))
    {
        return;
    }

    // `unidirectional == 1` is not supported yet
    MLOperatorAttributes attributes(context);
    if (attributes.GetOptionalAttribute<int32_t>(AttrName::Unidirectional, 0) != 0)
    {
        return;
    }

    // `do_rotary == 1` is not supported yet
    if (attributes.GetOptionalAttribute<int32_t>(AttrName::DoRotary, 0) != 0)
    {
        return;
    }

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(Attention, DmlOperatorAttention);
} // namespace Dml

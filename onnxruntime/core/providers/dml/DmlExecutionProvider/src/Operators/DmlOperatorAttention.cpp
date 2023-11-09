// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

/*
Abbreviations: B is batch_size, S is sequence_length, W is hidden_size
               N is number of attention heads, H is head size, and W=N*H
               M is mask_index tensor

     M               A  B  C    // M, A, B, and C are Inputs
     |                \ |  /
     |                 Gemm
     |                / |   \
     |               /  |    \
     |              /   |     \
     |          Slice  Slice  Slice
     |            |     |       |
     |            |     |       |
     |      Identity Identity Identity // The identities are used to transpose NCHW -> NHCW while
     |            |     |       |      // keeping the GEMM strides as NCHW to better target metacommands
     |            |     |       |
     ----------------- MHA -----
                        |
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
        enum DmlInputIndex : uint32_t
        {
            mhaQueryIndex,
            mhaKeyIndex,
            mhaValueIndex,
            mhaStackedQueryKeyIndex,
            mhaStackedKeyValueIndex,
            mhaStackedQueryKeyValueIndex,
            mhaBiasIndex,
            mhaMaskIndex,
            mhaRelativePositionBiasIndex,
            mhaPastKeyIndex,
            mhaPastValueIndex,
            mhaInputCount,
        };

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

        const uint32_t dmlInputIndex = inputIndex;
        const uint32_t dmlWeightsIndex = weightsIndex;
        const uint32_t dmlBiasIndex = biasIndex;
        const uint32_t dmlMaskIndex = maskIndex;
        const uint32_t dmlRelativePositionBiasIndex = relativePositionBiasIndex;

        const bool hasBias = kernelCreationContext.IsInputValid(biasIndex);
        const bool hasMask = kernelCreationContext.IsInputValid(maskIndex);
        const bool hasUnpaddedBounds = hasMask && kernelCreationContext.GetInputTensorDimensionCount(maskIndex) == 1;
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
        DML_MULTIHEAD_ATTENTION_MASK_TYPE maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
        if (hasMask)
        {
            if (hasUnpaddedBounds)
            {
                auto unpaddedKeyBoundsShape = m_inputTensorDescs[dmlMaskIndex].GetSizes();
                ML_CHECK_VALID_ARGUMENT(unpaddedKeyBoundsShape.size() == 1);

                const uint32_t batchGroupCount = unpaddedKeyBoundsShape[0] / batchSize;
                ML_CHECK_VALID_ARGUMENT(batchGroupCount == 1 || batchGroupCount == 2);

                uint32_t desiredShape[2] = {batchGroupCount, batchSize};
                m_inputTensorDescs[dmlMaskIndex] = TensorDesc(
                    m_inputTensorDescs[dmlMaskIndex].GetDmlDataType(),
                    desiredShape);

                maskType = batchGroupCount == 1
                    ? DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH
                    : DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_END_START;
            }
            else
            {
                auto maskIndexTensorShape = m_inputTensorDescs[dmlMaskIndex].GetSizes();
                ML_CHECK_VALID_ARGUMENT(maskIndexTensorShape.size() > 1 && maskIndexTensorShape.size() <= 4);

                maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN;
                std::vector<uint32_t> reshapedMaskIndexTensorShape(maskIndexTensorShape.begin(), maskIndexTensorShape.end());
                if (maskIndexTensorShape.size() == 4 && maskIndexTensorShape[2] != sequenceLength)
                {
                    hasMaxSequenceMask = true;
                    ML_CHECK_VALID_ARGUMENT(maskIndexTensorShape[2] == maskIndexTensorShape[3]);
                    const uint32_t maxSequenceLength = maskIndexTensorShape[2];
                    uint32_t desiredMaskIndexShape[4] {batchSize, numHeads, maxSequenceLength, maxSequenceLength};
                    maskTensorDataType = kernelCreationContext.GetInputEdgeDescription(maskIndex).tensorDataType;
                    m_inputTensorDescs[dmlMaskIndex] = TensorDesc::ConstructBroadcastedTensorDesc(maskTensorDataType, desiredMaskIndexShape, reshapedMaskIndexTensorShape);
                }
                else
                {
                    uint32_t maskIndexDimensionCount = gsl::narrow_cast<uint32_t>(maskIndexTensorShape.size());
                    reshapedMaskIndexTensorShape.insert(reshapedMaskIndexTensorShape.begin() + 1, 4 - maskIndexDimensionCount, 1);
                    uint32_t desiredMaskIndexShape[4] {batchSize, numHeads, sequenceLength, sequenceLength};
                    maskTensorDataType = kernelCreationContext.GetInputEdgeDescription(maskIndex).tensorDataType;
                    m_inputTensorDescs[dmlMaskIndex] = TensorDesc::ConstructBroadcastedTensorDesc(maskTensorDataType, desiredMaskIndexShape, reshapedMaskIndexTensorShape);
                }
            }
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

        std::array<uint32_t, 3> queryKeySlicedTensorShape {batchSize, sequenceLength, hiddenSize + hiddenSize};
        TensorDesc queryKeySlicedInputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, queryKeySlicedTensorShape);
        DML_TENSOR_DESC namedQueryKeySlicedInputTensorDesc = queryKeySlicedInputTensorDesc.GetDmlDesc();

        std::array<uint32_t, 3> valueSlicedTensorShape {batchSize, sequenceLength, vHiddenSize};
        TensorDesc valueSlicedInputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, valueSlicedTensorShape);
        DML_TENSOR_DESC namedValueSlicedInputTensorDesc = valueSlicedInputTensorDesc.GetDmlDesc();

        // Transpose slice QK from [batchSize, sequenceLength, 2, numHeads, headSize] to [batchSize, sequenceLength, numHeads, 2, headSize]
        std::array<uint32_t, 5> queryKeyTransposedTensorShape {batchSize, sequenceLength, numHeads, 2, headSize};
        std::array<uint32_t, 5> queryKeyTransposedStrides {
            sequenceLength * numHeads * 2 * headSize,
            numHeads * 2 * headSize,
            headSize,
            numHeads * headSize,
            1,
        };

        TensorDesc queryKeyTransposedInputTensorDesc = TensorDesc(
            m_inputTensorDescs[dmlInputIndex].GetDmlDataType(),
            queryKeyTransposedTensorShape,
            queryKeyTransposedStrides);
        DML_TENSOR_DESC namedQueryKeyTransposedInputTensorDesc = queryKeyTransposedInputTensorDesc.GetDmlDesc();

        TensorDesc queryKeyTransposedOutputTensorDesc = TensorDesc(
            m_inputTensorDescs[dmlInputIndex].GetDmlDataType(),
            queryKeyTransposedTensorShape);
        DML_TENSOR_DESC namedQueryKeyTransposedOutputTensorDesc = queryKeyTransposedOutputTensorDesc.GetDmlDesc();

        // Transpose QKV from [batchSize, sequenceLength, 3, numHeads, headSize] to [batchSize, sequenceLength, numHeads, 3, headSize]
        std::array<uint32_t, 5> queryKeyValueTransposedTensorShape {batchSize, sequenceLength, numHeads, 3, headSize};
        std::array<uint32_t, 5> queryKeyValueTransposedStrides {
            sequenceLength * numHeads * 3 * headSize,
            numHeads * 3 * headSize,
            headSize,
            numHeads * headSize,
            1,
        };

        TensorDesc queryKeyValueTransposedInputTensorDesc = TensorDesc(
            m_inputTensorDescs[dmlInputIndex].GetDmlDataType(),
            queryKeyValueTransposedTensorShape,
            queryKeyValueTransposedStrides);
        DML_TENSOR_DESC namedQueryKeyValueTransposedInputTensorDesc = queryKeyValueTransposedInputTensorDesc.GetDmlDesc();

        TensorDesc queryKeyValueTransposedOutputTensorDesc = TensorDesc(
            m_inputTensorDescs[dmlInputIndex].GetDmlDataType(),
            queryKeyValueTransposedTensorShape);
        DML_TENSOR_DESC namedQueryKeyValueTransposedOutputTensorDesc = queryKeyValueTransposedOutputTensorDesc.GetDmlDesc();

        std::array<uint32_t, 3> queryKeySliceOffset = {0, 0, 0};
        std::array<uint32_t, 3> queryKeySliceSize = {batchSize, sequenceLength, hiddenSize + hiddenSize};
        std::array<int32_t, 3> queryKeySliceStrides = {1, 1, 1};

        std::array<uint32_t, 3> valueSliceOffset = {0, 0, 2 * hiddenSize};
        std::array<uint32_t, 3> valueSliceSize = {batchSize, sequenceLength, vHiddenSize};
        std::array<int32_t, 3> valueSliceStrides = {1, 1, 1};
        const bool hasSlicedValue = hiddenSize != vHiddenSize;

        // We need to slice the value tensor when its hidden size is different from the query and key
        DML_SLICE1_OPERATOR_DESC queryKeySlicedOperatorDesc = {};
        DML_SLICE1_OPERATOR_DESC valueSlicedOperatorDesc = {};
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC transposeOperatorDesc = {};
        if (hasSlicedValue)
        {
            queryKeySlicedOperatorDesc.InputTensor = &namedFirstGemmOutputTensorDesc;
            queryKeySlicedOperatorDesc.OutputTensor = &namedQueryKeySlicedInputTensorDesc;
            queryKeySlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(queryKeySlicedTensorShape.size());
            queryKeySlicedOperatorDesc.InputWindowOffsets = queryKeySliceOffset.data();
            queryKeySlicedOperatorDesc.InputWindowSizes = queryKeySliceSize.data();
            queryKeySlicedOperatorDesc.InputWindowStrides = queryKeySliceStrides.data();

            valueSlicedOperatorDesc.InputTensor = &namedFirstGemmOutputTensorDesc;
            valueSlicedOperatorDesc.OutputTensor = &namedValueSlicedInputTensorDesc;
            valueSlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(valueSlicedTensorShape.size());
            valueSlicedOperatorDesc.InputWindowOffsets = valueSliceOffset.data();
            valueSlicedOperatorDesc.InputWindowSizes = valueSliceSize.data();
            valueSlicedOperatorDesc.InputWindowStrides = valueSliceStrides.data();

            transposeOperatorDesc.InputTensor = &namedQueryKeyTransposedInputTensorDesc;
            transposeOperatorDesc.OutputTensor = &namedQueryKeyTransposedOutputTensorDesc;
        }
        else
        {
            // When Q/K/V all have the same hidden size, we just have to transpose it before sending it to MHA
            transposeOperatorDesc.InputTensor = &namedQueryKeyValueTransposedInputTensorDesc;
            transposeOperatorDesc.OutputTensor = &namedQueryKeyValueTransposedOutputTensorDesc;
        }
        const DML_OPERATOR_DESC queryKeySlicedDesc = { DML_OPERATOR_SLICE1, &queryKeySlicedOperatorDesc};
        const DML_OPERATOR_DESC valueSlicedDesc = { DML_OPERATOR_SLICE1, &valueSlicedOperatorDesc};
        const DML_OPERATOR_DESC transposedDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &transposeOperatorDesc};

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
            maskSlicedOperatorDesc.InputTensor = &inputDescs[dmlMaskIndex];
            maskSlicedOperatorDesc.OutputTensor = &namedMaskSliceOutputTensorDesc;
            maskSlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(maskSliceOutputShape.size());
            maskSlicedOperatorDesc.InputWindowOffsets = maskSliceOffsets.data();
            maskSlicedOperatorDesc.InputWindowSizes = maskSliceOutputShape.data();
            maskSlicedOperatorDesc.InputWindowStrides = maskSliceStrides.data();
        }
        const DML_OPERATOR_DESC maskSlicedDesc = { DML_OPERATOR_SLICE1, &maskSlicedOperatorDesc};

        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC mhaOperatorDesc = {};
        mhaOperatorDesc.ValueTensor = hasSlicedValue ? &namedValueSlicedInputTensorDesc : nullptr;
        mhaOperatorDesc.StackedQueryKeyTensor = hasSlicedValue ? &namedQueryKeyTransposedOutputTensorDesc : nullptr;
        mhaOperatorDesc.StackedQueryKeyValueTensor = hasSlicedValue ? nullptr : &namedQueryKeyValueTransposedOutputTensorDesc;

        if (hasMaxSequenceMask)
        {
            mhaOperatorDesc.MaskTensor = &namedMaskSliceOutputTensorDesc;
        }
        else
        {
            mhaOperatorDesc.MaskTensor = hasMask ? &inputDescs[dmlMaskIndex] : nullptr;
        }

        mhaOperatorDesc.RelativePositionBiasTensor = hasRelativePositionBias ? &inputDescs[dmlRelativePositionBiasIndex] : nullptr;
        mhaOperatorDesc.OutputTensor = &outputDescs[outputIndex];
        mhaOperatorDesc.Scale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Scale, gsl::narrow_cast<float>(1.0f / std::sqrt(headSize)));
        mhaOperatorDesc.MaskFilterValue = kernelCreationContext.GetOptionalAttribute<float>(AttrName::MaskFilterValue, -10'000.0f);
        mhaOperatorDesc.HeadCount = numHeads;
        mhaOperatorDesc.MaskType = maskType;
        const DML_OPERATOR_DESC mhaDesc = { DML_OPERATOR_MULTIHEAD_ATTENTION, &mhaOperatorDesc };

        // Construct the graph
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

        std::vector<const DML_OPERATOR_DESC*> opDescs = {
            &gemmDesc,
            &mhaDesc,
        };

        uint32_t currentNodeIndex = 0;
        const uint32_t gemmNodeIndex = currentNodeIndex++;
        const uint32_t mhaNodeIndex = currentNodeIndex++;

        uint32_t valueSliceNodeIndex = 0;
        uint32_t queryKeySliceNodeIndex = 0;
        uint32_t queryKeyTransposedNodeIndex = 0;
        uint32_t queryKeyValueTransposedNodeIndex = 0;
        if (hasSlicedValue)
        {
            opDescs.push_back(&queryKeySlicedDesc);
            queryKeySliceNodeIndex = currentNodeIndex++;

            opDescs.push_back(&valueSlicedDesc);
            valueSliceNodeIndex = currentNodeIndex++;

            opDescs.push_back(&transposedDesc);
            queryKeyTransposedNodeIndex = currentNodeIndex++;
        }
        else
        {
            opDescs.push_back(&transposedDesc);
            queryKeyValueTransposedNodeIndex = currentNodeIndex++;
        }

        uint32_t maskSliceNodeIndex = 0;
        if (hasMaxSequenceMask)
        {
            opDescs.push_back(&maskSlicedDesc);
            maskSliceNodeIndex = currentNodeIndex++;
        }

        DML_INPUT_GRAPH_EDGE_DESC inputToGemmEdge = {};
        inputToGemmEdge.GraphInputIndex = dmlInputIndex;
        inputToGemmEdge.ToNodeIndex = gemmNodeIndex;
        inputToGemmEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(inputToGemmEdge);

        DML_INPUT_GRAPH_EDGE_DESC weightToGemmEdge = {};
        weightToGemmEdge.GraphInputIndex = dmlWeightsIndex;
        weightToGemmEdge.ToNodeIndex = gemmNodeIndex;
        weightToGemmEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(weightToGemmEdge);

        if (hasBias)
        {
            DML_INPUT_GRAPH_EDGE_DESC biasToGemmEdge = {};
            biasToGemmEdge.GraphInputIndex = dmlBiasIndex;
            biasToGemmEdge.ToNodeIndex = gemmNodeIndex;
            biasToGemmEdge.ToNodeInputIndex = 2;
            inputEdges.push_back(biasToGemmEdge);
        }

        if (hasMask)
        {
            if (hasUnpaddedBounds)
            {
                DML_INPUT_GRAPH_EDGE_DESC maskToMhaEdge = {};
                maskToMhaEdge.GraphInputIndex = dmlMaskIndex;
                maskToMhaEdge.ToNodeIndex = mhaNodeIndex;
                maskToMhaEdge.ToNodeInputIndex = mhaMaskIndex;
                inputEdges.push_back(maskToMhaEdge);
            }
            else if (hasMaxSequenceMask)
            {
                DML_INPUT_GRAPH_EDGE_DESC maskToMaskSliceEdge = {};
                maskToMaskSliceEdge.GraphInputIndex = dmlMaskIndex;
                maskToMaskSliceEdge.ToNodeIndex = maskSliceNodeIndex;
                maskToMaskSliceEdge.ToNodeInputIndex = 0;
                inputEdges.push_back(maskToMaskSliceEdge);

                DML_INTERMEDIATE_GRAPH_EDGE_DESC maskSliceToMhaEdge = {};
                maskSliceToMhaEdge.FromNodeIndex = maskSliceNodeIndex;
                maskSliceToMhaEdge.FromNodeOutputIndex = 0;
                maskSliceToMhaEdge.ToNodeIndex = mhaNodeIndex;
                maskSliceToMhaEdge.ToNodeInputIndex = mhaMaskIndex;
                intermediateEdges.push_back(maskSliceToMhaEdge);
            }
            else
            {
                DML_INPUT_GRAPH_EDGE_DESC maskToMhaEdge = {};
                maskToMhaEdge.GraphInputIndex = dmlMaskIndex;
                maskToMhaEdge.ToNodeIndex = mhaNodeIndex;
                maskToMhaEdge.ToNodeInputIndex = mhaMaskIndex;
                inputEdges.push_back(maskToMhaEdge);
            }
        }

        if (hasRelativePositionBias)
        {
            DML_INPUT_GRAPH_EDGE_DESC relativePositionBiasToMhaEdge = {};
            relativePositionBiasToMhaEdge.GraphInputIndex = dmlRelativePositionBiasIndex;
            relativePositionBiasToMhaEdge.ToNodeIndex = mhaNodeIndex;
            relativePositionBiasToMhaEdge.ToNodeInputIndex = mhaRelativePositionBiasIndex;
            inputEdges.push_back(relativePositionBiasToMhaEdge);
        }

        if (hasSlicedValue)
        {
            // We need to slice QK and V, and transpose QK
            DML_INTERMEDIATE_GRAPH_EDGE_DESC gemmToQueryKeySliceEdge = {};
            gemmToQueryKeySliceEdge.FromNodeIndex = gemmNodeIndex;
            gemmToQueryKeySliceEdge.FromNodeOutputIndex = 0;
            gemmToQueryKeySliceEdge.ToNodeIndex = queryKeySliceNodeIndex;
            gemmToQueryKeySliceEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(gemmToQueryKeySliceEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC queryKeySliceToTransposeEdge = {};
            queryKeySliceToTransposeEdge.FromNodeIndex = queryKeySliceNodeIndex;
            queryKeySliceToTransposeEdge.FromNodeOutputIndex = 0;
            queryKeySliceToTransposeEdge.ToNodeIndex = queryKeyTransposedNodeIndex;
            queryKeySliceToTransposeEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(queryKeySliceToTransposeEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC queryKeyTransposedToMhaEdge = {};
            queryKeyTransposedToMhaEdge.FromNodeIndex = queryKeyTransposedNodeIndex;
            queryKeyTransposedToMhaEdge.FromNodeOutputIndex = 0;
            queryKeyTransposedToMhaEdge.ToNodeIndex = mhaNodeIndex;
            queryKeyTransposedToMhaEdge.ToNodeInputIndex = mhaStackedQueryKeyIndex;
            intermediateEdges.push_back(queryKeyTransposedToMhaEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC gemmToValueSliceEdge = {};
            gemmToValueSliceEdge.FromNodeIndex = gemmNodeIndex;
            gemmToValueSliceEdge.FromNodeOutputIndex = 0;
            gemmToValueSliceEdge.ToNodeIndex = valueSliceNodeIndex;
            gemmToValueSliceEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(gemmToValueSliceEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC valueSliceToMhaEdge = {};
            valueSliceToMhaEdge.FromNodeIndex = valueSliceNodeIndex;
            valueSliceToMhaEdge.FromNodeOutputIndex = 0;
            valueSliceToMhaEdge.ToNodeIndex = mhaNodeIndex;
            valueSliceToMhaEdge.ToNodeInputIndex = mhaValueIndex;
            intermediateEdges.push_back(valueSliceToMhaEdge);
        }
        else
        {
            DML_INTERMEDIATE_GRAPH_EDGE_DESC gemmToQueryKeyValueTransposeEdge = {};
            gemmToQueryKeyValueTransposeEdge.FromNodeIndex = gemmNodeIndex;
            gemmToQueryKeyValueTransposeEdge.FromNodeOutputIndex = 0;
            gemmToQueryKeyValueTransposeEdge.ToNodeIndex = queryKeyValueTransposedNodeIndex;
            gemmToQueryKeyValueTransposeEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(gemmToQueryKeyValueTransposeEdge);

            // All we need to do here is transpose the stacked QKV tensor into something DML supports
            DML_INTERMEDIATE_GRAPH_EDGE_DESC queryKeyValueTransposedToMhaEdge = {};
            queryKeyValueTransposedToMhaEdge.FromNodeIndex = queryKeyValueTransposedNodeIndex;
            queryKeyValueTransposedToMhaEdge.FromNodeOutputIndex = 0;
            queryKeyValueTransposedToMhaEdge.ToNodeIndex = mhaNodeIndex;
            queryKeyValueTransposedToMhaEdge.ToNodeInputIndex = mhaStackedQueryKeyValueIndex;
            intermediateEdges.push_back(queryKeyValueTransposedToMhaEdge);
        }

        DML_OUTPUT_GRAPH_EDGE_DESC mhaToOutputEdge = {};
        mhaToOutputEdge.FromNodeIndex = mhaNodeIndex;
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

    // `past_present_share_buffer == 1` is not supported yet
    if (attributes.GetOptionalAttribute<int32_t>(AttrName::PastPresentShareBuffer, 0) != 0)
    {
        return;
    }

    *isSupported = true;
}

DML_OP_DEFINE_CREATION_FUNCTION(Attention, DmlOperatorAttention);
} // namespace Dml

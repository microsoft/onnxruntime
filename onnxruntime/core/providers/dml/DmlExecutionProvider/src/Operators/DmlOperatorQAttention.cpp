// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

/*
Abbreviations: B is batch_size, S is sequence_length, W is hidden_size
               N is number of attention heads, H is head size, and W=N*H
               M is mask_index tensor

M, A, B, C and P are Inputs

     M               A  B      C
     |               |  |     /
     |             MatMulIntToFloat
     |                / |   \
     |               /  |    \
     |              /   |     \
     |          Slice  Slice  Slice
     |            |     |       |
     |            |     |       |
     |      Identity Identity Identity // The identities are used to transpose NCHW -> NHCW while
     |            |     |       |      // keeping the GEMM strides as NCHW to better target metacommands
     |            |     |       |
     |            |     |       |        P
     |            |     |       |       / \
     |            |     |       |      /   \
     |            |     |       |  Slice   Slice
     |            |     |       |     |      |
     |            |     |       |     |      |
     |            |     |       |     |      |
     --------------------------MHA -----------
                              / | \
                             /  |   \
                            /   |     \
                           /    |       \
                          /     |         \
                         /      |           \
                        /  presentKey   presentValue
                       /         \       /
                      /           \     /
                     /             \   /
                    /             Concat
                   /                 |
               Output1            Output2 (present)

 This kernel creates a DML_GRAPH, as mentioned above.
 For reference, refer to this Doc:
 https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftqattention
 */

namespace Dml
{
class DmlOperatorQAttention : public DmlOperator
{
public:
    DmlOperatorQAttention(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {

        enum InputIndex : uint32_t
        {
            inputIndex,
            weightsIndex,
            biasIndex,
            inputScaleIndex,
            weightScaleIndex,
            maskIndex,
            inputZeroPointIndex,
            weightZeroPointIndex,
            pastIndex,
            inputCount,
        };

        enum OutputIndex : uint32_t
        {
            outputIndex,
            presentIndex,
            outputCount,
        };

        enum MhaInputIndex : uint32_t
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

        enum MhaOutputIndex : uint32_t
        {
            mhaOutputIndex,
            mhaPresentKeyIndex,
            mhaPresentValueIndex,
            mhaOutputCount,
        };

        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() >= 5);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() >= 1);

        const bool hasBias = kernelCreationContext.IsInputValid(biasIndex);
        const bool hasMask = kernelCreationContext.IsInputValid(maskIndex);
        const bool hasUnpaddedBounds = hasMask && kernelCreationContext.GetInputTensorDimensionCount(maskIndex) == 1;
        const bool hasPast = kernelCreationContext.IsInputValid(pastIndex);

        DmlOperator::Initialize(kernelCreationContext, std::nullopt, std::nullopt, std::nullopt, std::nullopt, 1);

        const bool unidirectional = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::Unidirectional));
        const uint32_t numHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::NumHeads));
        ML_CHECK_VALID_ARGUMENT(numHeads > 0); // to avoid process crash because of division by zero.

        auto inputTensorShape = m_inputTensorDescs[inputIndex].GetSizes();
        ML_CHECK_VALID_ARGUMENT(inputTensorShape.size() == 3);

        auto weightTensorShape = m_inputTensorDescs[weightsIndex].GetSizes();
        ML_CHECK_VALID_ARGUMENT(weightTensorShape.size() == 2);
        ML_CHECK_VALID_ARGUMENT(weightTensorShape[0] == inputTensorShape[2]);
        ML_CHECK_VALID_ARGUMENT(weightTensorShape[1] % 3 == 0);

        if (hasBias)
        {
            auto biasTensorShape = m_inputTensorDescs[biasIndex].GetSizes();
            ML_CHECK_VALID_ARGUMENT(biasTensorShape.size() == 1);
            ML_CHECK_VALID_ARGUMENT(biasTensorShape[0] % 3 == 0);
            ML_CHECK_VALID_ARGUMENT(weightTensorShape[1] == biasTensorShape[0]);
        }

        if (hasPast)
        {
            ML_CHECK_VALID_ARGUMENT(kernelCreationContext.IsOutputValid(presentIndex));
        }

        const uint32_t hiddenSize = weightTensorShape[1] / 3;
        const uint32_t headSize = hiddenSize / numHeads;
        const uint32_t batchSize = inputTensorShape[0];
        const uint32_t sequenceLength = inputTensorShape[1];
        const uint32_t pastSequenceLength = hasPast ? m_inputTensorDescs[pastIndex].GetSizes()[3] : 0;

        uint32_t desiredWeightTensorShape[3] = {batchSize, weightTensorShape[0], 3 * hiddenSize};
        MLOperatorTensorDataType dataType = kernelCreationContext.GetOutputEdgeDescription(outputIndex).tensorDataType;

        m_inputTensorDescs[weightsIndex] = TensorDesc::ConstructBroadcastedTensorDesc(
            kernelCreationContext.GetInputEdgeDescription(weightsIndex).tensorDataType,
            desiredWeightTensorShape,
            weightTensorShape);

        uint32_t desiredBiasTensorShape[3] = {batchSize, sequenceLength, 3 * hiddenSize};

        if (hasBias)
        {
            auto biasTensorShape = m_inputTensorDescs[biasIndex].GetSizes();
            m_inputTensorDescs[biasIndex] = TensorDesc::ConstructBroadcastedTensorDesc(kernelCreationContext.GetInputEdgeDescription(biasIndex).tensorDataType, desiredBiasTensorShape, biasTensorShape);
        }

        MLOperatorTensorDataType maskTensorDataType = MLOperatorTensorDataType::Undefined;
        bool hasMaxSequenceMask = false;
        DML_MULTIHEAD_ATTENTION_MASK_TYPE maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_NONE;
        if (hasMask)
        {
            if (hasUnpaddedBounds)
            {
                auto unpaddedKeyBoundsShape = m_inputTensorDescs[maskIndex].GetSizes();
                ML_CHECK_VALID_ARGUMENT(unpaddedKeyBoundsShape.size() == 1);

                const uint32_t batchGroupCount = unpaddedKeyBoundsShape[0] / batchSize;
                ML_CHECK_VALID_ARGUMENT(batchGroupCount == 1 || batchGroupCount == 2);

                uint32_t desiredShape[2] = {batchGroupCount, batchSize};
                m_inputTensorDescs[maskIndex] = TensorDesc(
                    m_inputTensorDescs[maskIndex].GetDmlDataType(),
                    desiredShape);

                maskType = batchGroupCount == 1
                    ? DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH
                    : DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_END_START;
            }
            else
            {
                auto maskIndexTensorShape = m_inputTensorDescs[maskIndex].GetSizes();
                ML_CHECK_VALID_ARGUMENT(maskIndexTensorShape.size() > 1 && maskIndexTensorShape.size() <= 4);

                maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_BOOLEAN;
                std::vector<uint32_t> reshapedMaskIndexTensorShape(maskIndexTensorShape.begin(), maskIndexTensorShape.end());
                if (maskIndexTensorShape.size() == 4 && maskIndexTensorShape[2] != sequenceLength)
                {
                    hasMaxSequenceMask = true;
                    ML_CHECK_VALID_ARGUMENT(maskIndexTensorShape[2] == maskIndexTensorShape[3]);
                    const uint32_t maxSequenceLength = maskIndexTensorShape[2];
                    uint32_t desiredMaskIndexShape[4] = {batchSize, numHeads, maxSequenceLength, maxSequenceLength};
                    maskTensorDataType = kernelCreationContext.GetInputEdgeDescription(maskIndex).tensorDataType;
                    m_inputTensorDescs[maskIndex] = TensorDesc::ConstructBroadcastedTensorDesc(maskTensorDataType, desiredMaskIndexShape, reshapedMaskIndexTensorShape);
                }
                else
                {
                    uint32_t maskIndexDimensionCount = gsl::narrow_cast<uint32_t>(maskIndexTensorShape.size());
                    reshapedMaskIndexTensorShape.insert(reshapedMaskIndexTensorShape.begin() + 1, 4 - maskIndexDimensionCount, 1);
                    uint32_t desiredMaskIndexShape[4] = {batchSize, numHeads, sequenceLength, sequenceLength};
                    maskTensorDataType = kernelCreationContext.GetInputEdgeDescription(maskIndex).tensorDataType;
                    m_inputTensorDescs[maskIndex] = TensorDesc::ConstructBroadcastedTensorDesc(maskTensorDataType, desiredMaskIndexShape, reshapedMaskIndexTensorShape);
                }
            }
        }

        MLOperatorTensorDataType pastTensorDataType = MLOperatorTensorDataType::Undefined;
        MLOperatorTensorDataType presentTensorDataType = MLOperatorTensorDataType::Undefined;
        if (hasPast)
        {
            pastTensorDataType = kernelCreationContext.GetInputEdgeDescription(pastIndex).tensorDataType;
            presentTensorDataType = kernelCreationContext.GetOutputEdgeDescription(presentIndex).tensorDataType;
        }

        TensorDesc matMulIntToFloatOutputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, desiredBiasTensorShape);
        DML_TENSOR_DESC namedMatMulIntToFloatOutputTensorDesc = matMulIntToFloatOutputTensorDesc.GetDmlDesc();

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_MATRIX_MULTIPLY_INTEGER_TO_FLOAT_OPERATOR_DESC matMulIntToFloatOperatorDesc = {};
        matMulIntToFloatOperatorDesc.ATensor = &inputDescs[InputIndex::inputIndex];
        matMulIntToFloatOperatorDesc.AScaleTensor = &inputDescs[InputIndex::inputScaleIndex];
        matMulIntToFloatOperatorDesc.AZeroPointTensor = &inputDescs[InputIndex::inputZeroPointIndex];
        matMulIntToFloatOperatorDesc.BTensor = &inputDescs[InputIndex::weightsIndex];
        matMulIntToFloatOperatorDesc.BScaleTensor = &inputDescs[InputIndex::weightScaleIndex];
        matMulIntToFloatOperatorDesc.BZeroPointTensor = &inputDescs[InputIndex::weightZeroPointIndex];
        matMulIntToFloatOperatorDesc.BiasTensor = hasBias ? &inputDescs[InputIndex::biasIndex] : nullptr;
        matMulIntToFloatOperatorDesc.OutputTensor = &namedMatMulIntToFloatOutputTensorDesc;

        const DML_OPERATOR_DESC matMulIntToFloatDesc = { DML_OPERATOR_MATRIX_MULTIPLY_INTEGER_TO_FLOAT, &matMulIntToFloatOperatorDesc};

        std::array<uint32_t, 3> queryKeySlicedTensorShape = {batchSize, sequenceLength, hiddenSize + hiddenSize};
        TensorDesc queryKeySlicedInputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, queryKeySlicedTensorShape);
        DML_TENSOR_DESC namedQueryKeySlicedInputTensorDesc = queryKeySlicedInputTensorDesc.GetDmlDesc();

        std::array<uint32_t, 3> valueSlicedTensorShape = {batchSize, sequenceLength, hiddenSize};
        TensorDesc valueSlicedInputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, valueSlicedTensorShape);
        DML_TENSOR_DESC namedValueSlicedInputTensorDesc = valueSlicedInputTensorDesc.GetDmlDesc();

        // Transpose slice QK from [batchSize, sequenceLength, 2, numHeads, headSize] to [batchSize, sequenceLength, numHeads, 2, headSize]
        std::array<uint32_t, 5> queryKeyTransposedTensorShape = {batchSize, sequenceLength, numHeads, 2, headSize};
        std::array<uint32_t, 5> queryKeyTransposedStrides = {
            sequenceLength * numHeads * 2 * headSize,
            numHeads * 2 * headSize,
            headSize,
            numHeads * headSize,
            1,
        };

        TensorDesc queryKeyTransposedInputTensorDesc = TensorDesc(
            GetDmlDataTypeFromMlDataType(dataType),
            queryKeyTransposedTensorShape,
            queryKeyTransposedStrides);
        DML_TENSOR_DESC namedQueryKeyTransposedInputTensorDesc = queryKeyTransposedInputTensorDesc.GetDmlDesc();

        TensorDesc queryKeyTransposedOutputTensorDesc = TensorDesc(
            GetDmlDataTypeFromMlDataType(dataType),
            queryKeyTransposedTensorShape);
        DML_TENSOR_DESC namedQueryKeyTransposedOutputTensorDesc = queryKeyTransposedOutputTensorDesc.GetDmlDesc();

        // Transpose QKV from [batchSize, sequenceLength, 3, numHeads, headSize] to [batchSize, sequenceLength, numHeads, 3, headSize]
        std::array<uint32_t, 5> queryKeyValueTransposedTensorShape = {batchSize, sequenceLength, numHeads, 3, headSize};
        std::array<uint32_t, 5> queryKeyValueTransposedStrides = {
            sequenceLength * numHeads * 3 * headSize,
            numHeads * 3 * headSize,
            headSize,
            numHeads * headSize,
            1,
        };

        TensorDesc queryKeyValueTransposedInputTensorDesc = TensorDesc(
            GetDmlDataTypeFromMlDataType(dataType),
            queryKeyValueTransposedTensorShape,
            queryKeyValueTransposedStrides);
        DML_TENSOR_DESC namedQueryKeyValueTransposedInputTensorDesc = queryKeyValueTransposedInputTensorDesc.GetDmlDesc();

        TensorDesc queryKeyValueTransposedOutputTensorDesc = TensorDesc(
            GetDmlDataTypeFromMlDataType(dataType),
            queryKeyValueTransposedTensorShape);
        DML_TENSOR_DESC namedQueryKeyValueTransposedOutputTensorDesc = queryKeyValueTransposedOutputTensorDesc.GetDmlDesc();

        std::array<uint32_t, 3> queryKeySliceOffset = {0, 0, 0};
        std::array<uint32_t, 3> queryKeySliceSize = {batchSize, sequenceLength, hiddenSize + hiddenSize};
        std::array<int32_t, 3> queryKeySliceStrides = {1, 1, 1};

        std::array<uint32_t, 3> valueSliceOffset = {0, 0, 2 * hiddenSize};
        std::array<uint32_t, 3> valueSliceSize = {batchSize, sequenceLength, hiddenSize};
        std::array<int32_t, 3> valueSliceStrides = {1, 1, 1};

        // When Q/K/V all have the same hidden size, we just have to transpose it before sending it to MHA
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC transposeOperatorDesc = {};

        transposeOperatorDesc.InputTensor = &namedQueryKeyValueTransposedInputTensorDesc;
        transposeOperatorDesc.OutputTensor = &namedQueryKeyValueTransposedOutputTensorDesc;

        const DML_OPERATOR_DESC transposedDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &transposeOperatorDesc};

        std::array<uint32_t, 4> maskSliceOutputShape = {batchSize, numHeads, sequenceLength, sequenceLength};
        std::array<int32_t, 4> maskSliceStrides = {1, 1, 1, 1};
        std::array<uint32_t, 4> maskSliceOffsets = {0, 0, 0, 0};
        TensorDesc maskSliceOutputTensorDesc;
        DML_TENSOR_DESC namedMaskSliceOutputTensorDesc;

        DML_SLICE1_OPERATOR_DESC maskSlicedOperatorDesc = {};
        if (hasMaxSequenceMask)
        {
            maskSliceOutputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(maskTensorDataType, maskSliceOutputShape);
            namedMaskSliceOutputTensorDesc = maskSliceOutputTensorDesc.GetDmlDesc();
            maskSlicedOperatorDesc.InputTensor = &inputDescs[maskIndex];
            maskSlicedOperatorDesc.OutputTensor = &namedMaskSliceOutputTensorDesc;
            maskSlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(maskSliceOutputShape.size());
            maskSlicedOperatorDesc.InputWindowOffsets = maskSliceOffsets.data();
            maskSlicedOperatorDesc.InputWindowSizes = maskSliceOutputShape.data();
            maskSlicedOperatorDesc.InputWindowStrides = maskSliceStrides.data();
        }
        const DML_OPERATOR_DESC maskSlicedDesc = { DML_OPERATOR_SLICE1, &maskSlicedOperatorDesc};

        // We need to slice Past to get PastValue and PastKey tensors for MHA
        std::array<uint32_t, 5> pastKeyOutputShape = {1, batchSize, numHeads, pastSequenceLength, headSize};
        std::array<int32_t, 5> pastKeyStrides = {1, 1, 1, 1, 1};
        std::array<uint32_t, 5> pastKeyOffsets = {0, 0, 0, 0, 0};
        TensorDesc pastKeyOutputTensorDesc;
        DML_TENSOR_DESC namedPastKeyOutputTensorDesc;
        
        std::array<uint32_t, 5> pastValueOutputShape = {1, batchSize, numHeads, pastSequenceLength, headSize};
        std::array<int32_t, 5> pastValueStrides = {1, 1, 1, 1, 1};
        std::array<uint32_t, 5> pastValueOffsets = {1, 0, 0, 0, 0};
        TensorDesc pastValueOutputTensorDesc;
        DML_TENSOR_DESC namedPastValueOutputTensorDesc;

        DML_SLICE1_OPERATOR_DESC pastKeySlicedOperatorDesc = {};
        DML_SLICE1_OPERATOR_DESC pastValueSlicedOperatorDesc = {};

        if (hasPast)
        {
            pastKeyOutputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(pastTensorDataType, pastKeyOutputShape);
            namedPastKeyOutputTensorDesc = pastKeyOutputTensorDesc.GetDmlDesc();
            pastKeySlicedOperatorDesc.InputTensor = &inputDescs[pastIndex];
            pastKeySlicedOperatorDesc.OutputTensor = &namedPastKeyOutputTensorDesc;
            pastKeySlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(pastKeyOutputShape.size());
            pastKeySlicedOperatorDesc.InputWindowOffsets = pastKeyOffsets.data();
            pastKeySlicedOperatorDesc.InputWindowSizes = pastKeyOutputShape.data();
            pastKeySlicedOperatorDesc.InputWindowStrides = pastKeyStrides.data();

            pastValueOutputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(pastTensorDataType, pastValueOutputShape);
            namedPastValueOutputTensorDesc = pastValueOutputTensorDesc.GetDmlDesc();
            pastValueSlicedOperatorDesc.InputTensor = &inputDescs[pastIndex];
            pastValueSlicedOperatorDesc.OutputTensor = &namedPastValueOutputTensorDesc;
            pastValueSlicedOperatorDesc.DimensionCount = gsl::narrow_cast<uint32_t>(pastValueOutputShape.size());
            pastValueSlicedOperatorDesc.InputWindowOffsets = pastValueOffsets.data();
            pastValueSlicedOperatorDesc.InputWindowSizes = pastValueOutputShape.data();
            pastValueSlicedOperatorDesc.InputWindowStrides = pastValueStrides.data();
        }

        const DML_OPERATOR_DESC pastKeySlicedDesc = { DML_OPERATOR_SLICE1, &pastKeySlicedOperatorDesc};
        const DML_OPERATOR_DESC pastValueSlicedDesc = { DML_OPERATOR_SLICE1, &pastValueSlicedOperatorDesc};

        // Causal Mask: [pastSequenceLength, pastSequenceLength + 1 ... pastSequenceLength + batchSize -1]
        // passed to MHA as maskIndex Tensor when unidirectional == 1
        std::array<uint32_t, 2> causalMaskOutputShape = {1, batchSize};
        TensorDesc causalMaskTensorDesc;
        DML_FILL_VALUE_SEQUENCE_OPERATOR_DESC causalMaskOperatorDesc = {};
        DML_TENSOR_DESC namedcausalMaskTensorDesc;

        if (unidirectional && !hasMask)
        {
            causalMaskTensorDesc = TensorDesc::ConstructDefaultTensorDesc(MLOperatorTensorDataType::Int32, causalMaskOutputShape);
            namedcausalMaskTensorDesc = causalMaskTensorDesc.GetDmlDesc();
            causalMaskOperatorDesc.ValueDataType = DML_TENSOR_DATA_TYPE_INT32;
            causalMaskOperatorDesc.ValueStart.Int32 = pastSequenceLength;
            causalMaskOperatorDesc.ValueDelta.Int32 = 1;
            causalMaskOperatorDesc.OutputTensor = &namedcausalMaskTensorDesc;

            maskType = DML_MULTIHEAD_ATTENTION_MASK_TYPE_KEY_SEQUENCE_LENGTH;
        }
        DML_OPERATOR_DESC causalMaskDesc = { DML_OPERATOR_FILL_VALUE_SEQUENCE, &causalMaskOperatorDesc };

        DML_MULTIHEAD_ATTENTION_OPERATOR_DESC mhaOperatorDesc = {};
        std::array<uint32_t, 5> presentKeyOutputShape = {1, batchSize, numHeads, pastSequenceLength + sequenceLength, headSize};
        std::array<uint32_t, 5> presentValueOutputShape = {1, batchSize, numHeads, pastSequenceLength + sequenceLength, headSize};
        TensorDesc presentKeyTensorDesc;
        TensorDesc presentValueTensorDesc;
        DML_TENSOR_DESC namedPresentKeyOutputTensorDesc;
        DML_TENSOR_DESC namedPresentValueOutputTensorDesc;

        mhaOperatorDesc.StackedQueryKeyValueTensor = &namedQueryKeyValueTransposedOutputTensorDesc;

        if (unidirectional && !hasMask)
        {
            mhaOperatorDesc.MaskTensor = &namedcausalMaskTensorDesc;
        }
        else if (hasMaxSequenceMask)
        {
            mhaOperatorDesc.MaskTensor = &namedMaskSliceOutputTensorDesc;
        }
        else
        {
            mhaOperatorDesc.MaskTensor = hasMask ? &inputDescs[maskIndex] : nullptr;
        }

        mhaOperatorDesc.RelativePositionBiasTensor = nullptr;
        mhaOperatorDesc.OutputTensor = &outputDescs[outputIndex];
        mhaOperatorDesc.Scale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Scale, gsl::narrow_cast<float>(1.0f / std::sqrt(headSize)));
        mhaOperatorDesc.MaskFilterValue = kernelCreationContext.GetOptionalAttribute<float>(AttrName::MaskFilterValue, -10'000.0f);
        mhaOperatorDesc.HeadCount = numHeads;
        mhaOperatorDesc.MaskType = maskType;
        if (hasPast)
        {
            presentKeyTensorDesc = TensorDesc::ConstructDefaultTensorDesc(presentTensorDataType, presentKeyOutputShape);
            namedPresentKeyOutputTensorDesc = presentKeyTensorDesc.GetDmlDesc();
            presentValueTensorDesc = TensorDesc::ConstructDefaultTensorDesc(presentTensorDataType, presentValueOutputShape);
            namedPresentValueOutputTensorDesc = presentValueTensorDesc.GetDmlDesc();
            mhaOperatorDesc.PastKeyTensor = &namedPastKeyOutputTensorDesc;
            mhaOperatorDesc.PastValueTensor = &namedPastValueOutputTensorDesc;
            mhaOperatorDesc.OutputPresentKeyTensor = &namedPresentKeyOutputTensorDesc;
            mhaOperatorDesc.OutputPresentValueTensor = &namedPresentValueOutputTensorDesc;
        }

        const DML_OPERATOR_DESC mhaDesc = { DML_OPERATOR_MULTIHEAD_ATTENTION, &mhaOperatorDesc };

        DML_JOIN_OPERATOR_DESC presentKeyValueJoinOperatorDesc = {};
        std::vector<DML_TENSOR_DESC> joinInputDesc;

        if (hasPast)
        {
            joinInputDesc.push_back(namedPresentKeyOutputTensorDesc);
            joinInputDesc.push_back(namedPresentValueOutputTensorDesc);
            presentKeyValueJoinOperatorDesc.InputCount = gsl::narrow_cast<uint32_t>(joinInputDesc.size());
            presentKeyValueJoinOperatorDesc.InputTensors = joinInputDesc.data();
            presentKeyValueJoinOperatorDesc.OutputTensor = &outputDescs[presentIndex];
            presentKeyValueJoinOperatorDesc.Axis = gsl::narrow_cast<uint32_t>(0);
        }

        DML_OPERATOR_DESC presentKeyValueJoinDesc = { DML_OPERATOR_JOIN, &presentKeyValueJoinOperatorDesc };

        // Construct the graph
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

        std::vector<const DML_OPERATOR_DESC*> opDescs = {
            &matMulIntToFloatDesc,
            &mhaDesc,
        };

        uint32_t currentNodeIndex = 0;
        const uint32_t matMulIntToFloatNodeIndex = currentNodeIndex++;
        const uint32_t mhaNodeIndex = currentNodeIndex++;

        uint32_t queryKeyValueTransposedNodeIndex = 0;

        opDescs.push_back(&transposedDesc);
        queryKeyValueTransposedNodeIndex = currentNodeIndex++;

        uint32_t maskSliceNodeIndex = 0;
        if (hasMaxSequenceMask)
        {
            opDescs.push_back(&maskSlicedDesc);
            maskSliceNodeIndex = currentNodeIndex++;
        }

        uint32_t pastKeySliceNodeIndex = 0;
        uint32_t pastValueSliceNodeIndex = 0;
        uint32_t concatNodeIndex = 0;
        if (hasPast)
        {
            opDescs.push_back(&pastKeySlicedDesc);
            pastKeySliceNodeIndex = currentNodeIndex++;
            opDescs.push_back(&pastValueSlicedDesc);
            pastValueSliceNodeIndex = currentNodeIndex++;
            opDescs.push_back(&presentKeyValueJoinDesc);
            concatNodeIndex = currentNodeIndex++;
        }

        uint32_t causalMaskNodeIndex = 0;
        if (unidirectional && !hasMask)
        {
            opDescs.push_back(&causalMaskDesc);
            causalMaskNodeIndex = currentNodeIndex++;
        }
        
        DML_INPUT_GRAPH_EDGE_DESC inputToMatMulIntToFloatEdge = {};
        inputToMatMulIntToFloatEdge.GraphInputIndex = InputIndex::inputIndex;
        inputToMatMulIntToFloatEdge.ToNodeIndex = matMulIntToFloatNodeIndex;
        inputToMatMulIntToFloatEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(inputToMatMulIntToFloatEdge);

        DML_INPUT_GRAPH_EDGE_DESC inputScaleToMatMulIntToFloatEdge = {};
        inputScaleToMatMulIntToFloatEdge.GraphInputIndex = InputIndex::inputScaleIndex;
        inputScaleToMatMulIntToFloatEdge.ToNodeIndex = matMulIntToFloatNodeIndex;
        inputScaleToMatMulIntToFloatEdge.ToNodeInputIndex = 1;
        inputEdges.push_back(inputScaleToMatMulIntToFloatEdge);

        DML_INPUT_GRAPH_EDGE_DESC inputZeroPointToMatMulIntToFloatEdge = {};
        inputZeroPointToMatMulIntToFloatEdge.GraphInputIndex = InputIndex::inputZeroPointIndex;
        inputZeroPointToMatMulIntToFloatEdge.ToNodeIndex = matMulIntToFloatNodeIndex;
        inputZeroPointToMatMulIntToFloatEdge.ToNodeInputIndex = 2;
        inputEdges.push_back(inputZeroPointToMatMulIntToFloatEdge);

        DML_INPUT_GRAPH_EDGE_DESC weightToMatMulIntToFloatEdge = {};
        weightToMatMulIntToFloatEdge.GraphInputIndex = InputIndex::weightsIndex;
        weightToMatMulIntToFloatEdge.ToNodeIndex = matMulIntToFloatNodeIndex;
        weightToMatMulIntToFloatEdge.ToNodeInputIndex = 3;
        inputEdges.push_back(weightToMatMulIntToFloatEdge);

        DML_INPUT_GRAPH_EDGE_DESC weightScaleToMatMulIntToFloatEdge = {};
        weightScaleToMatMulIntToFloatEdge.GraphInputIndex = InputIndex::weightScaleIndex;
        weightScaleToMatMulIntToFloatEdge.ToNodeIndex = matMulIntToFloatNodeIndex;
        weightScaleToMatMulIntToFloatEdge.ToNodeInputIndex = 4;
        inputEdges.push_back(weightScaleToMatMulIntToFloatEdge);

        DML_INPUT_GRAPH_EDGE_DESC weightZeroPointToMatMulIntToFloatEdge = {};
        weightZeroPointToMatMulIntToFloatEdge.GraphInputIndex = InputIndex::weightZeroPointIndex;
        weightZeroPointToMatMulIntToFloatEdge.ToNodeIndex = matMulIntToFloatNodeIndex;
        weightZeroPointToMatMulIntToFloatEdge.ToNodeInputIndex = 5;
        inputEdges.push_back(weightZeroPointToMatMulIntToFloatEdge);

        if (hasBias)
        {
            DML_INPUT_GRAPH_EDGE_DESC biasToMatMulIntToFloatEdge = {};
            biasToMatMulIntToFloatEdge.GraphInputIndex = InputIndex::biasIndex;
            biasToMatMulIntToFloatEdge.ToNodeIndex = matMulIntToFloatNodeIndex;
            biasToMatMulIntToFloatEdge.ToNodeInputIndex = 6;
            inputEdges.push_back(biasToMatMulIntToFloatEdge);
        }

        if (hasMask)
        {
            if (hasUnpaddedBounds)
            {
                DML_INPUT_GRAPH_EDGE_DESC maskToMhaEdge = {};
                maskToMhaEdge.GraphInputIndex = InputIndex::maskIndex;
                maskToMhaEdge.ToNodeIndex = mhaNodeIndex;
                maskToMhaEdge.ToNodeInputIndex = mhaMaskIndex;
                inputEdges.push_back(maskToMhaEdge);
            }
            else if (hasMaxSequenceMask)
            {
                DML_INPUT_GRAPH_EDGE_DESC maskToMaskSliceEdge = {};
                maskToMaskSliceEdge.GraphInputIndex = InputIndex::maskIndex;
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
                maskToMhaEdge.GraphInputIndex = InputIndex::maskIndex;
                maskToMhaEdge.ToNodeIndex = mhaNodeIndex;
                maskToMhaEdge.ToNodeInputIndex = mhaMaskIndex;
                inputEdges.push_back(maskToMhaEdge);
            }
        }
        else if (unidirectional)
        {
            DML_INTERMEDIATE_GRAPH_EDGE_DESC causalMaskToMhaEdge = {};
            causalMaskToMhaEdge.FromNodeIndex = causalMaskNodeIndex;
            causalMaskToMhaEdge.FromNodeOutputIndex = 0;
            causalMaskToMhaEdge.ToNodeIndex = mhaNodeIndex;
            causalMaskToMhaEdge.ToNodeInputIndex = mhaMaskIndex ;
            intermediateEdges.push_back(causalMaskToMhaEdge);
        }

        if (hasPast)
        {
            DML_INPUT_GRAPH_EDGE_DESC pastToPastKeySliceEdge = {};
            pastToPastKeySliceEdge.GraphInputIndex = InputIndex::pastIndex;
            pastToPastKeySliceEdge.ToNodeIndex = pastKeySliceNodeIndex;
            pastToPastKeySliceEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(pastToPastKeySliceEdge);

            DML_INPUT_GRAPH_EDGE_DESC pastToPastValueSliceEdge = {};
            pastToPastValueSliceEdge.GraphInputIndex = InputIndex::pastIndex;
            pastToPastValueSliceEdge.ToNodeIndex = pastValueSliceNodeIndex;
            pastToPastValueSliceEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(pastToPastValueSliceEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC pastKeyToMhaEdge = {};
            pastKeyToMhaEdge.FromNodeIndex = pastKeySliceNodeIndex;
            pastKeyToMhaEdge.FromNodeOutputIndex = 0;
            pastKeyToMhaEdge.ToNodeIndex = mhaNodeIndex;
            pastKeyToMhaEdge.ToNodeInputIndex = mhaPastKeyIndex;
            intermediateEdges.push_back(pastKeyToMhaEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC pastValueToMhaEdge = {};
            pastValueToMhaEdge.FromNodeIndex = pastValueSliceNodeIndex;
            pastValueToMhaEdge.FromNodeOutputIndex = 0;
            pastValueToMhaEdge.ToNodeIndex = mhaNodeIndex;
            pastValueToMhaEdge.ToNodeInputIndex = mhaPastValueIndex;
            intermediateEdges.push_back(pastValueToMhaEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC presentKeyToConcatEdge = {};
            presentKeyToConcatEdge.FromNodeIndex = mhaNodeIndex;
            presentKeyToConcatEdge.FromNodeOutputIndex = mhaPresentKeyIndex;
            presentKeyToConcatEdge.ToNodeIndex = concatNodeIndex;
            presentKeyToConcatEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(presentKeyToConcatEdge);

            DML_INTERMEDIATE_GRAPH_EDGE_DESC presentValueToConcatEdge = {};
            presentValueToConcatEdge.FromNodeIndex = mhaNodeIndex;
            presentValueToConcatEdge.FromNodeOutputIndex = mhaPresentValueIndex;
            presentValueToConcatEdge.ToNodeIndex = concatNodeIndex;
            presentValueToConcatEdge.ToNodeInputIndex = 1;
            intermediateEdges.push_back(presentValueToConcatEdge);
        }

        DML_INTERMEDIATE_GRAPH_EDGE_DESC matMulIntToFloatToQueryKeyValueTransposeEdge = {};
        matMulIntToFloatToQueryKeyValueTransposeEdge.FromNodeIndex = matMulIntToFloatNodeIndex;
        matMulIntToFloatToQueryKeyValueTransposeEdge.FromNodeOutputIndex = 0;
        matMulIntToFloatToQueryKeyValueTransposeEdge.ToNodeIndex = queryKeyValueTransposedNodeIndex;
        matMulIntToFloatToQueryKeyValueTransposeEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(matMulIntToFloatToQueryKeyValueTransposeEdge);

        // All we need to do here is transpose the stacked QKV tensor into something DML supports
        DML_INTERMEDIATE_GRAPH_EDGE_DESC queryKeyValueTransposedToMhaEdge = {};
        queryKeyValueTransposedToMhaEdge.FromNodeIndex = queryKeyValueTransposedNodeIndex;
        queryKeyValueTransposedToMhaEdge.FromNodeOutputIndex = 0;
        queryKeyValueTransposedToMhaEdge.ToNodeIndex = mhaNodeIndex;
        queryKeyValueTransposedToMhaEdge.ToNodeInputIndex = mhaStackedQueryKeyValueIndex;
        intermediateEdges.push_back(queryKeyValueTransposedToMhaEdge);

        DML_OUTPUT_GRAPH_EDGE_DESC mhaToOutputEdge = {};
        mhaToOutputEdge.FromNodeIndex = mhaNodeIndex;
        mhaToOutputEdge.FromNodeOutputIndex = mhaOutputIndex;
        mhaToOutputEdge.GraphOutputIndex = OutputIndex::outputIndex;
        outputEdges.push_back(mhaToOutputEdge);

        if (hasPast)
        {
            DML_OUTPUT_GRAPH_EDGE_DESC concatToOutputEdge = {};
            concatToOutputEdge.FromNodeIndex = concatNodeIndex;
            concatToOutputEdge.FromNodeOutputIndex = 0;
            concatToOutputEdge.GraphOutputIndex = OutputIndex::presentIndex;
            outputEdges.push_back(concatToOutputEdge);
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

void CALLBACK QueryQAttention(IMLOperatorSupportQueryContextPrivate* context, /*out*/ bool* isSupported)
{
    *isSupported = false;

    // `unidirectional == 1` with Mask Tensor is not supported yet
    MLOperatorAttributes attributes(context);
    if (attributes.GetOptionalAttribute<int32_t>(AttrName::Unidirectional, 0) != 0 && context->IsInputValid(5))
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

DML_OP_DEFINE_CREATION_FUNCTION(QAttention, DmlOperatorQAttention);
} // namespace Dml

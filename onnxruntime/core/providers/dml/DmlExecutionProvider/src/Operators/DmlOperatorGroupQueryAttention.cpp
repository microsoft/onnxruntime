// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{
class DmlOperatorGroupQueryAttention : public DmlOperator, public GroupQueryAttentionHelper
{
public:
    DmlOperatorGroupQueryAttention(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext),
        GroupQueryAttentionHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription())
    {
        enum InputIndex : uint32_t
        {
            queryIndex,
            keyIndex,
            valueIndex,
            pastKeyIndex,
            pastValueIndex,
            seqLensIndex,
            inputCount,
        };

        enum OutputIndex : uint32_t
        {
            outputIndex,
            outputPresentKeyIndex,
            outputPresentValueIndex,
            outputCount,
        };

        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() >= 1);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() >= 1);

        std::vector<std::optional<uint32_t>> inputIndices(inputCount);
        inputIndices[queryIndex] = queryIndex;
        inputIndices[keyIndex] = keyIndex;
        inputIndices[valueIndex] = valueIndex;

        const uint32_t sequenceLength = kernelCreationContext.GetInputTensorShape(queryIndex)[1];

        if (kernelCreationContext.GetInputTensorShape(queryIndex)[1] == 1)
        {
            inputIndices[seqLensIndex] = seqLensIndex;
        }

        std::vector<std::optional<uint32_t>> outputIndices = {
            outputIndex,
            outputPresentKeyIndex,
            outputPresentValueIndex,
        };
        DmlOperator::Initialize(kernelCreationContext, inputIndices, outputIndices, std::nullopt, std::nullopt, 1);

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[queryIndex].GetDimensionCount() == 3);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[keyIndex].GetDimensionCount() == 3);
        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[valueIndex].GetDimensionCount() == 3);

        const uint32_t queryNumHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::NumHeads));
        const uint32_t kvNumHeads = gsl::narrow_cast<uint32_t>(kernelCreationContext.GetAttribute<int64_t>(AttrName::KvNumHeads));

        auto querySizes = m_inputTensorDescs[queryIndex].GetSizes();
        auto keySizes = m_inputTensorDescs[keyIndex].GetSizes();
        auto valueSizes = m_inputTensorDescs[valueIndex].GetSizes();

        const uint32_t batchSize = querySizes[0];
        const uint32_t queryHiddenSize = querySizes[2];

        const uint32_t kvSequenceLength = keySizes[1];
        const uint32_t kvHiddenSize = keySizes[2];

        const uint32_t queryHeadSize = queryHiddenSize / queryNumHeads;
        const uint32_t kvHeadSize = kvHiddenSize / kvNumHeads;
        const uint32_t totalSequenceLength = GetTotalSequenceLength();

        // Validate Query dimensions
        ML_CHECK_VALID_ARGUMENT(querySizes[0] == batchSize);
        ML_CHECK_VALID_ARGUMENT(querySizes[1] == sequenceLength);
        ML_CHECK_VALID_ARGUMENT(querySizes[2] == queryHiddenSize);

        // Validate Key dimensions
        ML_CHECK_VALID_ARGUMENT(keySizes[0] == batchSize);
        ML_CHECK_VALID_ARGUMENT(keySizes[1] == kvSequenceLength);
        ML_CHECK_VALID_ARGUMENT(keySizes[2] == kvHiddenSize);

        // Validate Value dimensions
        ML_CHECK_VALID_ARGUMENT(valueSizes[0] == batchSize);
        ML_CHECK_VALID_ARGUMENT(valueSizes[1] == kvSequenceLength);
        ML_CHECK_VALID_ARGUMENT(valueSizes[2] == kvHiddenSize);

        if (sequenceLength == 1)
        {
            // Validate PastSequenceLengths dimensions
            if (m_inputTensorDescs[seqLensIndex].GetDimensionCount() == 1)
            {
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[seqLensIndex].GetSizes()[0] == batchSize);
            }
            else
            {
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[seqLensIndex].GetDimensionCount() == 2);
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[seqLensIndex].GetSizes()[0] == batchSize);
                ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs[seqLensIndex].GetSizes()[1] == 1);
            }
        }

        const std::array<uint32_t, 1> pastSequenceLengthsShape = {batchSize};
        auto pastSequenceLengthsDataType = MLOperatorTensorDataType::Int32;
        TensorDesc pastSequenceLengthsTensorDesc = TensorDesc::ConstructDefaultTensorDesc(pastSequenceLengthsDataType, pastSequenceLengthsShape);
        const DML_TENSOR_DESC pastSequenceLengthsDmlTensorDesc = pastSequenceLengthsTensorDesc.GetDmlDesc();

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        // GQA is very sensitive to overflows, so we cast all inputs to fp32 and cast the outputs back to fp16. At the DML level,
        // those casts will be eliminated and replaced with half precision computation instead, which mimics the CUDA EP behavior
        // of their flash attention kernel.
        TensorDesc queryCastTensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_inputTensorDescs[queryIndex].GetSizes());
        DML_TENSOR_DESC queryCastDmlTensorDesc = queryCastTensorDesc.GetDmlDesc();
        DML_CAST_OPERATOR_DESC queryCastOpDesc{};
        queryCastOpDesc.InputTensor = &inputDescs[queryIndex];
        queryCastOpDesc.OutputTensor = &queryCastDmlTensorDesc;
        DML_OPERATOR_DESC queryCastDmlDesc = { DML_OPERATOR_CAST, &queryCastOpDesc };

        TensorDesc keyCastTensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_inputTensorDescs[keyIndex].GetSizes());
        DML_TENSOR_DESC keyCastDmlTensorDesc = keyCastTensorDesc.GetDmlDesc();
        DML_CAST_OPERATOR_DESC keyCastOpDesc{};
        keyCastOpDesc.InputTensor = &inputDescs[keyIndex];
        keyCastOpDesc.OutputTensor = &keyCastDmlTensorDesc;
        DML_OPERATOR_DESC keyCastDmlDesc = { DML_OPERATOR_CAST, &keyCastOpDesc };

        TensorDesc valueCastTensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_inputTensorDescs[valueIndex].GetSizes());
        DML_TENSOR_DESC valueCastDmlTensorDesc = valueCastTensorDesc.GetDmlDesc();
        DML_CAST_OPERATOR_DESC valueCastOpDesc{};
        valueCastOpDesc.InputTensor = &inputDescs[valueIndex];
        valueCastOpDesc.OutputTensor = &valueCastDmlTensorDesc;
        DML_OPERATOR_DESC valueCastDmlDesc = { DML_OPERATOR_CAST, &valueCastOpDesc };

        TensorDesc outputCastTensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_outputTensorDescs[outputIndex].GetSizes());
        DML_TENSOR_DESC outputCastDmlTensorDesc = outputCastTensorDesc.GetDmlDesc();
        DML_CAST_OPERATOR_DESC outputCastOpDesc{};
        outputCastOpDesc.InputTensor = &outputCastDmlTensorDesc;
        outputCastOpDesc.OutputTensor = &outputDescs[outputIndex];
        DML_OPERATOR_DESC outputCastDmlDesc = { DML_OPERATOR_CAST, &outputCastOpDesc };

        TensorDesc outputPresentKeyCastTensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_outputTensorDescs[outputPresentKeyIndex].GetSizes());
        DML_TENSOR_DESC outputPresentKeyCastDmlTensorDesc = outputPresentKeyCastTensorDesc.GetDmlDesc();
        DML_CAST_OPERATOR_DESC outputPresentKeyCastOpDesc{};
        outputPresentKeyCastOpDesc.InputTensor = &outputPresentKeyCastDmlTensorDesc;
        outputPresentKeyCastOpDesc.OutputTensor = &outputDescs[outputPresentKeyIndex];
        DML_OPERATOR_DESC outputPresentKeyCastDmlDesc = { DML_OPERATOR_CAST, &outputPresentKeyCastOpDesc };

        TensorDesc outputPresentValueCastTensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, m_outputTensorDescs[outputPresentValueIndex].GetSizes());
        DML_TENSOR_DESC outputPresentValueCastDmlTensorDesc = outputPresentValueCastTensorDesc.GetDmlDesc();
        DML_CAST_OPERATOR_DESC outputPresentValueCastOpDesc{};
        outputPresentValueCastOpDesc.InputTensor = &outputPresentValueCastDmlTensorDesc;
        outputPresentValueCastOpDesc.OutputTensor = &outputDescs[outputPresentValueIndex];
        DML_OPERATOR_DESC outputPresentValueCastDmlDesc = { DML_OPERATOR_CAST, &outputPresentValueCastOpDesc };

        const bool isFp16 = m_inputTensorDescs[queryIndex].GetDmlDataType() == DML_TENSOR_DATA_TYPE_FLOAT16;

        DML_MULTIHEAD_ATTENTION1_OPERATOR_DESC mhaDesc = {};
        mhaDesc.QueryTensor = isFp16 ? &queryCastDmlTensorDesc : &inputDescs[queryIndex];
        mhaDesc.KeyTensor = isFp16 ? &keyCastDmlTensorDesc : &inputDescs[keyIndex];
        mhaDesc.ValueTensor = isFp16 ? &valueCastDmlTensorDesc : &inputDescs[valueIndex];
        mhaDesc.PastSequenceLengthsTensor = &pastSequenceLengthsDmlTensorDesc;
        mhaDesc.OutputTensor = isFp16 ? &outputCastDmlTensorDesc : &outputDescs[outputIndex];
        mhaDesc.OutputPresentKeyTensor = isFp16 ? &outputPresentKeyCastDmlTensorDesc : &outputDescs[outputPresentKeyIndex];
        mhaDesc.OutputPresentValueTensor = isFp16 ? &outputPresentValueCastDmlTensorDesc : &outputDescs[outputPresentValueIndex];
        mhaDesc.QueryHeadCount = queryNumHeads;
        mhaDesc.KeyValueHeadCount = kvNumHeads;
        mhaDesc.Scale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Scale, gsl::narrow_cast<float>(1.0f / std::sqrt(queryHeadSize)));
        mhaDesc.MaskFilterValue = -10'000.0f;
        DML_OPERATOR_DESC mhaDmlDesc = { DML_OPERATOR_MULTIHEAD_ATTENTION1, &mhaDesc };

        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC zeroScalarDesc = {};
        zeroScalarDesc.OutputTensor = &pastSequenceLengthsDmlTensorDesc;
        zeroScalarDesc.ValueDataType = pastSequenceLengthsTensorDesc.GetDmlDataType();
        DML_OPERATOR_DESC zeroScalarDmlDesc = { DML_OPERATOR_FILL_VALUE_CONSTANT, &zeroScalarDesc };

        std::vector<const DML_OPERATOR_DESC*> opDescs = {
            &mhaDmlDesc,
        };

        // Construct the graph
        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

        if (isFp16)
        {
            opDescs.push_back(&queryCastDmlDesc);
            opDescs.push_back(&keyCastDmlDesc);
            opDescs.push_back(&valueCastDmlDesc);
            opDescs.push_back(&outputCastDmlDesc);
            opDescs.push_back(&outputPresentKeyCastDmlDesc);
            opDescs.push_back(&outputPresentValueCastDmlDesc);

            // Link the query/key/value inputs to the cast nodes
            for (uint32_t i = 0; i < 3; ++i)
            {
                DML_INPUT_GRAPH_EDGE_DESC inputToMhaEdge = {};
                inputToMhaEdge.GraphInputIndex = i;
                inputToMhaEdge.ToNodeIndex = 1 + i;
                inputToMhaEdge.ToNodeInputIndex = 0;
                inputEdges.push_back(inputToMhaEdge);
            }

            // Link the input cast nodes to MHA
            for (uint32_t i = 0; i < 3; ++i)
            {
                DML_INTERMEDIATE_GRAPH_EDGE_DESC castToMhaEdge = {};
                castToMhaEdge.FromNodeIndex = 1 + i;
                castToMhaEdge.FromNodeOutputIndex = 0;
                castToMhaEdge.ToNodeIndex = 0;
                castToMhaEdge.ToNodeInputIndex = i;
                intermediateEdges.push_back(castToMhaEdge);
            }
        }
        else
        {
            // Link the query/key/value inputs to MHA
            for (uint32_t i = 0; i < 3; ++i)
            {
                DML_INPUT_GRAPH_EDGE_DESC inputToMhaEdge = {};
                inputToMhaEdge.GraphInputIndex = i;
                inputToMhaEdge.ToNodeIndex = 0;
                inputToMhaEdge.ToNodeInputIndex = i;
                inputEdges.push_back(inputToMhaEdge);
            }
        }

        constexpr uint32_t dmlPastSequenceLengthsIndex = 11;

        // The GQA offline fusion does this thing where it sums the number of 1's in the mask to figure out the value of the past sequence.
        // This doesn't work well for the first iteration since, obviously, there are no past sequences and the mask in this case represents
        // only the elements in the initial sequence. To work around this, the CUDA implementation of the operator ignores the value of
        // pastSequenceLengths for the first iteration and acts as if it was 0. This feels like a pretty dirty hack and something that should
        // be polished in the future, but for compatibility with the GQA fusion and the CUDA implementation we do the same thing here. We DO NOT
        // want to do this within DirectML since DirectML should be agnostic w.r.t which iteration it's currently executing MHA for, and such a
        // hack that is likely to be modified in the future shouldn't be enshrined within DirectML. Doing it here is OK because the nature of contrib
        // ops is that they can change at any time.
        if (sequenceLength == 1)
        {
            // Link the PastSequenceLengths input to MHA
            DML_INPUT_GRAPH_EDGE_DESC inputToMhaEdge = {};
            inputToMhaEdge.GraphInputIndex = seqLensIndex;
            inputToMhaEdge.ToNodeIndex = 0;
            inputToMhaEdge.ToNodeInputIndex = dmlPastSequenceLengthsIndex;
            inputEdges.push_back(inputToMhaEdge);
        }
        else
        {
            opDescs.push_back(&zeroScalarDmlDesc);

            // Link the zero scalar to MHA
            DML_INTERMEDIATE_GRAPH_EDGE_DESC zeroScalarToMhaEdge = {};
            zeroScalarToMhaEdge.FromNodeIndex = gsl::narrow_cast<uint32_t>(opDescs.size() - 1);
            zeroScalarToMhaEdge.FromNodeOutputIndex = 0;
            zeroScalarToMhaEdge.ToNodeIndex = 0;
            zeroScalarToMhaEdge.ToNodeInputIndex = dmlPastSequenceLengthsIndex;
            intermediateEdges.push_back(zeroScalarToMhaEdge);
        }

        if (isFp16)
        {
            // Output cast nodes start at the 4th index (previously we have the mha, query, key and value nodes)
            const uint32_t outputCastNodeStart = 4;

            // Link MHA's output to the output cast nodes
            for (uint32_t i = 0; i < 3; ++i)
            {
                DML_INTERMEDIATE_GRAPH_EDGE_DESC mhaToCastEdge = {};
                mhaToCastEdge.FromNodeIndex = 0;
                mhaToCastEdge.FromNodeOutputIndex = i;
                mhaToCastEdge.ToNodeIndex = outputCastNodeStart + i;
                mhaToCastEdge.ToNodeInputIndex = 0;
                intermediateEdges.push_back(mhaToCastEdge);
            }

            // Link the output cast nodes to the graph's outputs
            for (uint32_t i = 0; i < 3; ++i)
            {
                DML_OUTPUT_GRAPH_EDGE_DESC castToOutputEdge = {};
                castToOutputEdge.FromNodeIndex = outputCastNodeStart + i;
                castToOutputEdge.FromNodeOutputIndex = 0;
                castToOutputEdge.GraphOutputIndex = i;
                outputEdges.push_back(castToOutputEdge);
            }
        }
        else
        {
            // Link MHA's outputs to the graph's outputs
            for (uint32_t i = 0; i < 3; ++i)
            {
                DML_OUTPUT_GRAPH_EDGE_DESC mhaToOutputEdge = {};
                mhaToOutputEdge.FromNodeIndex = 0;
                mhaToOutputEdge.FromNodeOutputIndex = i;
                mhaToOutputEdge.GraphOutputIndex = i;
                outputEdges.push_back(mhaToOutputEdge);
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
        operatorGraphDesc.nodes = opDescs.data();
        SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(GroupQueryAttention, DmlOperatorGroupQueryAttention);
} // namespace Dml

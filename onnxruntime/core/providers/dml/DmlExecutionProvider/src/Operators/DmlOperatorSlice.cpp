// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorSlice : public DmlOperator, public SliceHelper
{
public:
    DmlOperatorSlice(const MLOperatorKernelCreationContext& kernelInfo, uint32_t opsetVersion)
    :   DmlOperator(kernelInfo),
        SliceHelper(kernelInfo, kernelInfo.GetTensorShapeDescription(), opsetVersion)
    {
        const uint32_t inputCount = kernelInfo.GetInputCount();
        ML_CHECK_VALID_ARGUMENT((opsetVersion <  10 && inputCount == 1)
                             || (opsetVersion >= 10 && inputCount >= 3 && inputCount <= 5));
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        std::vector<std::optional<uint32_t>> kernelInputIndices = { 0 }; // Only bind GPU to first 'data' tensor.
        DmlOperator::Initialize(kernelInfo, kernelInputIndices, std::nullopt, std::nullopt, std::nullopt, /*minimumDimensionCount*/ 1);

        const uint32_t inputTensorRank = m_inputTensorDescs[0].GetDimensionCount();
        assert(inputTensorRank >= gsl::narrow_cast<uint32_t>(m_offsets.size()));
        assert(inputTensorRank >= gsl::narrow_cast<uint32_t>(m_sizes.size()));
        assert(inputTensorRank >= gsl::narrow_cast<uint32_t>(m_strides.size()));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        int splitIndex = GetSplitIndex();

        if (splitIndex == -1)
        {
            DML_SLICE1_OPERATOR_DESC sliceDesc = {};
            sliceDesc.InputTensor = inputDescs.data();
            sliceDesc.OutputTensor = outputDescs.data();
            sliceDesc.DimensionCount = gsl::narrow_cast<uint32_t>(m_offsets.size());
            sliceDesc.InputWindowOffsets = m_offsets.data();
            sliceDesc.InputWindowSizes = m_sizes.data();
            sliceDesc.InputWindowStrides = m_strides.data();

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_SLICE1, &sliceDesc };
            SetDmlOperatorDesc(opDesc, kernelInfo);
        }
        else
        {
            MLOperatorTensorDataType dataType = kernelInfo.GetInputEdgeDescription(0).tensorDataType;
            auto inputSizes = m_inputTensorDescs[0].GetSizes();
            std::vector<uint32_t> leftOutputSizes(inputSizes.begin(), inputSizes.end());
            leftOutputSizes[splitIndex] = m_offsets[splitIndex];

            TensorDesc leftOutputTensorDesc = TensorDesc::ConstructDefaultTensorDesc(dataType, leftOutputSizes);

            std::array<DML_TENSOR_DESC, 2> splitOutputs = {
                leftOutputTensorDesc.GetDmlDesc(),
                outputDescs.back(),
            };

            DML_SPLIT_OPERATOR_DESC splitDesc = {};
            splitDesc.InputTensor = inputDescs.data();
            splitDesc.OutputTensors = splitOutputs.data();
            splitDesc.OutputCount = gsl::narrow_cast<uint32_t>(splitOutputs.size());
            splitDesc.Axis = static_cast<uint32_t>(splitIndex);
            const DML_OPERATOR_DESC splitDmlDesc {DML_OPERATOR_SPLIT, &splitDesc};

            std::array<const DML_OPERATOR_DESC*, 1> opDescs = {
                &splitDmlDesc,
            };

            std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
            std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;

            DML_INPUT_GRAPH_EDGE_DESC inputEdge = {};
            inputEdge.GraphInputIndex = 0;
            inputEdge.ToNodeIndex = 0;
            inputEdge.ToNodeInputIndex = 0;
            inputEdges.push_back(std::move(inputEdge));

            DML_OUTPUT_GRAPH_EDGE_DESC outputEdge = {};
            outputEdge.GraphOutputIndex = 0;
            outputEdge.FromNodeIndex = 0;
            outputEdge.FromNodeOutputIndex = 1;
            outputEdges.push_back(std::move(outputEdge));

            MLOperatorGraphDesc operatorGraphDesc = {};
            operatorGraphDesc.inputEdgeCount = gsl::narrow_cast<uint32_t>(inputEdges.size());
            operatorGraphDesc.inputEdges = inputEdges.data();
            operatorGraphDesc.outputEdgeCount = gsl::narrow_cast<uint32_t>(outputEdges.size());
            operatorGraphDesc.outputEdges = outputEdges.data();
            operatorGraphDesc.nodeCount = gsl::narrow_cast<uint32_t>(opDescs.size());
            operatorGraphDesc.nodesAsOpDesc = opDescs.data();

            SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelInfo);
        }
    }

private:
    int GetSplitIndex()
    {
        if (std::any_of(m_strides.begin(), m_strides.end(), [](int32_t stride){ return stride != 1; }))
        {
            return -1;
        }

        int axisIndex = -1;

        // For now, we only support cases where the left part of the tensor is getting cut off and we keep the right part
        for (uint32_t i = 0; i < m_offsets.size(); ++i)
        {
            if (m_offsets[i] != 0)
            {
                if (axisIndex != -1)
                {
                    return -1;
                }

                if (m_sizes[i] < m_inputTensorDescs[0].GetSizes()[i] - m_offsets[i])
                {
                    return -1;
                }

                axisIndex = i;
            }
        }

        return axisIndex;
    }
};

void CALLBACK QuerySlice(IMLOperatorSupportQueryContextPrivate* context, bool* isSupported)
{
    *isSupported = (context->GetInputCount() <= 5);
}

DML_OP_DEFINE_CREATION_FUNCTION(Slice7,  VersionedKernel<DmlOperatorSlice, 7> );
DML_OP_DEFINE_CREATION_FUNCTION(Slice10, VersionedKernel<DmlOperatorSlice, 10>);
DML_OP_DEFINE_CREATION_FUNCTION(Slice11, VersionedKernel<DmlOperatorSlice, 11>);
DML_OP_DEFINE_CREATION_FUNCTION(Slice13, VersionedKernel<DmlOperatorSlice, 13>);
} // namespace Dml

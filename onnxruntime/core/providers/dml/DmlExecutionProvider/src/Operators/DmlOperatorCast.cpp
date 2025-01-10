// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorCast : public DmlOperator
{
public:
    using Self = DmlOperatorCast;

    DmlOperatorCast(
        const MLOperatorKernelCreationContext& kernelInfo
        ) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);
        std::vector<std::optional<uint32_t>> inputIndices = { 0 }; // For CastLike, the second tensor ('target_type') is not bound.
        std::vector<std::optional<uint32_t>> outputIndices = { 0 };
        DmlOperator::Initialize(kernelInfo, inputIndices, outputIndices);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_CAST_OPERATOR_DESC castDesc = {};
        castDesc.InputTensor = inputDescs.data();
        castDesc.OutputTensor = outputDescs.data();

        if (kernelInfo.GetOutputEdgeDescription(0).tensorDataType == static_cast<MLOperatorTensorDataType>(ONNX_NAMESPACE::TensorProto_DataType_BOOL))
        {
            DML_OPERATOR_DESC dmlCastDesc = { DML_OPERATOR_CAST, &castDesc };

            DML_ELEMENT_WISE_CLIP1_OPERATOR_DESC clipDesc = {};
            clipDesc.InputTensor = outputDescs.data();
            clipDesc.OutputTensor = outputDescs.data();
            clipDesc.Min.UInt8 = 0;
            clipDesc.Max.UInt8 = 1;

            DML_OPERATOR_DESC dmlClipDesc = { DML_OPERATOR_ELEMENT_WISE_CLIP1, &clipDesc };

            std::vector<const DML_OPERATOR_DESC*> opDescs = { &dmlCastDesc, &dmlClipDesc };

            DML_INPUT_GRAPH_EDGE_DESC inputToCastEdge = {};
            inputToCastEdge.GraphInputIndex = 0;
            inputToCastEdge.ToNodeIndex = 0;
            inputToCastEdge.ToNodeInputIndex = 0;

            DML_INTERMEDIATE_GRAPH_EDGE_DESC castToClipEdge = {};
            castToClipEdge.FromNodeIndex = 0; 
            castToClipEdge.FromNodeOutputIndex = 0;
            castToClipEdge.ToNodeIndex = 1;
            castToClipEdge.ToNodeInputIndex = 0;

            DML_OUTPUT_GRAPH_EDGE_DESC clipToOutputEdge = {};
            clipToOutputEdge.FromNodeIndex = 1;
            clipToOutputEdge.FromNodeOutputIndex = 0;
            clipToOutputEdge.GraphOutputIndex = 0;

            MLOperatorGraphDesc operatorGraphDesc = {};
            operatorGraphDesc.nodeCount = gsl::narrow_cast<uint32_t>(opDescs.size());
            operatorGraphDesc.nodes = opDescs.data();

            operatorGraphDesc.inputEdgeCount = 1;
            operatorGraphDesc.inputEdges = &inputToCastEdge;

            operatorGraphDesc.intermediateEdgeCount = 1;
            operatorGraphDesc.intermediateEdges = &castToClipEdge;

            operatorGraphDesc.outputEdgeCount = 1;
            operatorGraphDesc.outputEdges = &clipToOutputEdge;

            SetDmlOperatorGraphDesc(std::move(operatorGraphDesc), kernelInfo);
        }
        else
        {        
            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_CAST, &castDesc };
            SetDmlOperatorDesc(opDesc, kernelInfo);
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensorsForExecute(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensorsForExecute(kernelContext);

        ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::make_span(inputTensors),
            gsl::make_span(outputTensors)
        ));
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Cast, DmlOperatorCast);
DML_OP_DEFINE_CREATION_FUNCTION(CastLike15, DmlOperatorCast);
DML_OP_DEFINE_CREATION_FUNCTION(CastLike19, DmlOperatorCast);
DML_OP_DEFINE_CREATION_FUNCTION(CastLike21, DmlOperatorCast);

} // namespace Dml

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorConcat : public DmlOperator, public ConcatHelper
{
public:
    using Self = DmlOperatorConcat;

    DmlOperatorConcat(const MLOperatorKernelCreationContext& kernelInfo)
    :   DmlOperator(kernelInfo),
        ConcatHelper(kernelInfo, kernelInfo.GetTensorShapeDescription())
    {
        auto tensorShapeDescription = kernelInfo.GetTensorShapeDescription();
        std::vector<std::optional<uint32_t>> kernelInputIndices;

        std::vector<DimensionType> tensorShape;

        for (uint32_t i = 0; i < kernelInfo.GetInputCount(); i++)
        {
            // Only keep the non-empty tensors
            if (!OperatorHelper::ContainsEmptyDimensions(tensorShapeDescription.GetInputTensorShape(i)))
            {
                kernelInputIndices.push_back(i);
            }
        }

        DmlOperator::Initialize(kernelInfo, kernelInputIndices);

        // Only execute Concat if it has at least one non-empty input
        if (!m_inputTensorDescs.empty())
        {
            uint32_t dmlAxis = GetDmlAdjustedAxis(m_axis, kernelInfo, m_inputTensorDescs.front().GetDimensionCount());

            std::vector<DML_TENSOR_DESC> inputDescs;
            inputDescs.reserve(m_inputTensorDescs.size());

            for (size_t i = 0; i < m_inputTensorDescs.size(); i++)
            {
                // DML doesn't support empty tensors for concat, so we ignore them
                if (!OperatorHelper::ContainsEmptyDimensions(m_inputTensorDescs[i].GetSizes()))
                {
                    inputDescs.push_back(m_inputTensorDescs[i].GetDmlDesc());
                }
            }

            std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

            DML_JOIN_OPERATOR_DESC joinDesc = {};
            joinDesc.InputCount = gsl::narrow_cast<uint32_t>(inputDescs.size());
            joinDesc.InputTensors = inputDescs.data();
            joinDesc.OutputTensor = outputDescs.data();
            joinDesc.Axis = dmlAxis;

            DML_OPERATOR_DESC opDesc = { DML_OPERATOR_JOIN, &joinDesc };

            SetDmlOperatorDesc(opDesc, kernelInfo);
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        std::vector<IMLOperatorTensor*> inputTensors = GetInputTensorsForExecute(kernelContext);
        std::vector<IMLOperatorTensor*> outputTensors = GetOutputTensorsForExecute(kernelContext);

        if (!inputTensors.empty())
        {
            ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
                kernelContext.GetAllocator(),
                m_compiledOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                gsl::make_span(inputTensors),
                gsl::make_span(outputTensors)));
        }
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(Concat, DmlOperatorConcat);

} // namespace Dml

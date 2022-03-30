// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorLayerNormalization : public DmlOperator
{
public:
    DmlOperatorLayerNormalization(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        std::cout<<"DML Contrib Ops gets invoked\n";
        std::vector<std::optional<uint32_t>> kernelInputIndices = {0, 1, 2};
        DmlOperator::Initialize(kernelCreationContext, kernelInputIndices);

        const float epsilon = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Epsilon, 0.0f);
        
        const std::vector<DimensionType> inputDimensions = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        const int32_t inputDimCount = gsl::narrow_cast<int32_t>(inputDimensions.size());
        
        std::vector<int32_t> onnxAxes = kernelCreationContext.GetOptionalAttributeVectorInt32(AttrName::Axes);
        if (onnxAxes.empty())
        {
            //int32_t defaultAxesArr[] = {1, 2, 3};
            //gsl::span<int32_t> defaultAxes(acrossChannels ? gsl::make_span(crossChannelAxes) : gsl::make_span(nonChannelAxes));
            //gsl::span<int32_t> defaultAxes(gsl::make_span(defaultAxesArr));
            //onnxAxes.assign(defaultAxes.begin(), defaultAxes.end());
            onnxAxes.clear();
            onnxAxes.resize(inputDimCount - 1);
            std::iota(onnxAxes.begin(), onnxAxes.end(), 1);
        }

        std::vector<uint32_t> dmlAxes;
        GetDmlAdjustedAxes(onnxAxes, inputDimCount, m_inputTensorDescs.front().GetDimensionCount(), /*out*/ dmlAxes);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        //std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelCreationContext);
        //DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        DML_MEAN_VARIANCE_NORMALIZATION1_OPERATOR_DESC operatorDesc = {};
        operatorDesc.InputTensor = &inputDescs[0];
        operatorDesc.ScaleTensor = &inputDescs[1];
        operatorDesc.BiasTensor = &inputDescs[2];
        operatorDesc.OutputTensor = outputDescs.data();
        operatorDesc.Axes = dmlAxes.data();
        operatorDesc.AxisCount = gsl::narrow_cast<uint32_t>(dmlAxes.size());
        operatorDesc.NormalizeVariance = false;
        operatorDesc.Epsilon = epsilon;
        operatorDesc.FusedActivation = nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION1, &operatorDesc };
        SetDmlOperatorDesc(opDesc, kernelCreationContext);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(LayerNormalization, DmlOperatorLayerNormalization);
//DML_OP_DEFINE_CREATION_FUNCTION(FusedMeanVarianceNormalization, DmlOperatorMeanVarNormalization);

} // namespace Dml

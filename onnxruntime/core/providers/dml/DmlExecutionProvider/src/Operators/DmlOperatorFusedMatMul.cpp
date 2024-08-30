// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorFusedMatMul : public DmlOperator
{

public:
    DmlOperatorFusedMatMul(const MLOperatorKernelCreationContext& kernelInfo)
        :   DmlOperator(kernelInfo)
    {
        // FusedMatMul has two inputs, but DML GEMM requires 3 input bindings (a null binding for the C Tensor).
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 2);

        // Need these shapes to apply transpose and
        // numpy MatMul's behavior https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.matmul.html
        std::vector<DimensionType> inputShape0 = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(0);
        std::vector<DimensionType> inputShape1 = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(1);
        std::vector<DimensionType> outputShape = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);

        const int32_t transBatchA = kernelInfo.GetOptionalAttribute<int32_t>(AttrName::TransBatchA, 0);
        const int32_t transA = kernelInfo.GetOptionalAttribute<int32_t>(AttrName::TransA, 0);
        const int32_t transBatchB = kernelInfo.GetOptionalAttribute<int32_t>(AttrName::TransBatchB, 0);
        const int32_t transB = kernelInfo.GetOptionalAttribute<int32_t>(AttrName::TransB, 0);

        // As of now, CPU FusedMatMul has this extra validation
        // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/providers/cpu/math/matmul_helper.h#L72
        // Although, DML kernel can work without this validation, but adding this just to be in sync.
        if (transBatchA || transBatchB)
        {
            ML_CHECK_VALID_ARGUMENT(inputShape0.size() > 2 && inputShape0.size() == inputShape1.size(),
                  "Two inputs should have same rank and rank >= 3 if transBatchA or transBatchB is true");
        }

        auto [sizesA, stridesA] = OperatorHelper::GetFusedMatMulSizesAndStrides(inputShape0, transBatchA, transA);

        auto [sizesB, stridesB] = OperatorHelper::GetFusedMatMulSizesAndStrides(inputShape1, transBatchB, transB);

        OperatorHelper::FusedMatMulShapeMapping(sizesA, stridesA, sizesB, stridesB, outputShape);

        // At this point, we have manipulated input/output shapes and strides and
        // we do not care about actual input shapes present in the model (.onnx file).
        // Create the TensorDesc with the manipulated input shapes because we don't want incorrect
        // broadcasting to be happen inside TensorDesc constructor.
        std::vector<std::optional<uint32_t>> inputIndices = { 0, 1, std::nullopt };
        gsl::span<const uint32_t> inputShapes[2] = {sizesA, sizesB};
        gsl::span<const uint32_t> outputShapes[1] = {outputShape};
        DmlOperator::InitializeWithShapes(kernelInfo, inputIndices, std::nullopt, inputShapes, outputShapes, 1);

        m_inputTensorDescs[0].SetStrides(stridesA);
        m_inputTensorDescs[1].SetStrides(stridesB);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        std::optional<ActivationOperatorDescWrapper> fusedActivation = FusionHelpers::TryGetGraphFusedActivationDesc(kernelInfo);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->desc.GetDmlDesc() : DML_OPERATOR_DESC();

        const float alpha = kernelInfo.GetOptionalAttribute<float>(AttrName::Alpha, 1.0f);

        DML_GEMM_OPERATOR_DESC gemmDesc = {};
        gemmDesc.ATensor = &inputDescs[0];
        gemmDesc.BTensor = &inputDescs[1];
        gemmDesc.CTensor = nullptr;
        gemmDesc.OutputTensor = &outputDescs[0];
        gemmDesc.TransA = (transA ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE);
        gemmDesc.TransB = (transB ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE);
        gemmDesc.Alpha = alpha;
        gemmDesc.Beta = 0.0f;
        gemmDesc.FusedActivation = fusedActivation ? &fusedActivationDmlDesc : nullptr;

        DML_OPERATOR_DESC opDesc = { DML_OPERATOR_GEMM, &gemmDesc };
        SetDmlOperatorDesc(opDesc, kernelInfo);
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(FusedMatMul, DmlOperatorFusedMatMul);
DML_OP_DEFINE_CREATION_FUNCTION(FusedMatMulActivation, DmlOperatorFusedMatMul);

} // namespace Dml

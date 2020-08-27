// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

class DmlOperatorEinSum : public DmlOperator, public EinSumHelper
{
public:
    DmlOperatorEinSum(const MLOperatorKernelCreationContext& kernelCreationContext, uint32_t opsetVersion)
    :   DmlOperator(kernelCreationContext), 
        EinSumHelper(kernelCreationContext, kernelCreationContext.GetTensorShapeDescription(), opsetVersion)
    {
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() + 1 == m_components.size(), "EinSum input tensor count is inconsistent with the equation component count.");
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1, "EinSum expects one output tensor.");

        DmlOperator::Initialize(kernelCreationContext);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        static_assert(RecognizedOperatorType::Total == static_cast<RecognizedOperatorType>(8), "Update this switch.");
        switch (m_recognizedOperatorType)
        {
        case RecognizedOperatorType::Multiply:
            {
                DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC operatorDesc = {};
                operatorDesc.ATensor = &inputDescs[0];
                operatorDesc.BTensor = &inputDescs[1];
                operatorDesc.OutputTensor = outputDescs.data();

                SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &operatorDesc}, kernelCreationContext);
            }
            break;

        case RecognizedOperatorType::MatMul:
        case RecognizedOperatorType::MatMulTransposeA:
        case RecognizedOperatorType::MatMulTransposeB:
            {
                DML_GEMM_OPERATOR_DESC operatorDesc = {};
                operatorDesc.ATensor = &inputDescs[0];
                operatorDesc.BTensor = &inputDescs[1];
                // No operatorDesc.CTensor
                operatorDesc.OutputTensor = &outputDescs[0];
                operatorDesc.TransA = (m_recognizedOperatorType == RecognizedOperatorType::MatMulTransposeA) ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
                operatorDesc.TransB = (m_recognizedOperatorType == RecognizedOperatorType::MatMulTransposeB) ? DML_MATRIX_TRANSFORM_TRANSPOSE : DML_MATRIX_TRANSFORM_NONE;
                operatorDesc.Alpha = 1.0;
                operatorDesc.Beta = 0.0;
                operatorDesc.FusedActivation = nullptr;

                SetDmlOperatorDesc({ DML_OPERATOR_GEMM, &operatorDesc }, kernelCreationContext);
            }
            break;

        case RecognizedOperatorType::ReduceSum:
            {
                // Get how many axes are kept in the final output, either 0 or 1 supported
                // meaning full reduction or partial with one dimension left. *It could be
                // generalized to support any number of output dimensions, but it would need
                // to accomodate for Transposition too if the output labels are reordered.
                auto keptAxes = m_components.back().GetLabels(m_labelIndices);
                assert(keptAxes.size() <= 1);

                // DML expects output rank to match input rank (as if ONNX ReduceSum keepdims=1).
                // So replace the existing tensor description with the input sizes, except that
                // reduced dimensions have size 1.
                std::vector<uint32_t> reducedAxes;
                std::vector<uint32_t> inputSizes = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
                std::vector<uint32_t> outputSizes = inputSizes;

                // Determine which axes are being reduced by taking the opposite of those kept.
                uint32_t keptAxesMask = 0;
                for (auto axis : keptAxes)
                {
                    keptAxesMask |= (1 << axis);
                }
                for (uint32_t axis = 0, axisCount = static_cast<uint32_t>(outputSizes.size()); axis < axisCount; ++axis)
                {
                    if (~keptAxesMask & (1<<axis))
                    {
                        reducedAxes.push_back(axis);
                        outputSizes[axis] = 1;
                    }
                }

                m_inputTensorDescs.front() = TensorDesc(m_inputTensorDescs.front().GetDmlDataType(), inputSizes, std::nullopt, 0);
                m_outputTensorDescs.front() = TensorDesc(m_outputTensorDescs.front().GetDmlDataType(), outputSizes, std::nullopt, 0);
                m_inputTensorDescs.front().GetDmlDesc(); // Discard value, but keep side effect of refreshing the DML view.
                m_outputTensorDescs.front().GetDmlDesc(); // Discard value, but keep side effect of refreshing the DML view.

                DML_REDUCE_OPERATOR_DESC operatorDesc = {};
                operatorDesc.InputTensor = inputDescs.data();
                operatorDesc.OutputTensor = outputDescs.data();
                operatorDesc.Function = DML_REDUCE_FUNCTION_SUM;
                operatorDesc.Axes = reducedAxes.data();
                operatorDesc.AxisCount = gsl::narrow_cast<uint32_t>(reducedAxes.size());

                SetDmlOperatorDesc({ DML_OPERATOR_REDUCE, &operatorDesc }, kernelCreationContext);
            }
            break;

        case RecognizedOperatorType::Transpose:
        case RecognizedOperatorType::Identity:
            {
                if (m_recognizedOperatorType == RecognizedOperatorType::Transpose)
                {
                    // Transpose via input strides. The output tensor is not strided.
                    assert(m_components.front().GetDimensionCount() == m_components.back().GetDimensionCount());
                    auto originalStrides = m_inputTensorDescs.front().GetStrides();
                    std::vector<uint32_t> inputSizes = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
                    std::vector<uint32_t> inputStrides(inputSizes.size());

                    // If there were no strides, compute them based in descending packed order
                    // based on the input sizes.
                    if (originalStrides.empty())
                    {
                        Dml::GetDescendingPackedStrides(inputSizes, /*out*/ inputStrides);
                    }
                    else // Copy the original strides.
                    {
                        assert(originalStrides.size() >= inputStrides.size());
                        size_t offset = originalStrides.size() - inputStrides.size();
                        inputStrides.assign(originalStrides.begin() + offset, originalStrides.end());
                    }

                    // Remap transposed strides using the component labels from input to output.
                    auto labelIndices = m_components.back().GetLabels(m_labelIndices);

                    std::vector<uint32_t> newStrides(inputStrides.size());
                    std::vector<uint32_t> newSizes(inputStrides.size());
                    for (size_t i = 0, dimensionCount = inputStrides.size(); i < dimensionCount; ++i)
                    {
                        uint32_t labelIndex = labelIndices[i];
                        assert(labelIndex < inputStrides.size());
                        newSizes[i] = inputSizes[labelIndex];
                        newStrides[i] = inputStrides[labelIndex];
                    }

                    // Override the initial input tensor with the new strides.
                    m_inputTensorDescs.front() = TensorDesc(m_inputTensorDescs.front().GetDmlDataType(), newSizes, newStrides, 0);
                    m_outputTensorDescs.front() = TensorDesc(m_outputTensorDescs.front().GetDmlDataType(), newSizes, std::nullopt, 0);
                    m_inputTensorDescs.front().GetDmlDesc(); // Discard value, but keep side effect of refreshing the DML view.
                    m_outputTensorDescs.front().GetDmlDesc(); // Discard value, but keep side effect of refreshing the DML view.
                }

                DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC operatorDesc = {};
                operatorDesc.InputTensor = inputDescs.data();
                operatorDesc.OutputTensor = outputDescs.data();

                SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &operatorDesc}, kernelCreationContext);
            }
            break;

        default:
            return;
        }
    }
};

void CALLBACK QueryEinSum(IMLOperatorSupportQueryContextPrivate* context, bool* isSupported)
{
    *isSupported = false;

    MLOperatorAttributes attributes(context);
    EinSumHelper helper(attributes);
    auto recognizedOperatorType = helper.GetRecognizedOperatorType();

    static_assert(EinSumHelper::RecognizedOperatorType::Total == static_cast<EinSumHelper::RecognizedOperatorType>(8), "Verify this test still matches the switch above.");
    *isSupported = (recognizedOperatorType != EinSumHelper::RecognizedOperatorType::None);
}

DML_OP_DEFINE_CREATION_FUNCTION(Einsum12, VersionedKernel<DmlOperatorEinSum, 12>);

} // namespace Dml

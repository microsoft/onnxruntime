// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"

namespace Dml
{

bool AreAllStridesIdentical(gsl::span<const TensorDesc> tensorDescs)
{
    const size_t tensorDescCount = tensorDescs.size();

    for (size_t i = 1; i < tensorDescCount; ++i)
    {
        gsl::span<const uint32_t> stridesA = tensorDescs[i - 1].GetStrides();
        gsl::span<const uint32_t> stridesB = tensorDescs[i].GetStrides();
        if (stridesA.size() != stridesB.size() || !std::equal(stridesA.begin(), stridesA.end(), stridesB.begin()))
        {
            return false;
        }
    }

    return true;
}

template <typename TOperatorDesc>
class DmlOperatorElementwiseUnary : public DmlOperator
{
public:
    DmlOperatorElementwiseUnary(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        TOperatorDesc opDesc = {};
        opDesc.InputTensor = inputDescs.data();
        opDesc.OutputTensor = outputDescs.data();

        SetDmlOperatorDesc({ ApiTraits::OperatorDescTraits<TOperatorDesc>::Type, &opDesc }, kernelInfo);
    }
};

template <>
class DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_ABS_OPERATOR_DESC> : public DmlOperator
{
public:
    DmlOperatorElementwiseUnary(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        assert(inputDescs[0].Type == DML_TENSOR_TYPE_BUFFER);
        if (IsSigned(reinterpret_cast<const DML_BUFFER_TENSOR_DESC*>(inputDescs[0].Desc)->DataType))
        {
            DML_ELEMENT_WISE_ABS_OPERATOR_DESC opDesc = {};
            opDesc.InputTensor = inputDescs.data();
            opDesc.OutputTensor = outputDescs.data();

            SetDmlOperatorDesc({ ApiTraits::OperatorDescTraits<DML_ELEMENT_WISE_ABS_OPERATOR_DESC>::Type, &opDesc }, kernelInfo);
        }
        else
        {
            // DML doesn't support UINT datatypes. So redirect to Identity because Abs doesn't do anything to UINT.
            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC opDesc = {};
            opDesc.InputTensor = inputDescs.data();
            opDesc.OutputTensor = outputDescs.data();

            SetDmlOperatorDesc({ ApiTraits::OperatorDescTraits<DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC>::Type, &opDesc }, kernelInfo);
        }
    }
};

template<typename T>
void SetFusedActivation(T& opDesc, const DML_OPERATOR_DESC* fusedActivation)
{
    // Activation is only fused for sum operators, which have a template specialization
    ORT_THROW_HR(E_INVALIDARG);
}

template<>
void SetFusedActivation(DML_ELEMENT_WISE_ADD1_OPERATOR_DESC& opDesc, const DML_OPERATOR_DESC* fusedActivation)
{
    opDesc.FusedActivation = fusedActivation;
}

template <typename TOperatorDesc>
class DmlOperatorElementwiseBinary : public DmlOperator
{
public:
    DmlOperatorElementwiseBinary(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 2);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelInfo);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        TOperatorDesc opDesc = {};
        opDesc.ATensor = &inputDescs[0];
        opDesc.BTensor = &inputDescs[1];
        opDesc.OutputTensor = outputDescs.data();

        DML_OPERATOR_DESC opDescDesc = { ApiTraits::OperatorDescTraits<TOperatorDesc>::Type, &opDesc};

        if (fusedActivation != std::nullopt)
        {
            // Activation is only fused for two-input sum operators
            ORT_THROW_HR_IF(E_INVALIDARG, opDescDesc.Type != DML_OPERATOR_ELEMENT_WISE_ADD1 || kernelInfo.GetInputCount() > 2);

            SetFusedActivation(opDesc, &fusedActivationDmlDesc);
        }

        SetDmlOperatorDesc(opDescDesc, kernelInfo);
    }
};

ComPtr<IDMLCompiledOperator> CreateSecondaryOperator(
    IDMLDevice* dmlDevice,
    DML_EXECUTION_FLAGS executionFlags,
    const DML_OPERATOR_DESC& operatorDesc,
    const MLOperatorKernelCreationContext& kernelInfo
    )
{
    ComPtr<IDMLOperator> dmlOperator;
    ComPtr<IDMLCompiledOperator> compiledOperator;
    ORT_THROW_IF_FAILED(dmlDevice->CreateOperator(&operatorDesc, IID_PPV_ARGS(&dmlOperator)));
    ORT_THROW_IF_FAILED(dmlDevice->CompileOperator(dmlOperator.Get(), executionFlags, IID_PPV_ARGS(&compiledOperator)));
    return compiledOperator;
}

template <typename TOperatorDesc>
class DmlOperatorElementwiseBinaryLoop : public DmlOperator
{
public:
    DmlOperatorElementwiseBinaryLoop(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();
        const size_t inputCount = m_inputTensorDescs.size();

        std::optional<ActivationOperatorDesc> fusedActivation = FusionHelpers::TryGetFusedActivationDesc(kernelInfo);
        DML_OPERATOR_DESC fusedActivationDmlDesc = fusedActivation ? fusedActivation->GetDmlDesc() : DML_OPERATOR_DESC();

        // Activation is only fused for two-input sum operators
        ORT_THROW_HR_IF(E_INVALIDARG, fusedActivation != std::nullopt && inputCount != 2);

        if (inputCount == 1)
        {
            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identityDesc = {};
            identityDesc.InputTensor = &inputDescs[0];
            identityDesc.OutputTensor = &outputDescs[0];
            SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &identityDesc }, kernelInfo);
        }
        else
        {
            // Create a single operator that applies to pairwise to every two inputs,
            // accumulated into the output tensor.

            TOperatorDesc opDesc = {};
            opDesc.ATensor = &inputDescs[0];
            opDesc.BTensor = &inputDescs[1];
            opDesc.OutputTensor = outputDescs.data();

            DML_OPERATOR_DESC opDescDesc = { ApiTraits::OperatorDescTraits<TOperatorDesc>::Type, &opDesc};

            if (fusedActivation != std::nullopt)
            {
                SetFusedActivation(opDesc, &fusedActivationDmlDesc);
            }

            SetDmlOperatorDesc(opDescDesc, kernelInfo);

            // If the tensor strides differ between pairs, then it's unsafe to reuse the same operator
            // for all pairs because the wrong stride would be used. So create operators for every additional
            // pair after the first. Given tensors {A, B, C}, the first operator handles A&B, the secondary
            // operator handles tensors B&C, and any additional after that would need another operator.

            if (inputCount >= 2 && !AreAllStridesIdentical(m_inputTensorDescs))
            {
                const DML_EXECUTION_FLAGS executionFlags = GetExecutionFlags();
                gsl::span<const DML_TENSOR_DESC> remainingInputDescs = gsl::make_span(inputDescs);
                remainingInputDescs = remainingInputDescs.subspan(2, remainingInputDescs.size() - 2);

                for (const DML_TENSOR_DESC& tensorDesc : remainingInputDescs)
                {
                    opDesc.ATensor = &tensorDesc;
                    opDesc.BTensor = &outputDescs[0];
                    // Already set - tOpDesc.OutputTensor = &outputDescs[0];
                    m_compiledOperators.push_back(CreateSecondaryOperator(m_dmlDevice.Get(), executionFlags, opDescDesc, kernelInfo));
                }
            }
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        // For 1 input, just return the input (identity).
        if (m_inputTensorDescs.size() == 1)
        {
            DmlOperator::Compute(kernelContext);
            return;
        }

        // Apply the operator to the first two inputs.
        std::array<IMLOperatorTensor*, 2> inputTensors;
        inputTensors[0] = kernelContext.GetInputTensor(0).GetInterface().Get();
        inputTensors[1] = kernelContext.GetInputTensor(1).GetInterface().Get();

        IMLOperatorTensor* outputTensor = kernelContext.GetOutputTensor(0).GetInterface().Get();
        gsl::span<IMLOperatorTensor*> outputTensors{ &outputTensor, 1 };

        // Combine the first two inputs and store the result in the output tensor.
        ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::make_span(inputTensors),
            outputTensors));

        // For each input after the first two, accumulate into the output tensor.
        for (size_t inputIndex = 2; inputIndex < m_inputTensorDescs.size(); ++inputIndex)
        {
            inputTensors[0] = kernelContext.GetInputTensor(gsl::narrow_cast<uint32_t>(inputIndex)).GetInterface().Get();
            inputTensors[1] = outputTensors[0];

            // Get the next operator for this pair, either reusing the first or using a distinct operator.
            IDMLCompiledOperator* compiledOperator = m_compiledOperators.empty()
                                                   ? m_compiledOperator.Get()
                                                   : m_compiledOperators[inputIndex - 2].Get();

            ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
                compiledOperator,
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                gsl::make_span(inputTensors),
                outputTensors));
        }
    }

    // If multiple compiled operators are needed, beyond m_compiledOperator, they are appended here.
    // The size of the vector will either be empty if all tensor pairs have identical properties,
    // or it will equal inputCount - 2, with the first operator in this vector corresponding to the
    // 3rd input tensor combined with the output of the previous 2 input tensors.
    std::vector<ComPtr<IDMLCompiledOperator>> m_compiledOperators;
};

class DmlOperatorElementwiseMean : public DmlOperator
{
    // Used with 3+ inputs to divide each element by the number of input tensors.
    ComPtr<IDMLCompiledOperator> m_compiledIdentityOp;

public:
    DmlOperatorElementwiseMean(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        const size_t inputCount = m_inputTensorDescs.size();
        if (inputCount == 1)
        {
            // For 1 input, just return the input
            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identityDesc = {};
            identityDesc.InputTensor = &inputDescs[0];
            identityDesc.OutputTensor = &outputDescs[0];
            SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IDENTITY, &identityDesc }, kernelInfo);
        }
        else if (inputCount == 2)
        {
            // For 2 inputs, use DML's mean operator.
            DML_ELEMENT_WISE_MEAN_OPERATOR_DESC meanDesc = {};
            meanDesc.ATensor = &inputDescs[0];
            meanDesc.BTensor = &inputDescs[1];
            meanDesc.OutputTensor = &outputDescs[0];

            SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_MEAN, &meanDesc}, kernelInfo);
        }
        else
        {
            // For 3+ inputs, use several DML adds followed by a divide (identity with scale=1/InputCount).
            assert(inputDescs.size() > 2);

            DML_ELEMENT_WISE_ADD_OPERATOR_DESC addDesc = {};
            addDesc.ATensor = &inputDescs[0];
            addDesc.BTensor = &inputDescs[1];
            addDesc.OutputTensor = &outputDescs[0];
            DML_OPERATOR_DESC addDescDesc = { DML_OPERATOR_ELEMENT_WISE_ADD, &addDesc};

            SetDmlOperatorDesc(addDescDesc, kernelInfo);

            if (!AreAllStridesIdentical(m_inputTensorDescs))
            {
                // Create operators for each input after the first two.
                const DML_EXECUTION_FLAGS executionFlags = GetExecutionFlags();
                gsl::span<const DML_TENSOR_DESC> remainingInputDescs = gsl::make_span(inputDescs);
                remainingInputDescs = remainingInputDescs.subspan(2, remainingInputDescs.size() - 2);

                for (const DML_TENSOR_DESC& tensorDesc : remainingInputDescs)
                {
                    addDesc.ATensor = &tensorDesc;
                    addDesc.BTensor = &outputDescs[0];
                    // Already set - addDesc.OutputTensor = &outputDescs[0];
                    m_compiledOperators.push_back(CreateSecondaryOperator(m_dmlDevice.Get(), executionFlags, addDescDesc, kernelInfo));
                }
            }

            // Create division operation using reciprocal of input tensor count.
            DML_SCALE_BIAS scaleBias = {};
            scaleBias.Scale = 1.0f / inputDescs.size();

            DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC identityDesc = {};
            identityDesc.InputTensor = &outputDescs[0];
            identityDesc.OutputTensor = &outputDescs[0];
            identityDesc.ScaleBias = &scaleBias;

            DML_OPERATOR_DESC identityDescDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &identityDesc };

            ComPtr<IDMLOperator> identityOp;
            ORT_THROW_IF_FAILED(m_dmlDevice->CreateOperator(&identityDescDesc, IID_PPV_ARGS(&identityOp)));

            ORT_THROW_IF_FAILED(m_dmlDevice->CompileOperator(identityOp.Get(), GetExecutionFlags(), IID_PPV_ARGS(&m_compiledIdentityOp)));
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext)
    {
        // Where there's only a single element, just return the input (identity).
        if (m_inputTensorDescs.size() == 1)
        {
            DmlOperator::Compute(kernelContext);
        }
        else if (!m_compiledIdentityOp)
        {
            // Use DML mean operator.
            DmlOperator::Compute(kernelContext);
        }
        else
        {
            // Do N-1 adds followed by a division, where N is the number of inputs.
            std::array<IMLOperatorTensor*, 2> inputTensors;
            inputTensors[0] = kernelContext.GetInputTensor(0).GetInterface().Get();
            inputTensors[1] = kernelContext.GetInputTensor(1).GetInterface().Get();

            IMLOperatorTensor* outputTensor = kernelContext.GetOutputTensor(0).GetInterface().Get();
            gsl::span<IMLOperatorTensor*> outputTensors{ &outputTensor, 1 };

            // Add the first two inputs and store the result in the output tensor.
            ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
                m_compiledOperator.Get(),
                m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                gsl::make_span(inputTensors),
                outputTensors));

            // For each input after the first two, accumulate into the output tensor.
            for (size_t inputIndex = 2; inputIndex < m_inputTensorDescs.size(); ++inputIndex)
            {
                inputTensors[0] = kernelContext.GetInputTensor(gsl::narrow_cast<uint32_t>(inputIndex)).GetInterface().Get();
                inputTensors[1] = outputTensors[0];

                // Get the next operator for this pair, either reusing the first or using a distinct operator.
                IDMLCompiledOperator* compiledOperator = m_compiledOperators.empty()
                    ? m_compiledOperator.Get()
                    : m_compiledOperators[inputIndex - 2].Get();

                ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
                    compiledOperator,
                    m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
                    gsl::make_span(inputTensors),
                    outputTensors));
            }

            // Dispatch the identity w/ scale operator in-place on the output.
            ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
                m_compiledIdentityOp.Get(),
                nullptr, // persistent resoruce binding
                outputTensors,
                outputTensors));
        }
    }

    // If multiple compiled operators are needed, beyond m_compiledOperator, they are appended here.
    std::vector<ComPtr<IDMLCompiledOperator>> m_compiledOperators;
};

class DmlOperatorElementwiseClip7 : public DmlOperator
{
public:
    DmlOperatorElementwiseClip7(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ELEMENT_WISE_CLIP_OPERATOR_DESC opDesc = {};
        opDesc.InputTensor = inputDescs.data();
        opDesc.OutputTensor = outputDescs.data();
        opDesc.Min = kernelInfo.GetOptionalAttribute<float>(AttrName::Min, std::numeric_limits<float>::lowest());
        opDesc.Max = kernelInfo.GetOptionalAttribute<float>(AttrName::Max, std::numeric_limits<float>::max());

        SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_CLIP, &opDesc}, kernelInfo);
    }
};

class DmlOperatorElementwiseClip11 : public DmlOperator
{
public:
    DmlOperatorElementwiseClip11(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() >= 1 && kernelInfo.GetInputCount() <= 3);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        std::vector<std::optional<uint32_t>> inputIndices = {0}; // min and max (1 and 2) are CPU-bound.
        std::vector<std::optional<uint32_t>> outputIndices = {0};
        DmlOperator::Initialize(kernelInfo, inputIndices, outputIndices, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ELEMENT_WISE_CLIP1_OPERATOR_DESC opDesc = {};
        opDesc.InputTensor = inputDescs.data();
        opDesc.OutputTensor = outputDescs.data();
        // MinMaxDataType will always be equal to inputDataTensorDataType
        // Assigning minMaxDataType to inputDataTensorDataType because this field
        // has to be assigned even if program does not go through below conditional
        // logic for some corner test case
        // Same applies to min and max value.
        opDesc.MinMaxDataType = this->m_inputTensorDescs[0].GetDmlDataType();
        CastToClampedScalarUnion<double>(opDesc.MinMaxDataType, -DBL_MAX, /*out*/&opDesc.Min);
        CastToClampedScalarUnion<double>(opDesc.MinMaxDataType, DBL_MAX, /*out*/&opDesc.Max);

        if (kernelInfo.IsInputValid(1))
        {
            ReadScalarTensorData(kernelInfo.GetConstantInputTensor(1), /*out*/ &opDesc.Min.Bytes, sizeof(opDesc.Min.Bytes));
        }
        if (kernelInfo.IsInputValid(2))
        {
            ReadScalarTensorData(kernelInfo.GetConstantInputTensor(2), /*out*/ &opDesc.Max.Bytes, sizeof(opDesc.Max.Bytes));
        }

        SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_CLIP1, &opDesc}, kernelInfo);
    }
};

// Same operator signature as 11. Only difference is new type support
using DmlOperatorElementwiseClip12 = DmlOperatorElementwiseClip11;
using DmlOperatorElementwiseClip13 = DmlOperatorElementwiseClip11;

class DmlOperatorElementwisePow : public DmlOperator
{
public:
    DmlOperatorElementwisePow(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 2);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        auto constExpTensor = kernelInfo.TryGetConstantCpuInputTensor(1);
        if (constExpTensor && constExpTensor->GetTotalElementCount() == 1)
        {
            std::vector<std::optional<uint32_t>> kernelInputIndices = {0};

            Initialize(kernelInfo, kernelInputIndices, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

            std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
            std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs(); 

            DML_ELEMENT_WISE_CONSTANT_POW_OPERATOR_DESC opDesc = {};
            opDesc.InputTensor = &inputDescs[0];
            opDesc.OutputTensor = &outputDescs[0];
            opDesc.Exponent = static_cast<float>(ReadScalarTensorCastToFloat64(*constExpTensor));

            SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_CONSTANT_POW, &opDesc}, kernelInfo);
        }
        else
        {        
            Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

            std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
            std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs(); 

            DML_ELEMENT_WISE_POW_OPERATOR_DESC opDesc = {};
            opDesc.InputTensor = &inputDescs[0];
            opDesc.ExponentTensor = &inputDescs[1];
            opDesc.OutputTensor = &outputDescs[0];

            SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_POW, &opDesc}, kernelInfo);
        }
    }
};

template <typename TOperatorDesc>
class DmlOperatorElementwiseQLinear : public DmlOperator
{
public:
    DmlOperatorElementwiseQLinear(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 3);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount()  == 1);

        std::vector<uint32_t> outputShape = kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0);
        const uint32_t outputShapeDimCount = gsl::narrow_cast<uint32_t>(outputShape.size());

        Initialize(kernelInfo, std::nullopt, std::nullopt);

        uint32_t axis = 0;

        // If an axis was given explicitly passed (or the default value 1 is set from the schema),
        // then other inputs are broadcasting to the shape of the input data tensor.
        if (kernelInfo.HasAttribute(AttrName::Axis, MLOperatorAttributeType::Int))
        {
            // Avoid validating the axis until later because the axis parameter is ignorable unless
            // broadcasting is actually needed. ONNX opset 13 returns a default value of 1 for the
            // "axis" attribute even when the attribute doesn't actually exist in the model, which
            // would cause a validation failure here.
            const int32_t signedAxis = gsl::narrow_cast<int32_t>(kernelInfo.GetAttribute<int64_t>(AttrName::Axis));
            axis = Dml::HandleNegativeAxis(signedAxis, outputShapeDimCount, /*validateAxis*/ false);
        }

        // Explicitly reshape each of the inputs after the first input (scale and zero point tensors).
        for (uint32_t index = 1, inputCount = gsl::narrow_cast<uint32_t>(m_inputTensorDescs.size()); index < inputCount; ++index)
        {
            auto edgeDesc = kernelInfo.GetInputEdgeDescription(index);
            assert(edgeDesc.edgeType == MLOperatorEdgeType::Tensor);

            // Fix up the the tensor shape by filling with trailing ones. So input[2,3] with axis=0 and scale[2]
            // becomes scale[2,1], so that broadcasting works correctly.
            std::vector<uint32_t> inputTensorShape = kernelInfo.GetTensorShapeDescription().GetInputTensorShape(index);

            // If the input tensor is a 1D vector, then extra massaging is needed to project their
            // 1D vectors back to the full shape for broadcasting along the given axis.
            // The 1D vector should have a length equal to the output tensor's dimension on that axis.
            if (inputTensorShape.size() == 1 && inputTensorShape != outputShape)
            {
                ML_CHECK_VALID_ARGUMENT(axis < outputShapeDimCount);
                uint32_t broadcastAxisLength = outputShape[axis];
                ML_CHECK_VALID_ARGUMENT(inputTensorShape[0] == broadcastAxisLength);
                inputTensorShape.insert(inputTensorShape.begin(), axis, 1);
                inputTensorShape.insert(inputTensorShape.end(), outputShapeDimCount - 1 - axis, 1);
            }
            // For any other shape (scalar/ND), leave it alone, and the TensorDesc constructor
            // will apply broadcasting with standard elementwise alignment.

            m_inputTensorDescs[index] = TensorDesc(
                edgeDesc.tensorDataType,
                gsl::make_span(outputShape),
                gsl::make_span(inputTensorShape),
                TensorAxis::DoNotCoerce,
                TensorAxis::W,
                TensorAxis::RightAligned,
                NchwDimensionCount, // minDimensionCount
                0 // guaranteedBaseOffsetAlignment
            );
        }

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        TOperatorDesc opDesc = {};
        opDesc.InputTensor = &inputDescs[0];
        opDesc.ScaleTensor = &inputDescs[1];
        opDesc.ZeroPointTensor = &inputDescs[2];
        opDesc.OutputTensor = &outputDescs[0];
        
        SetDmlOperatorDesc({ApiTraits::OperatorDescTraits<TOperatorDesc>::Type, &opDesc}, kernelInfo);
    }
};

class DmlOperatorElementwiseIf : public DmlOperator
{
public:
    DmlOperatorElementwiseIf(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 3);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ELEMENT_WISE_IF_OPERATOR_DESC opDesc = {};
        opDesc.ConditionTensor = &inputDescs[0];
        opDesc.ATensor = &inputDescs[1];
        opDesc.BTensor = &inputDescs[2];
        opDesc.OutputTensor = &outputDescs[0];

        SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IF, &opDesc }, kernelInfo);
    }
};

class DmlOperatorElementwiseMod : public DmlOperator
{
public:
    DmlOperatorElementwiseMod(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 2);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        auto fmod = kernelInfo.GetOptionalAttribute<int>(AttrName::Fmod, 0);

        // Note TRUNCATE and FLOOR modulus operator descriptions are identical.
        static_assert(sizeof(DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC) == sizeof(DML_ELEMENT_WISE_MODULUS_FLOOR_OPERATOR_DESC));
        DML_ELEMENT_WISE_MODULUS_TRUNCATE_OPERATOR_DESC opDesc = {};
        opDesc.ATensor = &inputDescs[0];
        opDesc.BTensor = &inputDescs[1];
        opDesc.OutputTensor = &outputDescs[0];

        DML_OPERATOR_TYPE type = fmod ? DML_OPERATOR_ELEMENT_WISE_MODULUS_TRUNCATE : DML_OPERATOR_ELEMENT_WISE_MODULUS_FLOOR;
        SetDmlOperatorDesc({ type, &opDesc}, kernelInfo);
    }
};

class DmlOperatorElementwiseBitShift : public DmlOperator
{
public:
    DmlOperatorElementwiseBitShift(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 2);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        // Note LEFT and RIGHT shift operator descriptions are identical.
        static_assert(sizeof(DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC) == sizeof(DML_ELEMENT_WISE_BIT_SHIFT_RIGHT_OPERATOR_DESC));
        DML_ELEMENT_WISE_BIT_SHIFT_LEFT_OPERATOR_DESC opDesc = {};
        opDesc.ATensor = &inputDescs[0];
        opDesc.BTensor = &inputDescs[1];
        opDesc.OutputTensor = &outputDescs[0];

        std::string mode = kernelInfo.GetOptionalAttribute<std::string>(AttrName::Direction, "");
        ML_CHECK_VALID_ARGUMENT(mode == "LEFT" || mode == "RIGHT");

        DML_OPERATOR_TYPE type = (mode == "LEFT") ? DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_LEFT : DML_OPERATOR_ELEMENT_WISE_BIT_SHIFT_RIGHT;
        SetDmlOperatorDesc({ type, &opDesc}, kernelInfo);
    }
};

class DmlOperatorElementwiseIsInf : public DmlOperator
{
public:
    DmlOperatorElementwiseIsInf(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        auto detectPositive = kernelInfo.GetOptionalAttribute<int>(AttrName::DetectPositive, 1);
        auto detectNegative = kernelInfo.GetOptionalAttribute<int>(AttrName::DetectNegative, 1);

        DML_ELEMENT_WISE_IS_INFINITY_OPERATOR_DESC opDesc = {};
        opDesc.InputTensor  = inputDescs.data();
        opDesc.OutputTensor = outputDescs.data();
        opDesc.InfinityMode = (detectPositive == detectNegative) ? DML_IS_INFINITY_MODE_EITHER
                            :  detectPositive                    ? DML_IS_INFINITY_MODE_POSITIVE
                            :                                      DML_IS_INFINITY_MODE_NEGATIVE;

        SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_IS_INFINITY, &opDesc}, kernelInfo);
    }
};

class DmlOperatorElementwiseRound : public DmlOperator
{
public:
    DmlOperatorElementwiseRound(const MLOperatorKernelCreationContext& kernelInfo) : DmlOperator(kernelInfo)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInfo.GetOutputCount() == 1);

        Initialize(kernelInfo, std::nullopt, std::nullopt, kernelInfo.GetTensorShapeDescription().GetOutputTensorShape(0));

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        DML_ELEMENT_WISE_ROUND_OPERATOR_DESC opDesc = {};
        opDesc.InputTensor = inputDescs.data();
        opDesc.OutputTensor = outputDescs.data();
        opDesc.RoundingMode = DML_ROUNDING_MODE_HALVES_TO_NEAREST_EVEN;

        SetDmlOperatorDesc({ DML_OPERATOR_ELEMENT_WISE_ROUND, &opDesc}, kernelInfo);
    }
};

// Unary operators:
DML_OP_DEFINE_CREATION_FUNCTION(Sqrt,             DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_SQRT_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Reciprocal,       DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_RECIP_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Cos,              DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_COS_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Sin,              DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_SIN_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Tan,              DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_TAN_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Acos,             DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_ACOS_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Asin,             DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_ASIN_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Atan,             DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_ATAN_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Exp,              DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_EXP_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Log,              DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_LOG_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Abs,              DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_ABS_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Ceil,             DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_CEIL_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Floor,            DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_FLOOR_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Not,              DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_LOGICAL_NOT_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Sign,             DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_SIGN_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(IsNaN,            DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_IS_NAN_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Sinh,             DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_SINH_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Cosh,             DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_COSH_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Asinh,            DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_ASINH_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Acosh,            DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_ACOSH_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Atanh,            DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_ATANH_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Erf,              DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_ERF_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(BitwiseNot,       DmlOperatorElementwiseUnary<DML_ELEMENT_WISE_BIT_NOT_OPERATOR_DESC>);

// Binary operators:
DML_OP_DEFINE_CREATION_FUNCTION(Greater,          DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Less,             DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(GreaterOrEqual,   DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_LOGICAL_GREATER_THAN_OR_EQUAL_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(LessOrEqual,      DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_LOGICAL_LESS_THAN_OR_EQUAL_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Equal,            DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_LOGICAL_EQUALS_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(And,              DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_LOGICAL_AND_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Or,               DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_LOGICAL_OR_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Xor,              DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_LOGICAL_XOR_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Add,              DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Sub,              DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_SUBTRACT_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Mul,              DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Div,              DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_DIVIDE_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(BitwiseAnd,       DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(BitwiseOr,        DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(BitwiseXor,       DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_BIT_XOR_OPERATOR_DESC>);

// Binary operators that support >2 inputs:
DML_OP_DEFINE_CREATION_FUNCTION(Sum,              DmlOperatorElementwiseBinaryLoop<DML_ELEMENT_WISE_ADD_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Min,              DmlOperatorElementwiseBinaryLoop<DML_ELEMENT_WISE_MIN_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Max,              DmlOperatorElementwiseBinaryLoop<DML_ELEMENT_WISE_MAX_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Mean,             DmlOperatorElementwiseMean);

// Operators with extra attributes:
DML_OP_DEFINE_CREATION_FUNCTION(Clip7,            DmlOperatorElementwiseClip7);
DML_OP_DEFINE_CREATION_FUNCTION(Clip11,           DmlOperatorElementwiseClip11);
DML_OP_DEFINE_CREATION_FUNCTION(Clip12,           DmlOperatorElementwiseClip12);
DML_OP_DEFINE_CREATION_FUNCTION(Clip13,           DmlOperatorElementwiseClip13);
DML_OP_DEFINE_CREATION_FUNCTION(Pow,              DmlOperatorElementwisePow);
DML_OP_DEFINE_CREATION_FUNCTION(QuantizeLinear,   DmlOperatorElementwiseQLinear<DML_ELEMENT_WISE_QUANTIZE_LINEAR_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(DequantizeLinear, DmlOperatorElementwiseQLinear<DML_ELEMENT_WISE_DEQUANTIZE_LINEAR_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(Where,            DmlOperatorElementwiseIf);
DML_OP_DEFINE_CREATION_FUNCTION(Mod,              DmlOperatorElementwiseMod);
DML_OP_DEFINE_CREATION_FUNCTION(BitShift,         DmlOperatorElementwiseBitShift);
DML_OP_DEFINE_CREATION_FUNCTION(IsInf,            DmlOperatorElementwiseIsInf);
DML_OP_DEFINE_CREATION_FUNCTION(Round,            DmlOperatorElementwiseRound);

// Fused operators:
DML_OP_DEFINE_CREATION_FUNCTION(DmlFusedAdd,         DmlOperatorElementwiseBinary<DML_ELEMENT_WISE_ADD1_OPERATOR_DESC>);
DML_OP_DEFINE_CREATION_FUNCTION(DmlFusedSum,         DmlOperatorElementwiseBinaryLoop<DML_ELEMENT_WISE_ADD1_OPERATOR_DESC>);

} // namespace Dml

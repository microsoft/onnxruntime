// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include <corecrt_math_defines.h>
#include "onnx/onnx-ml.pb.h"
#include "core/framework/random_seed.h"
#include "core/framework/random_generator.h"
#include "core/providers/dml/DmlExecutionProvider/src/MLOperatorAuthorImpl.h"
#include "core/providers/dml/DmlExecutionProvider/src/ExecutionProvider.h"

namespace Dml
{

class DmlOperatorRandomNormalLike : public DmlOperator
{
public:
    DmlOperatorRandomNormalLike(const MLOperatorKernelCreationContext& kernelCreationContext)
    :   DmlOperator(kernelCreationContext)
    {
        const std::vector<std::optional<uint32_t>> kernelOutputIndices = {std::optional<uint32_t>(0), std::nullopt};

        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelCreationContext.GetOutputCount() == 1);
        DmlOperator::Initialize(kernelCreationContext, std::nullopt, kernelOutputIndices);

        ML_CHECK_VALID_ARGUMENT(m_inputTensorDescs.size() == 1);
        ML_CHECK_VALID_ARGUMENT(m_outputTensorDescs.size() == 2);

        std::array<uint32_t, 4> stateTensorShape = {1, 1, 1, 6};
        m_inputTensorDescs[0] = TensorDesc(DML_TENSOR_DATA_TYPE_UINT32, stateTensorShape);
        m_outputTensorDescs[1] = TensorDesc(DML_TENSOR_DATA_TYPE_UINT32, stateTensorShape);

        std::vector<DML_TENSOR_DESC> inputDescs = GetDmlInputDescs();
        std::vector<DML_TENSOR_DESC> outputDescs = GetDmlOutputDescs();

        std::vector<DimensionType> inputShape = kernelCreationContext.GetTensorShapeDescription().GetInputTensorShape(0);
        m_elementCount = ComputeElementCountFromDimensions(inputShape);

        const float mean = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Mean, 0.0f);
        const float scale = kernelCreationContext.GetOptionalAttribute<float>(AttrName::Scale, 1.0f);
        m_hasSeed = kernelCreationContext.HasAttribute(AttrName::Seed, MLOperatorAttributeType::Float);

        if (m_hasSeed)
        {
            const float seed = kernelCreationContext.GetAttribute<float>(AttrName::Seed);
            m_generator.emplace(static_cast<uint64_t>(seed));
        }

        // Round up to the nearest even number since the normal distribution
        // algorithm needs to work with 2 samples at a time
        uint32_t evenElementCount = m_elementCount + (m_elementCount % 2);

        // 1. Generate the philox random bits
        std::array<uint32_t, 4> generatorOutputShape = {1, 1, evenElementCount / 2, 2};
        TensorDesc generatorOutputTensorDesc = TensorDesc(DML_TENSOR_DATA_TYPE_UINT32, generatorOutputShape);
        DML_TENSOR_DESC generatorOutputDmlTensorDesc = generatorOutputTensorDesc.GetDmlDesc();

        DML_RANDOM_GENERATOR_OPERATOR_DESC randomGeneratorDesc{};
        randomGeneratorDesc.InputStateTensor = &inputDescs[0];
        randomGeneratorDesc.OutputTensor = &generatorOutputDmlTensorDesc;
        randomGeneratorDesc.OutputStateTensor = &inputDescs[0];

        randomGeneratorDesc.Type = DML_RANDOM_GENERATOR_TYPE_PHILOX_4X32_10;
        DML_OPERATOR_DESC dmlRandomGeneratorDesc = { DML_OPERATOR_RANDOM_GENERATOR, &randomGeneratorDesc };

        // 2. Convert the random bits from uint32 to float32
        std::array<uint32_t, 4> scalarShape = {1, 1, 1, 1};
        TensorDesc scalarTensorDesc = TensorDesc(DML_TENSOR_DATA_TYPE_UINT32, scalarShape);
        DML_TENSOR_DESC scalarDmlTensorDesc = scalarTensorDesc.GetDmlDesc();

        constexpr uint32_t signAndExponentValue = ((1 << (8 - 1)) - 1) << 23;
        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC signAndExponentDesc{};
        signAndExponentDesc.OutputTensor = &scalarDmlTensorDesc;
        signAndExponentDesc.ValueDataType = DML_TENSOR_DATA_TYPE_UINT32;
        signAndExponentDesc.Value.UInt32 = signAndExponentValue;
        DML_OPERATOR_DESC dmlSignAndExponentDesc = { DML_OPERATOR_FILL_VALUE_CONSTANT, &signAndExponentDesc };

        constexpr uint32_t mantissaMaskValue = (1 << 23) - 1;
        DML_FILL_VALUE_CONSTANT_OPERATOR_DESC mantissaMaskDesc{};
        mantissaMaskDesc.OutputTensor = &scalarDmlTensorDesc;
        mantissaMaskDesc.ValueDataType = DML_TENSOR_DATA_TYPE_UINT32;
        mantissaMaskDesc.Value.UInt32 = mantissaMaskValue;
        DML_OPERATOR_DESC dmlMantissaMaskDesc = { DML_OPERATOR_FILL_VALUE_CONSTANT, &mantissaMaskDesc };

        std::array<uint32_t, 4> broadcastedScalarStrides = {0, 0, 0, 0};
        TensorDesc broadcastedScalarTensorDesc = TensorDesc(DML_TENSOR_DATA_TYPE_UINT32, generatorOutputShape, broadcastedScalarStrides);
        DML_TENSOR_DESC broadcastedScalarDmlTensorDesc = broadcastedScalarTensorDesc.GetDmlDesc();

        DML_ELEMENT_WISE_BIT_AND_OPERATOR_DESC bitAndDesc{};
        bitAndDesc.ATensor = &generatorOutputDmlTensorDesc;
        bitAndDesc.BTensor = &broadcastedScalarDmlTensorDesc;
        bitAndDesc.OutputTensor = &generatorOutputDmlTensorDesc;
        DML_OPERATOR_DESC dmlBitAndDesc = { DML_OPERATOR_ELEMENT_WISE_BIT_AND, &bitAndDesc };

        DML_ELEMENT_WISE_BIT_OR_OPERATOR_DESC bitOrDesc{};
        bitOrDesc.ATensor = &broadcastedScalarDmlTensorDesc;
        bitOrDesc.BTensor = &generatorOutputDmlTensorDesc;
        bitOrDesc.OutputTensor = &generatorOutputDmlTensorDesc;
        DML_OPERATOR_DESC dmlBitOrDesc = { DML_OPERATOR_ELEMENT_WISE_BIT_OR, &bitOrDesc };

        TensorDesc float32GeneratorOutputTensorDesc = TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, generatorOutputShape);
        DML_TENSOR_DESC float32GeneratorOutputDmlTensorDesc = float32GeneratorOutputTensorDesc.GetDmlDesc();

        DML_SCALE_BIAS minusScaleBias {1.0f, -1.0f};
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC minusDesc{};
        minusDesc.InputTensor = &float32GeneratorOutputDmlTensorDesc;
        minusDesc.OutputTensor = &float32GeneratorOutputDmlTensorDesc;
        minusDesc.ScaleBias = &minusScaleBias;
        DML_OPERATOR_DESC dmlMinusDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &minusDesc };

        // 3. Split the random bits in 2
        // The BoxMuller normal distribution algorithm handles elements 2 by 2,
        // so we put them on different columns and split the columns in 2
        std::array<uint32_t, 4> splitOutputShape = {1, 1, evenElementCount / 2, 1};
        TensorDesc splitOutputTensorDesc = TensorDesc(DML_TENSOR_DATA_TYPE_FLOAT32, splitOutputShape);
        DML_TENSOR_DESC splitOutputDmlTensorDesc = splitOutputTensorDesc.GetDmlDesc();
        std::array<DML_TENSOR_DESC, 2> splitOutputDmlTensorDescs = {splitOutputDmlTensorDesc, splitOutputDmlTensorDesc};

        DML_SPLIT_OPERATOR_DESC splitDesc{};
        splitDesc.InputTensor = &float32GeneratorOutputDmlTensorDesc;
        splitDesc.OutputTensors = splitOutputDmlTensorDescs.data();
        splitDesc.OutputCount = gsl::narrow_cast<uint32_t>(splitOutputDmlTensorDescs.size());
        splitDesc.Axis = gsl::narrow_cast<uint32_t>(splitOutputShape.size() - 1);
        DML_OPERATOR_DESC dmlSplitDesc = { DML_OPERATOR_SPLIT, &splitDesc };

        // 2. Apply the Clip from the Box-Muller algorithm
        DML_ELEMENT_WISE_CLIP_OPERATOR_DESC clipDesc{};
        clipDesc.InputTensor = &splitOutputDmlTensorDesc;
        clipDesc.OutputTensor = &splitOutputDmlTensorDesc;
        clipDesc.Min = 1.0e-7f;
        clipDesc.Max = std::numeric_limits<float>::max();
        DML_OPERATOR_DESC dmlClipDesc = { DML_OPERATOR_ELEMENT_WISE_CLIP, &clipDesc };

        // 3. Apply the Log from the Box-Muller algorithm
        DML_ELEMENT_WISE_LOG_OPERATOR_DESC logDesc{};
        logDesc.InputTensor = &splitOutputDmlTensorDesc;
        logDesc.OutputTensor = &splitOutputDmlTensorDesc;
        DML_OPERATOR_DESC dmlLogDesc = { DML_OPERATOR_ELEMENT_WISE_LOG, &logDesc };

        // 4. Apply the Sqrt from the Box-Muller algorithm
        DML_SCALE_BIAS sqrtScaleBias {-2.0f, 0.0f};
        DML_ELEMENT_WISE_SQRT_OPERATOR_DESC sqrtDesc{};
        sqrtDesc.InputTensor = &splitOutputDmlTensorDesc;
        sqrtDesc.OutputTensor = &splitOutputDmlTensorDesc;
        sqrtDesc.ScaleBias = &sqrtScaleBias;
        DML_OPERATOR_DESC dmlSqrtDesc = { DML_OPERATOR_ELEMENT_WISE_SQRT, &sqrtDesc };

        // 5. Apply the Sin from the Box-Muller algorithm
        DML_SCALE_BIAS sinCosScaleBias {2.0f * gsl::narrow_cast<float>(M_PI), 0.0f};
        DML_ELEMENT_WISE_SIN_OPERATOR_DESC sinDesc{};
        sinDesc.InputTensor = &splitOutputDmlTensorDesc;
        sinDesc.OutputTensor = &splitOutputDmlTensorDesc;
        sinDesc.ScaleBias = &sinCosScaleBias;
        DML_OPERATOR_DESC dmlSinDesc = { DML_OPERATOR_ELEMENT_WISE_SIN, &sinDesc };

        // 6. APply the Cos from the Box-Muller algorithm
        DML_ELEMENT_WISE_COS_OPERATOR_DESC cosDesc{};
        cosDesc.InputTensor = &splitOutputDmlTensorDesc;
        cosDesc.OutputTensor = &splitOutputDmlTensorDesc;
        cosDesc.ScaleBias = &sinCosScaleBias;
        DML_OPERATOR_DESC dmlCosDesc = { DML_OPERATOR_ELEMENT_WISE_COS, &cosDesc };

        // 7. Multiply the result of Sin and Cos with the result of Sqrt
        DML_ELEMENT_WISE_MULTIPLY_OPERATOR_DESC multiplyDesc{};
        multiplyDesc.ATensor = &splitOutputDmlTensorDesc;
        multiplyDesc.BTensor = &splitOutputDmlTensorDesc;
        multiplyDesc.OutputTensor = &splitOutputDmlTensorDesc;
        DML_OPERATOR_DESC dmlMultiplyDesc = { DML_OPERATOR_ELEMENT_WISE_MULTIPLY, &multiplyDesc };

        // 8. Join the result of Sqrt*Sin and Sqrt*Cos together
        DML_JOIN_OPERATOR_DESC joinDesc{};
        joinDesc.InputTensors = splitOutputDmlTensorDescs.data();
        joinDesc.OutputTensor = &float32GeneratorOutputDmlTensorDesc;
        joinDesc.InputCount = gsl::narrow_cast<uint32_t>(splitOutputDmlTensorDescs.size());
        joinDesc.Axis = gsl::narrow_cast<uint32_t>(splitOutputShape.size() - 1);
        DML_OPERATOR_DESC dmlJoinDesc = { DML_OPERATOR_JOIN, &joinDesc };

        // 9. Apply the scale and bias to the float result
        DML_SCALE_BIAS scaleMean {scale, mean};
        DML_ELEMENT_WISE_IDENTITY_OPERATOR_DESC scaleMeanDesc{};
        scaleMeanDesc.InputTensor = &float32GeneratorOutputDmlTensorDesc;
        scaleMeanDesc.OutputTensor = &float32GeneratorOutputDmlTensorDesc;
        scaleMeanDesc.ScaleBias = &scaleMean;
        DML_OPERATOR_DESC dmlScaleMeanDesc = { DML_OPERATOR_ELEMENT_WISE_IDENTITY, &scaleMeanDesc };

        // 10. Cast to the desired output type if needed
        TensorDesc outputTensorDesc = TensorDesc(m_outputTensorDescs[0].GetDmlDataType(), generatorOutputShape);
        DML_TENSOR_DESC outputDmlTensorDesc = outputTensorDesc.GetDmlDesc();

        const bool needCast = m_outputTensorDescs[0].GetDmlDataType() != DML_TENSOR_DATA_TYPE_FLOAT32;
        DML_CAST_OPERATOR_DESC castDesc{};
        if (needCast)
        {
            castDesc.InputTensor = &float32GeneratorOutputDmlTensorDesc;
            castDesc.OutputTensor = &outputDmlTensorDesc;
        }
        DML_OPERATOR_DESC dmlCastDesc = { DML_OPERATOR_CAST, &castDesc };

        enum NodeIndex
        {
            randomGeneratorNodeIndex,
            mantissaMaskNodeIndex,
            bitAndNodeIndex,
            signAndExponentNodeIndex,
            bitOrNodeIndex,
            minusNodeIndex,
            splitNodeIndex,
            clipNodeIndex,
            logNodeIndex,
            sqrtNodeIndex,
            sinNodeIndex,
            multiplySinNodeIndex,
            cosNodeIndex,
            multiplyCosNodeIndex,
            joinNodeIndex,
            scaleMeanNodeIndex,
            nodeCount,
        };

        // Construct the graph
        std::vector<const DML_OPERATOR_DESC*> opDescs(nodeCount);
        opDescs[randomGeneratorNodeIndex] = &dmlRandomGeneratorDesc;
        opDescs[mantissaMaskNodeIndex] = &dmlMantissaMaskDesc;
        opDescs[bitAndNodeIndex] = &dmlBitAndDesc;
        opDescs[signAndExponentNodeIndex] = &dmlSignAndExponentDesc;
        opDescs[bitOrNodeIndex] = &dmlBitOrDesc;
        opDescs[minusNodeIndex] = &dmlMinusDesc;
        opDescs[splitNodeIndex] = &dmlSplitDesc;
        opDescs[clipNodeIndex] = &dmlClipDesc;
        opDescs[logNodeIndex] = &dmlLogDesc;
        opDescs[sqrtNodeIndex] = &dmlSqrtDesc;
        opDescs[sinNodeIndex] = &dmlSinDesc;
        opDescs[multiplySinNodeIndex] = &dmlMultiplyDesc;
        opDescs[cosNodeIndex] = &dmlCosDesc;
        opDescs[multiplyCosNodeIndex] = &dmlMultiplyDesc;
        opDescs[joinNodeIndex] = &dmlJoinDesc;
        opDescs[scaleMeanNodeIndex] = &dmlScaleMeanDesc;

        uint32_t optionalNodeIndex = nodeCount;
        uint32_t castNodeIndex;
        if (needCast)
        {
            opDescs.push_back(&dmlCastDesc);
            castNodeIndex = optionalNodeIndex++;
        }

        std::vector<DML_INPUT_GRAPH_EDGE_DESC> inputEdges;
        inputEdges.reserve(1);

        std::vector<DML_INTERMEDIATE_GRAPH_EDGE_DESC> intermediateEdges;
        intermediateEdges.reserve(17);

        std::vector<DML_OUTPUT_GRAPH_EDGE_DESC> outputEdges;
        outputEdges.reserve(2);

        DML_INPUT_GRAPH_EDGE_DESC inputToRandomGeneratorEdge{};
        inputToRandomGeneratorEdge.GraphInputIndex = 0;
        inputToRandomGeneratorEdge.ToNodeIndex = randomGeneratorNodeIndex;
        inputToRandomGeneratorEdge.ToNodeInputIndex = 0;
        inputEdges.push_back(inputToRandomGeneratorEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC randomGeneratorToBitAndEdge{};
        randomGeneratorToBitAndEdge.FromNodeIndex = randomGeneratorNodeIndex;
        randomGeneratorToBitAndEdge.FromNodeOutputIndex = 0;
        randomGeneratorToBitAndEdge.ToNodeIndex = bitAndNodeIndex;
        randomGeneratorToBitAndEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(randomGeneratorToBitAndEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC mantissaMaskToBitAndEdge{};
        mantissaMaskToBitAndEdge.FromNodeIndex = mantissaMaskNodeIndex;
        mantissaMaskToBitAndEdge.FromNodeOutputIndex = 0;
        mantissaMaskToBitAndEdge.ToNodeIndex = bitAndNodeIndex;
        mantissaMaskToBitAndEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(mantissaMaskToBitAndEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC signAndExponentToBitOrEdge{};
        signAndExponentToBitOrEdge.FromNodeIndex = signAndExponentNodeIndex;
        signAndExponentToBitOrEdge.FromNodeOutputIndex = 0;
        signAndExponentToBitOrEdge.ToNodeIndex = bitOrNodeIndex;
        signAndExponentToBitOrEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(signAndExponentToBitOrEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC bitAndToBitOrEdge{};
        bitAndToBitOrEdge.FromNodeIndex = bitAndNodeIndex;
        bitAndToBitOrEdge.FromNodeOutputIndex = 0;
        bitAndToBitOrEdge.ToNodeIndex = bitOrNodeIndex;
        bitAndToBitOrEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(bitAndToBitOrEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC bitOrToMinusEdge{};
        bitOrToMinusEdge.FromNodeIndex = bitOrNodeIndex;
        bitOrToMinusEdge.FromNodeOutputIndex = 0;
        bitOrToMinusEdge.ToNodeIndex = minusNodeIndex;
        bitOrToMinusEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(bitOrToMinusEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC minusToSplitEdge{};
        minusToSplitEdge.FromNodeIndex = minusNodeIndex;
        minusToSplitEdge.FromNodeOutputIndex = 0;
        minusToSplitEdge.ToNodeIndex = splitNodeIndex;
        minusToSplitEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(minusToSplitEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC splitToClipEdge{};
        splitToClipEdge.FromNodeIndex = splitNodeIndex;
        splitToClipEdge.FromNodeOutputIndex = 0;
        splitToClipEdge.ToNodeIndex = clipNodeIndex;
        splitToClipEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(splitToClipEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC clipToLogEdge{};
        clipToLogEdge.FromNodeIndex = clipNodeIndex;
        clipToLogEdge.FromNodeOutputIndex = 0;
        clipToLogEdge.ToNodeIndex = logNodeIndex;
        clipToLogEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(clipToLogEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC logToSqrtEdge{};
        logToSqrtEdge.FromNodeIndex = logNodeIndex;
        logToSqrtEdge.FromNodeOutputIndex = 0;
        logToSqrtEdge.ToNodeIndex = sqrtNodeIndex;
        logToSqrtEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(logToSqrtEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC splitToSinEdge{};
        splitToSinEdge.FromNodeIndex = splitNodeIndex;
        splitToSinEdge.FromNodeOutputIndex = 1;
        splitToSinEdge.ToNodeIndex = sinNodeIndex;
        splitToSinEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(splitToSinEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC splitToCosEdge{};
        splitToCosEdge.FromNodeIndex = splitNodeIndex;
        splitToCosEdge.FromNodeOutputIndex = 1;
        splitToCosEdge.ToNodeIndex = cosNodeIndex;
        splitToCosEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(splitToCosEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC sinToMultiplySinEdge{};
        sinToMultiplySinEdge.FromNodeIndex = sinNodeIndex;
        sinToMultiplySinEdge.FromNodeOutputIndex = 0;
        sinToMultiplySinEdge.ToNodeIndex = multiplySinNodeIndex;
        sinToMultiplySinEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(sinToMultiplySinEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC sqrtToMultiplySinEdge{};
        sqrtToMultiplySinEdge.FromNodeIndex = sqrtNodeIndex;
        sqrtToMultiplySinEdge.FromNodeOutputIndex = 0;
        sqrtToMultiplySinEdge.ToNodeIndex = multiplySinNodeIndex;
        sqrtToMultiplySinEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(sqrtToMultiplySinEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC cosToMultiplyCosEdge{};
        cosToMultiplyCosEdge.FromNodeIndex = cosNodeIndex;
        cosToMultiplyCosEdge.FromNodeOutputIndex = 0;
        cosToMultiplyCosEdge.ToNodeIndex = multiplyCosNodeIndex;
        cosToMultiplyCosEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(cosToMultiplyCosEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC sqrtToMultiplyCosEdge{};
        sqrtToMultiplyCosEdge.FromNodeIndex = sqrtNodeIndex;
        sqrtToMultiplyCosEdge.FromNodeOutputIndex = 0;
        sqrtToMultiplyCosEdge.ToNodeIndex = multiplyCosNodeIndex;
        sqrtToMultiplyCosEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(sqrtToMultiplyCosEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC multiplySinToJoinEdge{};
        multiplySinToJoinEdge.FromNodeIndex = multiplySinNodeIndex;
        multiplySinToJoinEdge.FromNodeOutputIndex = 0;
        multiplySinToJoinEdge.ToNodeIndex = joinNodeIndex;
        multiplySinToJoinEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(multiplySinToJoinEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC multiplyCosToJoinEdge{};
        multiplyCosToJoinEdge.FromNodeIndex = multiplyCosNodeIndex;
        multiplyCosToJoinEdge.FromNodeOutputIndex = 0;
        multiplyCosToJoinEdge.ToNodeIndex = joinNodeIndex;
        multiplyCosToJoinEdge.ToNodeInputIndex = 1;
        intermediateEdges.push_back(multiplyCosToJoinEdge);

        DML_INTERMEDIATE_GRAPH_EDGE_DESC joinToScaleMeanEdge{};
        joinToScaleMeanEdge.FromNodeIndex = joinNodeIndex;
        joinToScaleMeanEdge.FromNodeOutputIndex = 0;
        joinToScaleMeanEdge.ToNodeIndex = scaleMeanNodeIndex;
        joinToScaleMeanEdge.ToNodeInputIndex = 0;
        intermediateEdges.push_back(joinToScaleMeanEdge);

        if (needCast)
        {
            DML_INTERMEDIATE_GRAPH_EDGE_DESC scaleMeanToCastEdge{};
            scaleMeanToCastEdge.FromNodeIndex = scaleMeanNodeIndex;
            scaleMeanToCastEdge.FromNodeOutputIndex = 0;
            scaleMeanToCastEdge.ToNodeIndex = castNodeIndex;
            scaleMeanToCastEdge.ToNodeInputIndex = 0;
            intermediateEdges.push_back(scaleMeanToCastEdge);

            DML_OUTPUT_GRAPH_EDGE_DESC castToOutputEdge{};
            castToOutputEdge.FromNodeIndex = castNodeIndex;
            castToOutputEdge.FromNodeOutputIndex = 0;
            castToOutputEdge.GraphOutputIndex = 0;
            outputEdges.push_back(castToOutputEdge);
        }
        else
        {
            DML_OUTPUT_GRAPH_EDGE_DESC scaleMeanToOutputEdge{};
            scaleMeanToOutputEdge.FromNodeIndex = scaleMeanNodeIndex;
            scaleMeanToOutputEdge.FromNodeOutputIndex = 0;
            scaleMeanToOutputEdge.GraphOutputIndex = 0;
            outputEdges.push_back(scaleMeanToOutputEdge);
        }

        DML_OUTPUT_GRAPH_EDGE_DESC randomGeneratorStateToOutputEdge{};
        randomGeneratorStateToOutputEdge.FromNodeIndex = randomGeneratorNodeIndex;
        randomGeneratorStateToOutputEdge.FromNodeOutputIndex = 1;
        randomGeneratorStateToOutputEdge.GraphOutputIndex = 1;
        outputEdges.push_back(randomGeneratorStateToOutputEdge);

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

        onnxruntime::TensorShape stateShape({1, 1, 1, 6});
        ExecutionProviderImpl* executionProvider = static_cast<ExecutionProviderImpl*>(m_executionProvider.Get());

        m_stateTensorCpu = onnxruntime::Tensor(onnxruntime::DataTypeImpl::GetType<uint32_t>(), stateShape, executionProvider->GetCpuInputAllocator());
        m_stateTensorCpuWrapper = wil::MakeOrThrow<Windows::AI::MachineLearning::Adapter::TensorWrapper>(
            &m_stateTensorCpu,
            false,
            executionProvider,
            true);

        m_stateTensorDml = onnxruntime::Tensor(onnxruntime::DataTypeImpl::GetType<uint32_t>(), stateShape, executionProvider->GetGpuAllocator());
        m_stateTensorDmlWrapper = wil::MakeOrThrow<Windows::AI::MachineLearning::Adapter::TensorWrapper>(
            &m_stateTensorDml,
            true,
            executionProvider,
            true);

        // If a seed was given, it will be the same for all executions of this operator and therefore we only need to upload
        // the key once from the CPU
        if (m_hasSeed)
        {
            UpdateState();
        }
    }

    void Compute(const MLOperatorKernelContext& kernelContext) final
    {
        if (!m_hasSeed)
        {
            UpdateState();
        }

        std::array<IMLOperatorTensor*, 1> inputTensors = {m_stateTensorDmlWrapper.Get()};
        std::array<IMLOperatorTensor*, 2> outputTensors = {GetOutputTensorsForExecute(kernelContext)[0], m_stateTensorDmlWrapper.Get()};

        ORT_THROW_IF_FAILED(m_executionProvider->ExecuteOperator(
            m_compiledOperator.Get(),
            m_persistentResourceBinding ? &*m_persistentResourceBinding : nullptr,
            gsl::make_span(inputTensors),
            gsl::make_span(outputTensors)));
    }

private:
    mutable std::optional<onnxruntime::PhiloxGenerator> m_generator;
    bool m_hasSeed;
    uint32_t m_elementCount;
    onnxruntime::Tensor m_stateTensorCpu;
    Microsoft::WRL::ComPtr<IMLOperatorTensor> m_stateTensorCpuWrapper;
    onnxruntime::Tensor m_stateTensorDml;
    Microsoft::WRL::ComPtr<IMLOperatorTensor> m_stateTensorDmlWrapper;

    onnxruntime::PhiloxGenerator& GetPhiloxGenerator() const
    {
        return m_generator.has_value() ? *m_generator : onnxruntime::PhiloxGenerator::Default();
    }

    void UpdateState()
    {
        auto& generator = GetPhiloxGenerator();
        auto [seed, offset] = generator.NextPhiloxSeeds(m_elementCount);

        uint32_t* cpuState = m_stateTensorCpu.MutableData<uint32_t>();
        cpuState[0] = gsl::narrow_cast<uint32_t>(offset);
        cpuState[1] = gsl::narrow_cast<uint32_t>(offset >> 32);
        cpuState[2] = 0;
        cpuState[3] = 0;
        cpuState[4] = gsl::narrow_cast<uint32_t>(seed);
        cpuState[5] = gsl::narrow_cast<uint32_t>(seed >> 32);

        ORT_THROW_IF_FAILED(m_executionProvider->CopyTensor(
            m_stateTensorDmlWrapper.Get(),
            m_stateTensorCpuWrapper.Get()));
    }
};

DML_OP_DEFINE_CREATION_FUNCTION(RandomNormalLike, DmlOperatorRandomNormalLike);

} // namespace Dml

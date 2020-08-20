// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "OperatorHelper.h"

namespace OperatorHelper
{
    bool ContainsEmptyDimensions(gsl::span<const DimensionType> dimensions)
    {
        return std::find(dimensions.begin(), dimensions.end(), 0) != dimensions.end();
    }

    // Convert any negative axis into an absolute axis relative to the back end.
    // So given 3 dimensions, a -1 refers to axis 2, and -3 to axis 0.
    uint32_t HandleNegativeAxis(int32_t signedOnnxAxis, uint32_t dimCount)
    {
        if (signedOnnxAxis < 0)
        {
            signedOnnxAxis += dimCount;
        }
        uint32_t absoluteAxis = gsl::narrow_cast<uint32_t>(signedOnnxAxis);
        ML_CHECK_VALID_ARGUMENT(absoluteAxis < dimCount);
        return absoluteAxis;
    }

    void HandleNegativeAxes(gsl::span<int32_t> onnxAxes, uint32_t dimCount)
    {
        for (int32_t& axis : onnxAxes)
        {
            axis = HandleNegativeAxis(axis, dimCount);
        }
    }

    void ReadCpuLocalTensorIntoInt32(
        const MLOperatorTensor& tensor,
        std::vector<int32_t>& result
        )
    {
        result.clear();
        ML_CHECK_VALID_ARGUMENT(tensor.IsCpuData(), "Tensor must be CPU Tensor.");

        const std::vector<uint32_t>& tensorDimensions = tensor.GetShape();
        const uint32_t elementCount = ComputeElementCountFromDimensions(tensorDimensions);

        switch (tensor.GetTensorDataType())
        {
        case MLOperatorTensorDataType::Int32:
            {
                const int32_t* data = tensor.GetData<int32_t>();
                result.assign(data, data + elementCount);
            }
            break;

        case MLOperatorTensorDataType::Int64:
            {
                const int64_t* data = tensor.GetData<int64_t>();
                result.reserve(elementCount);

                // Use clamped cast rather than static_cast/narrow_cast,
                // because it's not uncommon for a model to specify a
                // 64-bit INTMAX constant as a sentinel value to mean
                // the largest possible value (even though the actual
                // dimension values come nowhere close to that, far
                // less than 32-bit INTMAX).
                for (auto d : gsl::make_span(data, data + elementCount))
                {
                    result.push_back(clamp_cast<int32_t>(d));
                }
            }
            break;

        default:
            ML_INVALID_ARGUMENT("Expecting CPU local tensor of type int32 or int64.");
            break;
        }
    }

    void ReadCpuLocalTensorIntoFloat32(
        const MLOperatorTensor& tensor,
        std::vector<float>& result
        )
    {
        result.clear();
        ML_CHECK_VALID_ARGUMENT(tensor.IsCpuData(), "Tensor must be CPU Tensor.");

        const std::vector<uint32_t>& tensorDimensions = tensor.GetShape();
        const uint32_t elementCount = ComputeElementCountFromDimensions(tensorDimensions);

        switch (tensor.GetTensorDataType())
        {
        case MLOperatorTensorDataType::Float:
            {
                const float* data = tensor.GetData<float>();
                result.assign(data, data + elementCount);
            }
            break;

        default:
            ML_INVALID_ARGUMENT("Expecting CPU local tensor of type float32.");
            break;
        }
    }

    void DowncastDimensions(gsl::span<const int64_t> inputDimensions, std::vector<DimensionType>& outputDimensions)
    {
        outputDimensions.reserve(inputDimensions.size());
        outputDimensions.clear();

        for (int64_t dim : inputDimensions)
        {
            outputDimensions.push_back(gsl::narrow_cast<uint32_t>(std::clamp<int64_t>(dim, INT32_MIN, INT32_MAX)));
        }
    }

    int64_t CastToInt64(MLOperatorTensorDataType tensorDataType, const void* p)
    {
        switch (tensorDataType)
        {
        case MLOperatorTensorDataType::Float:      return static_cast<int64_t>(*reinterpret_cast<const float*>(p));
        case MLOperatorTensorDataType::UInt8:      return static_cast<int64_t>(*reinterpret_cast<const uint8_t*>(p));
        case MLOperatorTensorDataType::Int8:       return static_cast<int64_t>(*reinterpret_cast<const int8_t*>(p));
        case MLOperatorTensorDataType::UInt16:     return static_cast<int64_t>(*reinterpret_cast<const uint16_t*>(p));
        case MLOperatorTensorDataType::Int16:      return static_cast<int64_t>(*reinterpret_cast<const int16_t*>(p));
        case MLOperatorTensorDataType::Int32:      return static_cast<int64_t>(*reinterpret_cast<const int32_t*>(p));
        case MLOperatorTensorDataType::Int64:      return static_cast<int64_t>(*reinterpret_cast<const int64_t*>(p));
        case MLOperatorTensorDataType::String:     ML_INVALID_ARGUMENT("MLOperatorTensorDataType::String type is unsupported for reading as an integer.");
        case MLOperatorTensorDataType::Bool:       return static_cast<int64_t>(*reinterpret_cast<const uint8_t*>(p));
        case MLOperatorTensorDataType::Float16:    ML_INVALID_ARGUMENT("MLOperatorTensorDataType::Float16 type is unsupported for reading as an integer.");
        case MLOperatorTensorDataType::Double:     return static_cast<int64_t>(*reinterpret_cast<const double*>(p));
        case MLOperatorTensorDataType::UInt32:     return static_cast<int64_t>(*reinterpret_cast<const uint32_t*>(p));
        case MLOperatorTensorDataType::UInt64:     return static_cast<int64_t>(*reinterpret_cast<const uint64_t*>(p));
        case MLOperatorTensorDataType::Complex64:  return static_cast<int64_t>(*reinterpret_cast<const float*>(p)); // Read the real component.
        case MLOperatorTensorDataType::Complex128: return static_cast<int64_t>(*reinterpret_cast<const double*>(p)); // Read the real component.
        case MLOperatorTensorDataType::Undefined:
        default: ML_INVALID_ARGUMENT("Unknown MLOperatorTensorDataType.");
        };
    }

    double CastToFloat64(MLOperatorTensorDataType tensorDataType, const void* p)
    {
        switch (tensorDataType)
        {
        case MLOperatorTensorDataType::Float:      return static_cast<double>(*reinterpret_cast<const float*>(p));
        case MLOperatorTensorDataType::UInt8:      return static_cast<double>(*reinterpret_cast<const uint8_t*>(p));
        case MLOperatorTensorDataType::Int8:       return static_cast<double>(*reinterpret_cast<const int8_t*>(p));
        case MLOperatorTensorDataType::UInt16:     return static_cast<double>(*reinterpret_cast<const uint16_t*>(p));
        case MLOperatorTensorDataType::Int16:      return static_cast<double>(*reinterpret_cast<const int16_t*>(p));
        case MLOperatorTensorDataType::Int32:      return static_cast<double>(*reinterpret_cast<const int32_t*>(p));
        case MLOperatorTensorDataType::Int64:      return static_cast<double>(*reinterpret_cast<const int64_t*>(p));
        case MLOperatorTensorDataType::String:     ML_INVALID_ARGUMENT("MLOperatorTensorDataType::String type is unsupported for reading as an integer.");
        case MLOperatorTensorDataType::Bool:       return static_cast<double>(*reinterpret_cast<const uint8_t*>(p));
        case MLOperatorTensorDataType::Float16:    ML_INVALID_ARGUMENT("MLOperatorTensorDataType::Float16 type is unsupported for reading as an integer.");
        case MLOperatorTensorDataType::Double:     return static_cast<double>(*reinterpret_cast<const double*>(p));
        case MLOperatorTensorDataType::UInt32:     return static_cast<double>(*reinterpret_cast<const uint32_t*>(p));
        case MLOperatorTensorDataType::UInt64:     return static_cast<double>(*reinterpret_cast<const uint64_t*>(p));
        case MLOperatorTensorDataType::Complex64:  return static_cast<double>(*reinterpret_cast<const float*>(p)); // Read the real component.
        case MLOperatorTensorDataType::Complex128: return static_cast<double>(*reinterpret_cast<const double*>(p)); // Read the real component.
        case MLOperatorTensorDataType::Undefined:
        default: ML_INVALID_ARGUMENT("Unknown MLOperatorTensorDataType.");
        };
    }

    int64_t IsFloatDataType(MLOperatorTensorDataType tensorDataType)
    {
        switch (tensorDataType)
        {
        case MLOperatorTensorDataType::Float:
        case MLOperatorTensorDataType::Float16:
        case MLOperatorTensorDataType::Double:
        case MLOperatorTensorDataType::Complex64:
        case MLOperatorTensorDataType::Complex128:
            return true;
        };
        return false;
    }

    void ReadScalarTensorData(const MLOperatorTensor& tensor, /*out*/ void* data, size_t dataByteSize)
    {
        // Read the tensor bytes of a scalar value into the output data,
        // validating dimensions and byte size.
        const uint32_t elementCount = ComputeElementCountFromDimensions(tensor.GetShape());
        const size_t elementByteSize = GetByteSizeFromMlDataType(tensor.GetTensorDataType());
        ML_CHECK_VALID_ARGUMENT(tensor.IsCpuData(), "Tensor must be a CPU Tensor.");
        ML_CHECK_VALID_ARGUMENT(elementCount == 1, "Scalar tensors must have exactly 1 element.");
        ML_CHECK_VALID_ARGUMENT(dataByteSize >= elementByteSize, "Scalar tensor element byte size is too large.");

        memcpy(data, tensor.GetByteData(), elementByteSize);
    }

    int64_t ReadScalarTensorCastToInt64(const MLOperatorTensor& tensor)
    {
        std::byte tensorBytes[8];
        ReadScalarTensorData(tensor, /*out*/ &tensorBytes, sizeof(tensorBytes));
        return CastToInt64(tensor.GetTensorDataType(), &tensorBytes);
    }

    double ReadScalarTensorCastToFloat64(const MLOperatorTensor& tensor)
    {
        std::byte tensorBytes[8];
        ReadScalarTensorData(tensor, /*out*/ &tensorBytes, sizeof(tensorBytes));
        return CastToFloat64(tensor.GetTensorDataType(), &tensorBytes);
    }

    // Calculates the spatial dimensions from input dimensions and a kernel. The non-spatial (leading)
    // dimensions will be initialized to match the input dimensions. This assumes the spatial dimensions
    // are ordered such that they are at the end (e.g. NCHW or NCDHW).
    std::vector<DimensionType> InitializeKernelOutputDimensions(
        gsl::span<const DimensionType> inputDimensions,
        const KernelArgs& args
    )
    {
        ML_CHECK_VALID_ARGUMENT(gsl::narrow_cast<uint32_t>(inputDimensions.size()) >= args.spatialDimensionCount);
        int dimOffset = gsl::narrow_cast<int>(inputDimensions.size()) - args.spatialDimensionCount;

        std::vector<DimensionType> outputDimensions(inputDimensions.begin(), inputDimensions.end());

        for (size_t dim = 0; dim < args.spatialDimensionCount; ++dim)
        {
            uint32_t inputLength = gsl::narrow_cast<uint32_t>(inputDimensions[dimOffset + dim]);
            uint32_t paddedLength = inputLength + args.startPadding[dim] + args.endPadding[dim];
            uint32_t kernelLength = 1 + (args.windowSize[dim] - 1) * args.dilations[dim];

            ML_CHECK_VALID_ARGUMENT(kernelLength <= paddedLength, "kernelLength must be < paddedLength.");
            ML_CHECK_VALID_ARGUMENT(args.strides[dim] != 0, "strides must be non-zero.");
            uint32_t stride = args.strides[dim];
            uint32_t strideableOutputLength = paddedLength - kernelLength;
            uint32_t outputLength = 1 + (strideableOutputLength / stride)
                                      + (args.useCeilingOutputShape && (strideableOutputLength % stride != 0));

            outputDimensions[dimOffset + dim] = outputLength;
        }

        return outputDimensions;
    }

    // Calculates required input spatial dimensions to produce the given output dimensions.
    std::vector<DimensionType> InitializeKernelOutputDimsTranspose(
        gsl::span<const DimensionType> inputDimensions,
        const KernelArgs& args
    )
    {
        ML_CHECK_VALID_ARGUMENT(gsl::narrow_cast<uint32_t>(inputDimensions.size()) >= args.spatialDimensionCount);
        int dimOffset = gsl::narrow_cast<int>(inputDimensions.size()) - args.spatialDimensionCount;

        std::vector<DimensionType> outputDimensions(inputDimensions.begin(), inputDimensions.end());

        for (size_t dim = 0; dim < args.spatialDimensionCount; ++dim)
        {
            uint32_t padding = args.startPadding[dim] + args.endPadding[dim];
            uint32_t kernelLength = 1 + (args.windowSize[dim] - 1) * args.dilations[dim];
            
            outputDimensions[dimOffset + dim] = (inputDimensions[dimOffset + dim] - 1) * args.strides[dim] + kernelLength + args.outputPadding[dim] - padding;
        }

        return outputDimensions;
    }

    // Creates a kernel that spans the entire spatial dimensions of the input.
    KernelArgs InitializeGlobalKernel(gsl::span<const DimensionType> inputDimensions)
    {
        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() > NonspatialDimensionCount); // Must be at least 1D convolution (in 3D tensor)
        uint32_t spatialDimensionCount = gsl::narrow_cast<uint32_t>(inputDimensions.size()) - NonspatialDimensionCount;
        ML_CHECK_VALID_ARGUMENT(spatialDimensionCount <= NcdhwSpatialDimensionCount); // Support up to 3D convolution (in 5D tensor).

        KernelArgs args(spatialDimensionCount);

        for (size_t dim = 0; dim < spatialDimensionCount; ++dim)
        {
            args.strides[dim] = 1;
            args.dilations[dim] = 1;
            args.startPadding[dim] = 0;
            args.endPadding[dim] = 0;
            args.windowSize[dim] = gsl::narrow_cast<uint32_t>(inputDimensions[inputDimensions.size() - spatialDimensionCount + dim]);
        }

        return args;
    }

    // Creates a kernel from operator parameters.
    KernelArgs InitializeKernel(
        const MLOperatorAttributes& kernelInfo,
        uint32_t inputDimensionCount,
        gsl::span<const uint32_t> filterTensorShape
    )
    {
        static_assert(std::extent<decltype(OperatorHelper::KernelArgs::strides)>::value       == NcdhwSpatialDimensionCount);
        static_assert(std::extent<decltype(OperatorHelper::KernelArgs::dilations)>::value     == NcdhwSpatialDimensionCount);
        static_assert(std::extent<decltype(OperatorHelper::KernelArgs::windowSize)>::value    == NcdhwSpatialDimensionCount);
        static_assert(std::extent<decltype(OperatorHelper::KernelArgs::startPadding)>::value  == NcdhwSpatialDimensionCount);
        static_assert(std::extent<decltype(OperatorHelper::KernelArgs::endPadding)>::value    == NcdhwSpatialDimensionCount);
        static_assert(std::extent<decltype(OperatorHelper::KernelArgs::outputPadding)>::value == NcdhwSpatialDimensionCount);

        ML_CHECK_VALID_ARGUMENT(inputDimensionCount > NonspatialDimensionCount); // Must be at least 1D convolution (in 3D tensor)
        uint32_t spatialDimensionCount = inputDimensionCount - NonspatialDimensionCount;
        ML_CHECK_VALID_ARGUMENT(spatialDimensionCount <= NcdhwSpatialDimensionCount); // Support up to 3D convolution (in 5D tensor).

        KernelArgs args(spatialDimensionCount);

        if (kernelInfo.HasAttribute(AttrName::Strides, MLOperatorAttributeType::IntArray))
        {
            std::vector<int> kernelStrides = kernelInfo.GetOptionalAttributeVectorInt32(AttrName::Strides);
            ML_CHECK_VALID_ARGUMENT(kernelStrides.size() >= spatialDimensionCount);

            std::copy(kernelStrides.begin(), kernelStrides.begin() + spatialDimensionCount, args.strides);
        }
        else
        {
            std::fill(args.strides, args.strides + spatialDimensionCount, 1);
        }

        if (kernelInfo.HasAttribute(AttrName::Dilations, MLOperatorAttributeType::IntArray))
        {
            std::vector<int> kernelDilations = kernelInfo.GetOptionalAttributeVectorInt32(AttrName::Dilations);
            ML_CHECK_VALID_ARGUMENT(kernelDilations.size() >= spatialDimensionCount);

            std::copy(kernelDilations.begin(), kernelDilations.begin() + spatialDimensionCount, args.dilations);
        }
        else
        {
            std::fill(args.dilations, args.dilations + spatialDimensionCount, 1);
        }

        std::vector<int> kernelShape;

        if (kernelInfo.HasAttribute(AttrName::KernelShape, MLOperatorAttributeType::IntArray))
        {
            kernelShape = kernelInfo.GetOptionalAttributeVectorInt32(AttrName::KernelShape);
        }

        if (!kernelShape.empty())
        {
            std::copy(kernelShape.end() - spatialDimensionCount, kernelShape.end(), args.windowSize);
        }
        else if (!filterTensorShape.empty())
        {
            // If the kernel shape attribute is undefined, use the W weight tensor's shape.
            // For Conv and ConvTranspose, the ONNX spec specifies that kernel_shape should be inferred.
            std::copy(filterTensorShape.end() - spatialDimensionCount, filterTensorShape.end(), args.windowSize);
        }
        else
        {
            std::fill(args.windowSize, args.windowSize + spatialDimensionCount, 1);
        }

        std::string autoPad = kernelInfo.GetOptionalAttribute<std::string>(AttrName::AutoPad, AttrValue::NotSet);

        if (autoPad == AttrValue::NotSet)
        {
            // Use the pad values in the pads argument.
            std::vector<int> pads = kernelInfo.GetOptionalAttributeVectorInt32(AttrName::Pads);

            // if pads are not specified, assume all pad values are 0
            if (pads.empty())
            {
                pads.resize(2 * spatialDimensionCount);
            }

            ML_CHECK_VALID_ARGUMENT(pads.size() >= 2 * spatialDimensionCount);

            std::copy(pads.begin(), pads.begin() + spatialDimensionCount, args.startPadding);
            std::copy(pads.begin() + spatialDimensionCount, pads.begin() + spatialDimensionCount * 2, args.endPadding);
        }
        else if (autoPad == AttrValue::Valid)
        {
            std::fill(args.startPadding, args.startPadding + spatialDimensionCount, 0);
            std::fill(args.endPadding, args.endPadding + spatialDimensionCount, 0);
        }
        else
        {
            args.autoPad = true;
            args.autoPadSameUpper = autoPad == AttrValue::SameUpper;
            assert(args.autoPadSameUpper || autoPad == AttrValue::SameLower);
        }

        if (kernelInfo.HasAttribute(AttrName::OutputPadding, MLOperatorAttributeType::IntArray))
        {
            std::vector<int> outputPadding = kernelInfo.GetOptionalAttributeVectorInt32(AttrName::OutputPadding);
            ML_CHECK_VALID_ARGUMENT(outputPadding.size() >= 2);
        
            std::copy(outputPadding.begin(), outputPadding.begin() + spatialDimensionCount, args.outputPadding);
        }
        else
        {
            std::fill(args.outputPadding, args.outputPadding + spatialDimensionCount, 0);
        }

        args.useCeilingOutputShape = kernelInfo.GetOptionalAttribute<bool>(AttrName::CeilMode, 0);

        return args;
    }

    void ResolveAutoPadding(
        KernelArgs& args,
        gsl::span<const DimensionType> inputDimensions
    )
    {
        if (!args.autoPad)
        {
            return;
        }

        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() > NonspatialDimensionCount); // Must be at least 1D convolution (in 3D tensor)
        uint32_t spatialDimensionCount = gsl::narrow_cast<uint32_t>(inputDimensions.size()) - NonspatialDimensionCount;
        ML_CHECK_VALID_ARGUMENT(spatialDimensionCount <= NcdhwSpatialDimensionCount); // Support up to 3D convolution (in 5D tensor).

        const int dimOffset = gsl::narrow_cast<int>(inputDimensions.size()) - spatialDimensionCount;

        for (size_t dim = 0; dim < spatialDimensionCount; ++dim)
        {
            // Auto-padding is used to ensure the output dimensions match the input
            // dimensions. For example, if the input has length 3 and the kernel has
            // length 3, then the output size (without padding) would be 1. To get the
            // output size to 3, we need enough padding to fit the kernel two more times.
            uint32_t inputLength = gsl::narrow_cast<uint32_t>(inputDimensions[dimOffset + dim]);
            uint32_t stridedOutputLength = (inputLength + args.strides[dim] - 1) / args.strides[dim];
            uint32_t kernelLength = 1 + (args.windowSize[dim] - 1) * args.dilations[dim];

            // Reserve enough space for one kernel (kernelLength), then add space for N
            // strided "slides" of the kernel. N is (stridedOutputLength-1), because the output
            // length must be equal to the input length; -1 is for the initial kernelLength
            // space that has been accounted for.
            uint32_t lengthNeeded = args.strides[dim] * (stridedOutputLength - 1) + kernelLength;

            // Once the total length needed is known, subtract the given length to get the
            // padding. The padding is distributed evenly to both sides, with the odd amount
            // going to the start or end based on the auto padding mode.
            uint32_t padding = (lengthNeeded <= inputLength) ? 0 : (lengthNeeded - inputLength);

            if (args.autoPadSameUpper)
            {
                args.startPadding[dim] = padding / 2;
            }
            else
            {
                args.startPadding[dim] = (padding + 1) / 2;
            }

            args.endPadding[dim] = padding - args.startPadding[dim];
        }
    }
    
    void MatMulShapeMapping(
        std::vector<DimensionType>& inputShape0,
        std::vector<DimensionType>& inputShape1,
        std::vector<DimensionType>& outputShape)
    {
        // Get the padded input shapes and undo the effect of padding removal from the output shape
        if (inputShape1.size() == 1)
        {
            inputShape1.push_back(1);
            outputShape.push_back(1);
        }

        if (inputShape0.size() == 1)
        {
            inputShape0.insert(inputShape0.begin(), 1);
            outputShape.insert(outputShape.end() - 1, 1);
        }

        // Remove the batch dimensions from each input, then re-add the broadcasted batch dimensions
        // based on the output shape
        inputShape0.erase(inputShape0.begin(), inputShape0.end() - 2);
        inputShape1.erase(inputShape1.begin(), inputShape1.end() - 2);

        inputShape0.insert(inputShape0.begin(), outputShape.begin(), outputShape.end() - 2);
        inputShape1.insert(inputShape1.begin(), outputShape.begin(), outputShape.end() - 2);
    }

    std::vector<EdgeShapes> GetOutputShapeAsInputShapeHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        assert(shapeInfo.GetInputCount() > m_inputTensorIndex);
        std::vector<DimensionType> outputDimensions = shapeInfo.GetInputTensorShape(m_inputTensorIndex);
        return { std::move(outputDimensions) };
    }

    std::vector<EdgeShapes> GetBroadcastedOutputShapeHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputCount() >= 1);

        std::vector<DimensionType> accumulatedInputShape = shapeInfo.GetInputTensorShape(0);
        for (uint32_t i = 1, ci = shapeInfo.GetInputCount(); i < ci; ++i)
        {
            std::vector<DimensionType> nextInputShape = shapeInfo.GetInputTensorShape(i);
            accumulatedInputShape = BroadcastTensorShape(accumulatedInputShape, nextInputShape);
        }

        return { std::move(accumulatedInputShape) };
    }

    std::vector<DimensionType> BroadcastTensorShape(
        gsl::span<const DimensionType> inputShape0,
        gsl::span<const DimensionType> inputShape1
        )
    {
        if (inputShape0 == inputShape1)
        {
            return { inputShape0.begin(), inputShape0.end() };
        }

        const auto outputRank = std::max(inputShape0.size(), inputShape1.size());
        std::vector<DimensionType> outputShape(outputRank);

        // Walk backwards through both input shapes and broadcast each dimension
        auto inDim0Iter = inputShape0.rbegin();
        auto inDim1Iter = inputShape1.rbegin();
        for (auto outDimIter = outputShape.rbegin(); outDimIter != outputShape.rend(); ++outDimIter)
        {
            DimensionType inDimension0 = 1;
            if (inDim0Iter != inputShape0.rend())
            {
                inDimension0 = *inDim0Iter;
                ++inDim0Iter;
            }

            DimensionType inDimension1 = 1;
            if (inDim1Iter != inputShape1.rend())
            {
                inDimension1 = *inDim1Iter;
                ++inDim1Iter;
            }

            ML_CHECK_VALID_ARGUMENT((inDimension0 == inDimension1) || (inDimension0 == 1) || (inDimension1 == 1));
            *outDimIter = std::max(inDimension0, inDimension1);
        }

        return outputShape;
    }

    void ConvolutionHelperBase::ResolvingPadding(gsl::span<const DimensionType> inputDimensions)
    {
        ResolveAutoPadding(m_kernel, inputDimensions);
    }

    std::vector<EdgeShapes> GemmHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto inputShapeA = shapeInfo.GetInputTensorShape(IN_A);
        auto inputShapeB = shapeInfo.GetInputTensorShape(IN_B);

        ML_CHECK_VALID_ARGUMENT(inputShapeA.size() >= 2, "Need 2 dimensions in A.");
        ML_CHECK_VALID_ARGUMENT(inputShapeB.size() >= 2, "Need 2 dimensions in B.");

        uint32_t m = gsl::narrow_cast<uint32_t>(m_transA ? inputShapeA[1] : inputShapeA[0]);
        uint32_t n = gsl::narrow_cast<uint32_t>(m_transB ? inputShapeB[0] : inputShapeB[1]);

        EdgeShapes outputShape({ m, n });
        return { std::move(outputShape) };
    }

    void SplitHelper::Initialize(
        const MLOperatorAttributes& operatorAttributes,
        gsl::span<const DimensionType> inputDimensions
        )
    {
        const uint32_t inputDimCount = gsl::narrow_cast<int32_t>(inputDimensions.size());
        m_axis = static_cast<int>(HandleNegativeAxis(operatorAttributes.GetOptionalAttribute<int32_t>(AttrName::Axis, 0), inputDimCount));
        m_split = operatorAttributes.GetOptionalAttributeVectorInt32(AttrName::Split);
    }

    std::vector<EdgeShapes> SplitHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        const std::vector<DimensionType> inputDimensions = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(m_axis < gsl::narrow_cast<int>(inputDimensions.size()));

        const uint32_t outputCount = shapeInfo.GetOutputCount();
        ML_CHECK_VALID_ARGUMENT(outputCount > 0);
        std::vector<std::vector<DimensionType>> outputDimensionsList(outputCount);

        if (!m_split.empty())
        {
            // Runtime mismatch between the specified split attribute and actual number of outputs desired.
            ML_CHECK_VALID_ARGUMENT(m_split.size() == outputCount);

            int totalOperatorElementCount = 0;
            for (int operatorElementCount : m_split)
            {
                // Output stream should have 0 or more elements.
                totalOperatorElementCount += operatorElementCount;
            }

            // Sum of each split should match input element count along that axis.
            ML_CHECK_VALID_ARGUMENT(totalOperatorElementCount == gsl::narrow_cast<int>(inputDimensions[m_axis]));

            for (uint32_t i = 0; i < outputCount; ++i)
            {
                outputDimensionsList[i] = inputDimensions;
                outputDimensionsList[i][m_axis] = m_split[i];
            }
        }
        else
        {
            // Split input stream equally across output streams.
            ML_CHECK_VALID_ARGUMENT(inputDimensions[m_axis] % outputCount == 0);

            DimensionType equalSplit = inputDimensions[m_axis] / outputCount;
            for (uint32_t i = 0; i < outputCount; ++i)
            {
                outputDimensionsList[i] = inputDimensions;
                outputDimensionsList[i][m_axis] = equalSplit;
            }
        }

        std::vector<EdgeShapes> edgeShapes;
        for (uint32_t i = 0; i != outputCount; ++i)
        {
            edgeShapes.push_back(outputDimensionsList[i]);
        }

        return edgeShapes;
    }

    std::vector<EdgeShapes> SliceHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return { m_outputDimensions };
    }

    void PaddingHelper::Initialize(const MLOperatorAttributes& operatorAttributes, gsl::span<int32_t> padding, uint32_t opsetVersion)
    {
        ML_CHECK_VALID_ARGUMENT(padding.size() % 2 == 0, "Padding must be even count, including begin/end pairs.");

        uint32_t dimCount = gsl::narrow_cast<uint32_t>(padding.size() / 2);
        m_startPadding.resize(dimCount);
        m_endPadding.resize(dimCount);
        std::copy(padding.begin(), padding.begin() + dimCount, m_startPadding.begin());
        std::copy(padding.begin() + dimCount, padding.begin() + dimCount * 2, m_endPadding.begin());
    }

    std::vector<EdgeShapes> PaddingHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        std::vector<DimensionType> outputDimensions = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(
            m_startPadding.size() == outputDimensions.size() &&
            m_endPadding.size()   == outputDimensions.size()
            );
            
        for (size_t i = 0; i < outputDimensions.size(); ++i)
        {
            outputDimensions[i] += m_startPadding[i] + m_endPadding[i];
        }

        return { std::move(outputDimensions) };
    }

    std::vector<EdgeShapes> MultinomialHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        int32_t sampleSize = gsl::narrow_cast<int32_t>(shapeInfo.GetOptionalAttribute<int64_t>(AttrName::SampleSize, -1));
        ML_CHECK_VALID_ARGUMENT(sampleSize > 0);

        // Set last dimension to sample size.
        std::vector<DimensionType> outputDimensions = shapeInfo.GetInputTensorShape(0);
        uint32_t rank = gsl::narrow_cast<uint32_t>(outputDimensions.size());
        ML_CHECK_VALID_ARGUMENT(rank == 2);
        outputDimensions[rank - 1] = sampleSize;

        return { std::move(outputDimensions) };
    }

    void GatherHelper::Initialize(
        const MLOperatorAttributes& operatorAttributes,
        gsl::span<const DimensionType> inputDimensions
        )
    {
        int32_t signedOnnxAxis = operatorAttributes.GetOptionalAttribute<int>(AttrName::Axis, 0);
        uint32_t inputRank = gsl::narrow_cast<int>(inputDimensions.size());
        m_axis = HandleNegativeAxis(signedOnnxAxis, inputRank);
    }

    std::vector<EdgeShapes> GatherHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        std::vector<DimensionType> inputDimensions = shapeInfo.GetInputTensorShape(0);
        std::vector<DimensionType> indicesDimensions = shapeInfo.GetInputTensorShape(1);

        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() >= 1);
        ML_CHECK_VALID_ARGUMENT(indicesDimensions.size() >= 0);
        int outDimCount = gsl::narrow_cast<int>(inputDimensions.size() + indicesDimensions.size() - 1);
        ML_CHECK_VALID_ARGUMENT(outDimCount >= 0 && outDimCount <= NchwDimensionCount);

        std::vector<DimensionType> outputDimensions(outDimCount, 1);

        // The input dimensions following the gather axis determine the final output dimensions.
        int outputDim = outDimCount - 1;
        int inputDim = gsl::narrow_cast<int>(inputDimensions.size() - 1);
        for (; inputDim > m_axis; --outputDim, --inputDim)
        {
            outputDimensions[outputDim] = inputDimensions[inputDim];
        }

        // The shape of the index tensor is reflected in the middle dimensions of the output tensor.
        int indexDim = gsl::narrow_cast<int>(indicesDimensions.size() - 1);
        for (; indexDim >= 0; --outputDim, --indexDim)
        {
            outputDimensions[outputDim] = indicesDimensions[indexDim];
        }

        // The gather dimension is skipped for the purposes of sizing because the index values choose slices
        // across it.  Preceding input dimensions determine the shape of the output's leading dimensions.
        inputDim = m_axis - 1;
        for (; outputDim >= 0 && inputDim >= 0; --outputDim, --inputDim)
        {
            outputDimensions[outputDim] = inputDimensions[inputDim];
        }

        return { EdgeShapes(std::move(outputDimensions)) };
    }

    std::vector<EdgeShapes> GatherNdHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        std::vector<DimensionType> inputDimensions = shapeInfo.GetInputTensorShape(0);
        std::vector<DimensionType> indicesDimensions = shapeInfo.GetInputTensorShape(1);

        // Determine the number of output dimensions.
        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() >= 1);
        ML_CHECK_VALID_ARGUMENT(indicesDimensions.size() >= 1);
        const uint32_t numberOfCoordinatesPerIndex = indicesDimensions.back();
        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() >= numberOfCoordinatesPerIndex);
        const uint32_t numberOfOutputDimensionsFromInput = static_cast<uint32_t>(inputDimensions.size()) - numberOfCoordinatesPerIndex;
        const uint32_t numberOfOutputDimensionsFromIndices = static_cast<uint32_t>(indicesDimensions.size()) - 1; // Strip off last dimension.
        uint32_t outputDimensionCount = gsl::narrow_cast<uint32_t>(numberOfOutputDimensionsFromIndices + numberOfOutputDimensionsFromInput);
        ML_CHECK_VALID_ARGUMENT(outputDimensionCount > 0 && outputDimensionCount <= NchwDimensionCount);

        // Form the full expected size by concatenating the prefix part of the indices tensor shape
        // with the suffix of the input tensor shape.
        std::vector<DimensionType> outputDimensions;
        outputDimensions.assign(indicesDimensions.begin(), indicesDimensions.end() - 1);
        outputDimensions.insert(outputDimensions.end(), inputDimensions.end() - numberOfOutputDimensionsFromInput, inputDimensions.end());

        return { EdgeShapes(std::move(outputDimensions)) };
    }

    void TransposeHelper::Initialize(
        const MLOperatorAttributes& operatorAttributes,
        gsl::span<const DimensionType> inputDimensions
        )
    {
        // Perm is list of indices for the ordering of the output tensor dimensions.
        // For example, if the input shape is {2,3,4}:
        // perm = {0,1,2} outputs a tensor with shape {2,3,4}
        // perm = {2,1,0} outputs a tensor with shape {4,3,2}
        // perm = {2,0,1} outputs a tensor with shape {4,2,3}
        m_permutations = operatorAttributes.GetOptionalAttributeVectorInt32("perm");

        // If no permutations were given, the default behavior is to reverse the axes.
        // e.g [1, 0] would swap horizontal and vertical axes.
        if (m_permutations.empty())
        {
            m_permutations.resize(inputDimensions.size());
            int index = gsl::narrow_cast<int>(inputDimensions.size());
            for (auto& p : m_permutations)
            {
                p = --index;
            }
        }
    }

    std::vector<EdgeShapes> TransposeHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        std::vector<DimensionType> inputDimensions = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(m_permutations.size() == inputDimensions.size());

        // Permute the shape.
        std::vector<DimensionType> outputDimensions(inputDimensions.size());
        for (int dimInput = 0, dimCount = gsl::narrow_cast<int>(inputDimensions.size()); dimInput < dimCount; ++dimInput)
        {
            auto dimPermuted = gsl::narrow_cast<size_t>(m_permutations[dimInput]);
            ML_CHECK_VALID_ARGUMENT(dimPermuted < inputDimensions.size());
            outputDimensions[dimInput] = inputDimensions[dimPermuted];
        }

        return { EdgeShapes(std::move(outputDimensions)) };
    }

    std::vector<EdgeShapes> ReduceHelperBase::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        // Example:
        // Input dims   : {3,2,5}
        // Input axes   : {0,2}
        // Reduced      : {T,F,T}
        // Reduced dims : {1,2,1}
        // DML axes     : {1,3}
        // DML dims     : {1,1,2,1}
        // Pruned dims  : {2}
        // Dim Offset   : 1

        std::vector<DimensionType> reducedDims = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(reducedDims.size() <= NchwDimensionCount);

        std::vector<bool> reduced(reducedDims.size(), false);

        for (auto& dim : m_axes)
        {
            ML_CHECK_VALID_ARGUMENT(static_cast<size_t>(dim) < reduced.size(), "Axis in 'axes' is invalid beyond range.");
            reduced[dim] = true;
            reducedDims[dim] = 1;
        }

        if (m_keepDims)
        {
            return { std::move(reducedDims) };
        }
        else
        {
            // Can't simply prune dims of size 1, because they may have been
            // size 1 in the input but not listed in the axes to reduce. This
            // is the reason for the reduced bool array.
            std::vector<DimensionType> prunedDims;
            for (int i = 0, ci = gsl::narrow_cast<int>(reducedDims.size()); i < ci; ++i)
            {
                if (!reduced[i])
                {
                    prunedDims.push_back(reducedDims[i]);
                }
            }

            return { std::move(prunedDims) };
        }
    }

    void ReduceHelperBase::AdjustAxesAndOutputShape(const std::vector<uint32_t>& inputShape)
    {
        ML_CHECK_VALID_ARGUMENT(inputShape.size() <= NchwDimensionCount);

        // If axes is not specified, reduce over all the dimensions
        if (m_axes.empty())
        {
            m_axes.resize(inputShape.size());
            std::iota(m_axes.begin(), m_axes.end(), 0);
        }
    }
    
    std::vector<EdgeShapes> MatMulHelperBase::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputCount() >= 2);

        // Following numpy.matmul for shape inference:
        // https://docs.scipy.org/doc/numpy/reference/generated/numpy.matmul.html
        // The behavior depends on the arguments in the following way.
        // * If both arguments are 2 - D they are multiplied like conventional matrices.
        // * If either argument is N - D, N > 2, it is treated as a stack of matrices residing in the last two indexes and broadcast accordingly.
        // * If the first argument is 1 - D, it is promoted to a matrix by prepending a 1 to its dimensions. After matrix multiplication the prepended 1 is removed.
        // * If the second argument is 1 - D, it is promoted to a matrix by appending a 1 to its dimensions. After matrix multiplication the appended 1 is removed.

        auto inputShape0 = shapeInfo.GetInputTensorShape(m_aTensorIndex);
        auto inputShape1 = shapeInfo.GetInputTensorShape(m_bTensorIndex);
        ML_CHECK_VALID_ARGUMENT(inputShape0.size() >= 1);
        ML_CHECK_VALID_ARGUMENT(inputShape1.size() >= 1);

        std::vector<uint32_t> outputMatrixDims;

        // Modify the input and truncated output shapes per the above comments.
        // The extra dimensions of the output beyond the two matrix dimensions
        // will be computed afterward by broadcasting.
        if (inputShape0.size() == 1)
        {
            inputShape0.insert(inputShape0.begin(), 1);
        }
        else
        {
            outputMatrixDims.push_back(inputShape0[inputShape0.size() - 2]);
        }

        if (inputShape1.size() == 1)
        {
            inputShape1.push_back(1);
        }
        else
        {
            outputMatrixDims.push_back(inputShape1[inputShape1.size() - 1]);
        }

        // Remove the matrix dimensions from each input, resulting in broadcastable shapes
        std::vector<uint32_t> batchDims0(inputShape0.begin(), inputShape0.end() - 2);
        std::vector<uint32_t> batchDims1(inputShape1.begin(), inputShape1.end() - 2);

        // Broadcast the extra dimensions of each input, then add the truncated matrix dimensions
        std::vector<uint32_t> outputDims = BroadcastTensorShape(batchDims0, batchDims1);
        for (uint32_t matrixDim : outputMatrixDims)
        {
            outputDims.push_back(matrixDim);
        }

        return {std::move(outputDims)};
    }
    
    std::vector<EdgeShapes> TopKHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        assert(m_axis >= 0);

        std::vector<DimensionType> outputDimensions = shapeInfo.GetInputTensorShape(0);
        outputDimensions[m_axis] = m_k;

        EdgeShapes outputShape(outputDimensions);

        // There are two outputs for TopK: sorted data and its corresponding index.
        return { outputShape, outputShape };
    }

    std::vector<EdgeShapes> RecurrentHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        // X = [SEQ_LENGTH, BATCH_SIZE, ...]
        // W = [NUM_DIRECTIONS, ...]
        auto inputXShape = shapeInfo.GetInputTensorShape(0); // input X
        auto inputWShape = shapeInfo.GetInputTensorShape(1); // input W
        const DimensionType seqLength = inputXShape[0];
        const DimensionType batchSize = inputXShape[1];
        const DimensionType numDirections = inputWShape[0];

        DimensionType outputDimensionsSequence[4] = { seqLength, numDirections, batchSize, gsl::narrow_cast<DimensionType>(m_hiddenSize) };
        DimensionType outputDimensionsSingle[3] = { numDirections, batchSize, gsl::narrow_cast<DimensionType>(m_hiddenSize) };
        DimensionType cellOutputDimensionsSingle[3] = { numDirections, batchSize, gsl::narrow_cast<DimensionType>(m_hiddenSize) };

        auto outputCount = shapeInfo.GetOutputCount();
        switch (outputCount)
        {
        case 0:
        {
            return {};
        }
        case 1:
        {
            return { std::move(EdgeShapes(outputDimensionsSequence)) };
        }
        case 2:
        {
            std::vector<EdgeShapes> outputShapes = {
                EdgeShapes(outputDimensionsSequence),
                EdgeShapes(outputDimensionsSingle)
            };

            return std::move(outputShapes);
        }
        case 3:
        {
            std::vector<EdgeShapes> outputShapes = {
                EdgeShapes(outputDimensionsSequence),
                EdgeShapes(outputDimensionsSingle),
                EdgeShapes(cellOutputDimensionsSingle)
            };

            return std::move(outputShapes);
        }
        default:
        {
            assert(false);
            return {};
        }
        }
    }
    
    std::vector<EdgeShapes> RandomUniformHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        EdgeShapes outputShape(m_tensorShape);
        return { std::move(outputShape) };
    }

    std::vector<EdgeShapes> RandomNormalHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        EdgeShapes outputShape(m_tensorShape);
        return { std::move(outputShape) };
    }

    void ConcatHelper::Initialize(
        const MLOperatorAttributes& operatorAttributes,
        gsl::span<const DimensionType> inputDimensions
        )
    {
        uint32_t inputDimCount = gsl::narrow_cast<uint32_t>(inputDimensions.size());
        m_axis = static_cast<int>(HandleNegativeAxis(operatorAttributes.GetOptionalAttribute<int>(AttrName::Axis, -1), inputDimCount));
        ML_CHECK_VALID_ARGUMENT(m_axis < static_cast<int>(inputDimensions.size()));
    }

    std::vector<EdgeShapes> ConcatHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto outputShape = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(outputShape.size() <= NchwDimensionCount);

        uint32_t inputCount = shapeInfo.GetInputCount();

        for (uint32_t i = 1; i < inputCount; ++i)
        {
            auto inputShape = shapeInfo.GetInputTensorShape(i);
            for (size_t j = 0; j < outputShape.size(); ++j)
            {
                if (m_axis == j)
                {
                    outputShape[j] += inputShape[j];
                }
            }
        }

        return { std::move(EdgeShapes(outputShape)) };
    }

    void CropHelper::Initialize(
        const MLOperatorAttributes& operatorAttributes,
        gsl::span<const DimensionType> inputDimensions
    )
    {
        std::vector<int> border = operatorAttributes.GetOptionalAttributeVectorInt32(AttrName::Border);
        ML_CHECK_VALID_ARGUMENT(border.size() == 4u, "Border must be size 4.");

        m_offsets[N] = 0;
        m_offsets[C] = 0;
        m_offsets[H] = border[Top];
        m_offsets[W] = border[Left];

        if (operatorAttributes.HasAttribute(AttrName::Scale, MLOperatorAttributeType::IntArray))
        {
            // Scale overrides whatever is in the border right/bottom.
            std::vector<int> scale = operatorAttributes.GetOptionalAttributeVectorInt32(AttrName::Scale);
            m_sizes[Height] = scale[Height];
            m_sizes[Width] = scale[Width];
        }
        else
        {
            ML_CHECK_VALID_ARGUMENT(inputDimensions.size() == 4, "Wrong number of dimensions.");
            m_sizes[Height] = inputDimensions[H] - border[Bottom] - border[Top];
            m_sizes[Width] = inputDimensions[W] -border[Right] - border[Left];
        }
    }

    std::vector<EdgeShapes> CropHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto inputDimensions = shapeInfo.GetInputTensorShape(0);

        DimensionType outputDimensions[4] =
        {
            inputDimensions[N],
            inputDimensions[C],
            m_sizes[Height],
            m_sizes[Width]
        };

        return { std::move(EdgeShapes(outputDimensions)) };
    }

    std::vector<EdgeShapes> DepthToSpaceHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto inputDimensions = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() == 4, "Wrong number of dimensions.");

        assert(m_blockSize > 0);

        DimensionType outputDimensions[4] =
        {
            inputDimensions[N],
            inputDimensions[C] / (m_blockSize * m_blockSize),
            inputDimensions[H] * m_blockSize,
            inputDimensions[W] * m_blockSize,
        };

        return { std::move(EdgeShapes(outputDimensions)) };
    }

    void FlattenHelper::Initialize(
        const MLOperatorAttributes& operatorAttributes,
        gsl::span<const DimensionType> inputDimensions
        )
    {
        int32_t inputDimCount = gsl::narrow_cast<int32_t>(inputDimensions.size());
        int32_t axis = operatorAttributes.GetOptionalAttribute<int32_t>(AttrName::Axis, 1);
        // Flatten can accept an axis [-r, r], including one past the last absolute index.
        m_axis = (axis == inputDimCount) ? axis : static_cast<int>(HandleNegativeAxis(axis, inputDimCount));
    }

    std::vector<EdgeShapes> FlattenHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto inputDimensions = shapeInfo.GetInputTensorShape(0);

        std::vector<DimensionType> outputDimensions;

        // Axis indicates which input dimensions should be flattened to
        // the first dimension of the output. The default axis is 1, which
        // preserves the first dimension (typically batch size), and flattens
        // the remaining dimensions. An axis of 0 means no input dimensions
        // are flattened into the first output dimension, so the output is 1D 
        // padded with a 1 in the first diemension.
        ML_CHECK_VALID_ARGUMENT(m_axis >= 0 && m_axis <= gsl::narrow_cast<int>(inputDimensions.size()));
        gsl::span<const DimensionType> outputDimensionsSpan(inputDimensions);
        DimensionType elementsToAxis = ComputeElementCountFromDimensions(outputDimensionsSpan.subspan(0, m_axis));
        DimensionType elementsFromAxis = ComputeElementCountFromDimensions(outputDimensionsSpan.subspan(m_axis, inputDimensions.size() - m_axis));
        outputDimensions.assign({ elementsToAxis, elementsFromAxis });

        return { std::move(outputDimensions) };
    }

    std::vector<EdgeShapes> PoolingHelperBase::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto inputShape = shapeInfo.GetInputTensorShape(0);
        std::vector<DimensionType> outputDimensions = InitializeKernelOutputDimensions(inputShape, m_kernel);

        // MaxPool may have both an output and an indices tensor (both the same size).
        const uint32_t outputCount = shapeInfo.GetOutputCount();
        assert(outputCount == 1 || outputCount == 2);

        std::vector<EdgeShapes> outputShapes;
        for (uint32_t i = 0; i < outputCount; ++i)
        {
            outputShapes.push_back(outputDimensions);
        }
        return outputShapes;
    }

    std::vector<EdgeShapes> RoiPoolingHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto roiShape = shapeInfo.GetInputTensorShape(InputTensors::ROIS);
        auto inputShape = shapeInfo.GetInputTensorShape(InputTensors::INPUT);
        ML_CHECK_VALID_ARGUMENT(inputShape.size() >= 4, "inputShape must be >= 4.");
        
        DimensionType outputDimensions[4] =
        {
            roiShape[0], // number of ROIs
            inputShape[C], // number of channels
            static_cast<DimensionType>(m_pooledSizeH),
            static_cast<DimensionType>(m_pooledSizeW),
        };

        return { std::move(EdgeShapes(outputDimensions)) };
    }

    void UnpoolingHelper::Initialize()
    {
        ResolveAutoPadding(m_kernel, m_inputShape);
        m_inferredOutputDimensions = InitializeKernelOutputDimsTranspose(m_inputShape, m_kernel);
    }

    std::vector<EdgeShapes> UnpoolingHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        std::vector<DimensionType> outputDimensions;

        if (shapeInfo.IsInputValid(2))
        {
            // Read the dimensions from the output_shape tensor.
            MLOperatorTensor outputShapeTensor = shapeInfo.GetConstantInputTensor(2);
            ML_CHECK_VALID_ARGUMENT(outputShapeTensor.IsCpuData(), "MaxUnpool's scales tensor must be CPU Tensor.");

            const std::vector<uint32_t> outputShapeTensorDimensions = outputShapeTensor.GetShape();
            ML_CHECK_VALID_ARGUMENT(outputShapeTensorDimensions.size() == 1, "output_shape tensor must be 1D.");
            const size_t dimCount = outputShapeTensorDimensions[0];
            const int64_t* data = outputShapeTensor.GetData<int64_t>();

            ML_CHECK_VALID_ARGUMENT(dimCount == m_inputShape.size(), "Input dimensions and output_shape must have same rank.");
            DowncastDimensions(gsl::make_span(data, dimCount), /*out*/ outputDimensions);
        }
        else if (shapeInfo.HasAttribute(AttrName::OutputShape, MLOperatorAttributeType::IntArray))
        {
            std::vector<int64_t> outputDimensions64bit = shapeInfo.GetAttributeVector<int64_t>(AttrName::OutputShape);
            ML_CHECK_VALID_ARGUMENT(outputDimensions64bit.size() == m_inputShape.size(), "Input dimensions and output_shape must have same rank.");
            DowncastDimensions(outputDimensions64bit, /*out*/ outputDimensions);
        }
        else
        {
            outputDimensions = m_inferredOutputDimensions;
        }

        return { std::move(outputDimensions) };
    }

    void SqueezeHelper::Initialize(
        gsl::span<const int32_t> axes,
        gsl::span<const DimensionType> inputDimensions
        )
    {
        m_axes.assign(axes.begin(), axes.end());
        HandleNegativeAxes(/*inout*/ m_axes, gsl::narrow_cast<uint32_t>(inputDimensions.size()));
        std::sort(m_axes.begin(), m_axes.end());
    }

    std::vector<EdgeShapes> SqueezeHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto outputDimensions = shapeInfo.GetInputTensorShape(0);
        auto inputDimCount = gsl::narrow_cast<int>(outputDimensions.size());

        // Flags that indicates if a dimension should be removed. Smallest dimension is the least significant bit.
        uint32_t shouldSqueezeDim = 0;

        for (size_t i = 0; i < m_axes.size(); ++i)
        {
            int dimIndex = m_axes[i];
            ML_CHECK_VALID_ARGUMENT(dimIndex >= 0 && dimIndex < inputDimCount, "'axes' must be valid with within actual input dimensions.");
            if (outputDimensions[dimIndex] == 1)
            {
                shouldSqueezeDim |= 1 << dimIndex;
            }
        }

        uint32_t newOutputDimCount = 0;

        for (uint32_t i = 0; i < outputDimensions.size(); i++)
        {
            if (!(shouldSqueezeDim & (1 << i)))
            {
                outputDimensions[newOutputDimCount++] = outputDimensions[i];
            }
        }

        outputDimensions.resize(newOutputDimCount);

        return { std::move(outputDimensions) };
    }

    void UnsqueezeHelper::Initialize(
        gsl::span<const int32_t> axes,
        gsl::span<const DimensionType> inputDimensions
        )
    {
        m_axes.assign(axes.begin(), axes.end());
        const uint32_t outputDimensionCount = gsl::narrow_cast<uint32_t>(inputDimensions.size() + axes.size());
        HandleNegativeAxes(/*inout*/ m_axes, outputDimensionCount);
        std::sort(m_axes.begin(), m_axes.end());
    }

    std::vector<EdgeShapes> UnsqueezeHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto inputDimensions = shapeInfo.GetInputTensorShape(0);
        std::vector<DimensionType> outputDimensions(m_axes.size() + inputDimensions.size());

        outputDimensions.assign(inputDimensions.begin(), inputDimensions.end());

        for (size_t i = 0; i < m_axes.size(); ++i)
        {
            ML_CHECK_VALID_ARGUMENT(m_axes[i] >= 0, "Attribute axes should contain non-negative integers.");
            ML_CHECK_VALID_ARGUMENT(m_axes[i] <= gsl::narrow_cast<int32_t>(outputDimensions.size()), "Attribute axes must be within range.");
            outputDimensions.insert(outputDimensions.begin() + m_axes[i], 1);
        }

        return { std::move(outputDimensions) };
    }
    
    std::vector<EdgeShapes> SpaceToDepthHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto inputDimensions = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() == 4, "Wrong number of dimensions.");

        assert(m_blockSize > 0);

        DimensionType outputDimensions[4] =
        {
            inputDimensions[N],
            inputDimensions[C] * m_blockSize * m_blockSize,
            inputDimensions[H] / m_blockSize,
            inputDimensions[W] / m_blockSize,
        };

        return { std::move(EdgeShapes(outputDimensions)) };
    }
        
    std::vector<EdgeShapes> ReshapeHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        // Fill in the output dimensions. The shape may have -1 in a single dimension,
        // which means to infer the size from the remaining elements. For example, if
        // the input has 16 elements and the shape is {2,2,-1}, then the last dimension
        // must be 4 (2*2*4 = 16 elements).
        const std::vector<DimensionType> inputDimensions = shapeInfo.GetInputTensorShape(0);

        std::vector<DimensionType> outputDimensions(m_shapeDims.size());

        DimensionType outElementCount = 1;
        int inferDim = -1;

        DimensionType inElementCount = ComputeElementCountFromDimensions(inputDimensions);

        for (int i = 0, ci = gsl::narrow_cast<int>(m_shapeDims.size()); i < ci; ++i)
        {
            switch (m_shapeDims[i])
            {
            case -1:
                ML_CHECK_VALID_ARGUMENT(inferDim == -1, "Only one dimension can be inferred.");
                inferDim = i;
                break;

            case 0:
                outputDimensions[i] = inputDimensions[i];
                outElementCount *= outputDimensions[i];
                break;

            default:
                outputDimensions[i] = m_shapeDims[i];
                outElementCount *= outputDimensions[i];
                break;
            }
        }

        if (inferDim != -1)
        {
            outputDimensions[inferDim] = inElementCount / outElementCount;
            outElementCount *= outputDimensions[inferDim];
        }
        
        return { std::move(EdgeShapes(outputDimensions)) };
    }

    std::vector<EdgeShapes> ExpandHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        std::vector<uint32_t> outputDimensions;

        // The 'shape' tensor is a 1D tensor holding the new shape to expand to,
        // and the first element of its own shape holds how many dimensions there
        // will be for the output.

        std::vector<uint32_t> actualInputTensorShape = shapeInfo.GetInputTensorShape(0);
        std::vector<uint32_t> shapeTensorDimensions = shapeInfo.GetInputTensorShape(1);
        ML_CHECK_VALID_ARGUMENT(shapeTensorDimensions.size() == 1, "Expand's shape tensor must be 1D.");
        const size_t dimCount = shapeTensorDimensions[0];

        MLOperatorTensor shapeTensor = shapeInfo.GetConstantInputTensor(1);
        const int64_t* shapeData = shapeTensor.GetData<int64_t>();

        // First element of shape tensor is how many dims to expand to.
        std::vector<uint32_t> desiredTensorShape;
        DowncastDimensions(gsl::make_span(shapeData, dimCount), /*out*/ desiredTensorShape);

        // Determine the broadcasted input shape.
        outputDimensions = OperatorHelper::BroadcastTensorShape(actualInputTensorShape, desiredTensorShape);
        
        return { std::move(EdgeShapes(outputDimensions)) };
    }
    
    std::vector<EdgeShapes> ConstantOfShapeHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        std::vector<uint32_t> outputDimensions;

        // The 'shape' tensor is a 1D tensor holding the new shape to expand to,
        // and the first element of its own shape holds how many dimensions there
        // will be for the output.

        std::vector<uint32_t> shapeTensorDimensions = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(shapeTensorDimensions.size() == 1, "ConstantOfShapeHelper's shape tensor must be 1D.");
        const size_t dimCount = shapeTensorDimensions[0];

        MLOperatorTensor shapeTensor = shapeInfo.GetConstantInputTensor(0);
        const int64_t* shapeData = shapeTensor.GetData<int64_t>();

        // First element of shape tensor is how many dims to expand to.
        std::vector<uint32_t> desiredTensorShape;
        DowncastDimensions(gsl::make_span(shapeData, dimCount), /*out*/ desiredTensorShape);

        return { std::move(EdgeShapes(desiredTensorShape)) };
    }

    std::vector<EdgeShapes> TileHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return { std::move(EdgeShapes(m_outputDimensions)) };
    }

    void ResizeHelper::Initialize(
        gsl::span<const int32_t> outputSizes
        )
    {
        assert(m_outputDimensions.empty());
        ML_CHECK_VALID_ARGUMENT(m_scales.empty() || outputSizes.empty(), "scales and roi cannot both be present.");

        const size_t rank = m_inputDimensions.size();

        if (outputSizes.empty())
        {
            // Compute output size from scales and normalized region of interest (each axis 0 to 1).
            ML_CHECK_VALID_ARGUMENT(m_scales.size() == rank, "The 'scales' parameter must have same rank as input dimensions.");
            ML_CHECK_VALID_ARGUMENT(m_regionOfInterest.empty() || m_regionOfInterest.size() == rank * 2, "The 'roi' parameter must have two values for each input dimension.");

            for (size_t i = 0; i < rank; ++i)
            {
                float scale = m_scales[i];
                ML_CHECK_VALID_ARGUMENT(scale > FLT_EPSILON, "Scale values should be positive.");
                float roiRange = m_regionOfInterest.empty() ? 1.0f : m_regionOfInterest[i + rank] - m_regionOfInterest[i];
                m_outputDimensions.push_back(gsl::narrow_cast<uint32_t>(floor(m_inputDimensions[i] * roiRange * scale)));
            }
        }
        else
        {
            // Determine scales from output / input ratio.
            ML_CHECK_VALID_ARGUMENT(outputSizes.size() == rank, "Input dimensions and 'sizes' must have same rank.");

            m_scales.resize(rank);
            for (size_t i = 0; i < rank; ++i)
            {
                float scale = float(outputSizes[i]) / std::max(m_inputDimensions[i], 1u);
                m_scales[i] = scale;
                m_outputDimensions.push_back(gsl::narrow_cast<uint32_t>(outputSizes[i]));
            }
        }
    }

    std::vector<EdgeShapes> ResizeHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return { m_outputDimensions };
    }

    void RangeHelper::Initialize(
        const MLOperatorTensor& startTensor,
        const MLOperatorTensor& limitTensor,
        const MLOperatorTensor& deltaTensor
        )
    {
        ReadScalarTensorData(startTensor, &m_valueStart, sizeof(m_valueStart));
        ReadScalarTensorData(limitTensor, &m_valueLimit, sizeof(m_valueLimit));
        ReadScalarTensorData(deltaTensor, &m_valueDelta, sizeof(m_valueDelta));
        m_tensorDataType = startTensor.GetTensorDataType();

        // The output size is a 1D tensor ranging from start up to limit,
        // where:
        //
        //  number_of_elements = max(ceil((limit - start) / delta), 0)
        //
        uint32_t totalElementCount = 0;
        if (IsFloatDataType(m_tensorDataType))
        {
            double start = CastToFloat64(m_tensorDataType, &m_valueStart);
            double limit = CastToFloat64(m_tensorDataType, &m_valueLimit);
            double delta = CastToFloat64(m_tensorDataType, &m_valueDelta);
            totalElementCount = gsl::narrow_cast<uint32_t>(std::max(ceil((limit - start) / delta), 0.0));
        }
        else
        {
            int64_t start = CastToInt64(m_tensorDataType, &m_valueStart);
            int64_t limit = CastToInt64(m_tensorDataType, &m_valueLimit);
            int64_t delta = CastToInt64(m_tensorDataType, &m_valueDelta);
            int64_t range = limit - start;
            totalElementCount = gsl::narrow_cast<uint32_t>(std::max((range / delta) + (range % delta != 0), int64_t(0)));
        }
        m_outputDimensions.push_back(totalElementCount);
    }

    std::vector<EdgeShapes> RangeHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return { m_outputDimensions };
    }

    std::vector<EdgeShapes> OneHotHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return { std::move(EdgeShapes(m_outputDimensions)) };
    }

} // namespace OperatorHelper

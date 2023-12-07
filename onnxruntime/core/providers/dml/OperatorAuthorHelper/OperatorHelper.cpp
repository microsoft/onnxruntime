// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "precomp.h"
#include "OperatorHelper.h"
#include "core/providers/common.h"
#include "core/providers/cpu/math/matmul_helper.h"

namespace OperatorHelper
{
    template <typename T = uint32_t>
    T DivideRoundUp(T x, T y)
    {
        assert(y != 0);
        return (x + y - 1) / y;
    }

    bool ContainsEmptyDimensions(gsl::span<const DimensionType> dimensions)
    {
        return std::find(dimensions.begin(), dimensions.end(), 0u) != dimensions.end();
    }

    // Convert any negative axis into an absolute axis relative to the back end.
    // So given 3 dimensions, a -1 refers to axis 2, and -3 to axis 0.
    uint32_t HandleNegativeAxis(int32_t signedOnnxAxis, uint32_t dimCount, bool validateAxis)
    {
        if (signedOnnxAxis < 0)
        {
            signedOnnxAxis += dimCount;
        }
        uint32_t absoluteAxis = gsl::narrow_cast<uint32_t>(signedOnnxAxis);
        ML_CHECK_VALID_ARGUMENT(!validateAxis || absoluteAxis < dimCount);
        return absoluteAxis;
    }

    void HandleNegativeAxes(gsl::span<int32_t> onnxAxes, uint32_t dimCount)
    {
        for (int32_t& axis : onnxAxes)
        {
            axis = HandleNegativeAxis(axis, dimCount);
        }
    }

    void HandleEmptyAxes(
        /*inout*/std::vector<int32_t>& axes,
        gsl::span<const uint32_t> inputShape,
        bool treatEmptyAsNop
        )
    {
        // If axes is not specified, reduce over all the dimensions.
        // If empty axes should be treated as a nop, then just leave them as-is.
        if (axes.empty() && !treatEmptyAsNop)
        {
            axes.resize(inputShape.size());
            std::iota(axes.begin(), axes.end(), 0);
        }
    }

    float CastFloat16ToFloat32(uint16_t input)
    {
        // Promote float16m10e5s1 to float32m23e8s1.
        // Note this works on machines of both ascending and descending byte
        // endianness, so long as float32 and uint32 endianness match.
        // It does not work for a few abberant architectures which store
        // float32 and uint32 with opposite endianness.

        const uint32_t float16unsignedValueMask = 0x7FFF;
        const uint32_t float16signMask          = 0x8000;
        const uint32_t float16exponentMask      = 0x7C00;
        const uint32_t float32exponentMask      = 0x7F800000;

        uint32_t float16unsignedValue = input & float16unsignedValueMask;
        uint32_t float16sign          = input & float16signMask;
        uint32_t float16exponent      = input & float16exponentMask;

        // Shift mantissa bits left (23 - 10 = 13).
        // Adjust exponent bias (127 - 15 = 112, 112 << 23 == 0x38000000).
        // Move sign bit to float32 MSB (32 - 16 = 16).
        uint32_t float32unsignedValue = (float16unsignedValue << 13) + 0x38000000;
        uint32_t float32sign          = float16sign << 16;
        uint32_t result               = (float16exponent == 0) ? (float32unsignedValue & ~float32exponentMask) : // Denormal
                                        (float16exponent == float16exponentMask) ? (float32unsignedValue | float32exponentMask) : // Infinity
                                        float32unsignedValue; // Any other normal value
        result |= float32sign;

        return reinterpret_cast<float&>(result);
    }

    #pragma warning(push)
    #pragma warning(disable:4702)
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
        default:
            ML_INVALID_ARGUMENT("Unknown MLOperatorTensorDataType.");
            return 0;
        };
    }
    #pragma warning(pop)

    #pragma warning(push)
    #pragma warning(disable:4702)
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
        case MLOperatorTensorDataType::Float16:    return static_cast<double>(CastFloat16ToFloat32(*reinterpret_cast<const uint16_t*>(p)));
        case MLOperatorTensorDataType::Double:     return static_cast<double>(*reinterpret_cast<const double*>(p));
        case MLOperatorTensorDataType::UInt32:     return static_cast<double>(*reinterpret_cast<const uint32_t*>(p));
        case MLOperatorTensorDataType::UInt64:     return static_cast<double>(*reinterpret_cast<const uint64_t*>(p));
        case MLOperatorTensorDataType::Complex64:  return static_cast<double>(*reinterpret_cast<const float*>(p)); // Read the real component.
        case MLOperatorTensorDataType::Complex128: return static_cast<double>(*reinterpret_cast<const double*>(p)); // Read the real component.
        case MLOperatorTensorDataType::Undefined:
        default:
            ML_INVALID_ARGUMENT("Unknown MLOperatorTensorDataType.");
            return 0.0;
        };
    }
    #pragma warning(pop)

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
        result.resize(elementCount);

        switch (tensor.GetTensorDataType())
        {
        case MLOperatorTensorDataType::Float16:
            {
                const uint16_t* data = reinterpret_cast<const uint16_t*>(tensor.GetByteData());
                std::transform(data, data + elementCount, result.begin(), CastFloat16ToFloat32);
            }
            break;

        case MLOperatorTensorDataType::/*Float32*/Float:
            {
                const float* data = tensor.GetData<float>();
                result.assign(data, data + elementCount);
            }
            break;

        case MLOperatorTensorDataType::/*Float64*/Double:
            {
                const double* data = tensor.GetData<double>();
                std::transform(data, data + elementCount, result.begin(), [](auto v) {return static_cast<float>(v); });
            }
            break;

        case MLOperatorTensorDataType::Int32:
            {
                const int32_t* data = tensor.GetData<int32_t>();
                std::transform(data, data + elementCount, result.begin(), [](auto v) {return static_cast<float>(v); });
            }
            break;

        case MLOperatorTensorDataType::UInt32:
            {
                const uint32_t* data = tensor.GetData<uint32_t>();
                std::transform(data, data + elementCount, result.begin(), [](auto v) {return static_cast<float>(v); });
            }
            break;

        case MLOperatorTensorDataType::Int64:
            {
                const int64_t* data = tensor.GetData<int64_t>();
                std::transform(data, data + elementCount, result.begin(), [](auto v) {return static_cast<float>(v); });
            }
            break;

        case MLOperatorTensorDataType::UInt64:
            {
                const uint64_t* data = tensor.GetData<uint64_t>();
                std::transform(data, data + elementCount, result.begin(), [](auto v) {return static_cast<float>(v); });
            }
            break;

        default:
            ML_INVALID_ARGUMENT("Expecting CPU local tensor of type float32.");
            break;
        }
    }

    template <typename T>
    void DowncastDimensions(gsl::span<const T> inputDimensions, std::vector<DimensionType>& outputDimensions)
    {
        outputDimensions.reserve(inputDimensions.size());
        outputDimensions.clear();

        for (T dim : inputDimensions)
        {
            outputDimensions.push_back(gsl::narrow_cast<uint32_t>(std::clamp<T>(dim, INT32_MIN, INT32_MAX)));
        }
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
        const KernelArgs& args,
        bool isNhwc
    )
    {
        ML_CHECK_VALID_ARGUMENT(gsl::narrow_cast<uint32_t>(inputDimensions.size()) >= args.spatialDimensionCount);
        int dimOffset = isNhwc ? 1 : gsl::narrow_cast<int>(inputDimensions.size()) - args.spatialDimensionCount;

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
    KernelArgs InitializeGlobalKernel(
        const MLOperatorAttributes& kernelInfo,
        gsl::span<const DimensionType> inputDimensions)
    {
        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() > NonspatialDimensionCount); // Must be at least 1D convolution (in 3D tensor)
        uint32_t spatialDimensionCount = gsl::narrow_cast<uint32_t>(inputDimensions.size()) - NonspatialDimensionCount;
        ML_CHECK_VALID_ARGUMENT(spatialDimensionCount <= NcdhwSpatialDimensionCount); // Support up to 3D convolution (in 5D tensor).

        KernelArgs args(spatialDimensionCount);
        args.useCeilingOutputShape = kernelInfo.GetOptionalAttribute<bool>(AttrName::CeilMode, 0);
        args.channelsLast = kernelInfo.GetOptionalAttribute<bool>(AttrName::ChannelsLast, 0);
        // For Global Pooling, kernel size equal to the spatial dimension of input tensor
        // NHWC layout need to offset by one dim to acount for channel placed at the end
        int dimOffset = args.channelsLast ? 1 : 0;

        for (size_t dim = 0; dim < spatialDimensionCount; ++dim)
        {
            args.strides[dim] = 1;
            args.dilations[dim] = 1;
            args.startPadding[dim] = 0;
            args.endPadding[dim] = 0;
            args.windowSize[dim] = gsl::narrow_cast<uint32_t>(inputDimensions[inputDimensions.size() - spatialDimensionCount + dim - dimOffset]);
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

        std::string autoPadStr = kernelInfo.GetOptionalAttribute<std::string>(AttrName::AutoPad, AttrValue::NotSet);
        auto autoPad = onnxruntime::StringToAutoPadType(autoPadStr);

        if (autoPad == onnxruntime::AutoPadType::NOTSET)
        {
            // Use the pad values in the pads argument.
            std::vector<int> pads = kernelInfo.GetOptionalAttributeVectorInt32(AttrName::Pads);

            // if pads are not specified, assume all pad values are 0
            if (pads.empty())
            {
                pads.resize(2 * static_cast<uint64_t>(spatialDimensionCount));
            }

            ML_CHECK_VALID_ARGUMENT(pads.size() >= 2 * spatialDimensionCount);

            std::copy(pads.begin(), pads.begin() + spatialDimensionCount, args.startPadding);
            std::copy(pads.begin() + spatialDimensionCount, pads.begin() + spatialDimensionCount * 2, args.endPadding);
        }
        else if (autoPad == onnxruntime::AutoPadType::VALID)
        {
            std::fill(args.startPadding, args.startPadding + spatialDimensionCount, 0);
            std::fill(args.endPadding, args.endPadding + spatialDimensionCount, 0);
        }
        else
        {
            args.autoPad = true;
            args.autoPadSameUpper = autoPad == onnxruntime::AutoPadType::SAME_UPPER;
            assert(args.autoPadSameUpper || autoPad == onnxruntime::AutoPadType::SAME_LOWER);
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
        args.channelsLast = kernelInfo.GetOptionalAttribute<bool>(AttrName::ChannelsLast, 0);

        return args;
    }

    void ResolveAutoPadding(
        KernelArgs& args,
        gsl::span<const DimensionType> inputDimensions,
        bool isNhwc
    )
    {
        if (!args.autoPad)
        {
            return;
        }

        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() > NonspatialDimensionCount); // Must be at least 1D convolution (in 3D tensor)
        uint32_t spatialDimensionCount = gsl::narrow_cast<uint32_t>(inputDimensions.size()) - NonspatialDimensionCount;
        ML_CHECK_VALID_ARGUMENT(spatialDimensionCount <= NcdhwSpatialDimensionCount); // Support up to 3D convolution (in 5D tensor).

        ML_CHECK_VALID_ARGUMENT(!isNhwc || inputDimensions.size() == 4);

        const int dimOffset = isNhwc ? 1 : gsl::narrow_cast<int>(inputDimensions.size()) - spatialDimensionCount;

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

    void FusedMatMulShapeMapping(
        std::vector<DimensionType>& inputShape0,
        std::vector<DimensionType>& inputStride0,
        std::vector<DimensionType>& inputShape1,
        std::vector<DimensionType>& inputStride1,
        std::vector<DimensionType>& outputShape)
    {
        // Get the padded input shapes, and undo the effect of padding removal from the output shape.
        if (inputShape1.size() == 1)
        {
            inputShape1.push_back(1);
            inputStride1.push_back(0);
            outputShape.push_back(1);
        }

        if (inputShape0.size() == 1)
        {
            inputShape0.insert(inputShape0.begin(), 1);
            inputStride0.insert(inputStride0.begin(), 0);
            outputShape.insert(outputShape.end() - 1, 1);
        }

        auto broadcastedRank = std::max(inputShape0.size(), inputShape1.size());
        inputShape0.insert(inputShape0.begin(), (broadcastedRank - inputShape0.size()), 1);
        inputStride0.insert(inputStride0.begin(), (broadcastedRank - inputStride0.size()), 0);

        inputShape1.insert(inputShape1.begin(), (broadcastedRank - inputShape1.size()), 1);
        inputStride1.insert(inputStride1.begin(), (broadcastedRank - inputStride1.size()), 0);

        BroadcastTensorShapeAndSetStrides(
            gsl::make_span(inputShape0.data(), broadcastedRank - 2),
            gsl::make_span(inputStride0.data(), broadcastedRank - 2),
            gsl::make_span(inputShape1.data(), broadcastedRank - 2),
            gsl::make_span(inputStride1.data(), broadcastedRank - 2)
        );
    }

    std::pair<std::vector<uint32_t>, std::vector<uint32_t>> GetFusedMatMulSizesAndStrides(
        gsl::span<const uint32_t> sizes,
        int32_t transBatch,
        int32_t transpose)
    {
        const uint32_t dimensionCount = gsl::narrow_cast<uint32_t>(sizes.size());
        std::vector<uint32_t> newStrides(dimensionCount);
        std::vector<uint32_t> newSizes(sizes.begin(), sizes.end());

        // Calculate packed strides.
        uint32_t stride = 1;
        for (int i = dimensionCount - 1; i >= 0; i--)
        {
            newStrides[i] = stride;
            stride *= sizes[i];
        }

        // According to contrib ops shape inference
        // https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/core/graph/contrib_ops/contrib_defs.cc#L215
        // `transBatch` needs to be applied first and then `transpose`.
        if (transBatch)
        {
            ML_CHECK_VALID_ARGUMENT(dimensionCount > 2,
                "FusedMatMul operator: Tensor size should be more than 2, if attribute transBatch is true");

            std::rotate(newSizes.begin(), newSizes.begin() + 1, newSizes.end() - 1);
            std::rotate(newStrides.begin(), newStrides.begin() + 1, newStrides.end() - 1);
        }

        if (transpose && dimensionCount > 1)
        {
            std::swap(newStrides[dimensionCount - 2], newStrides[dimensionCount - 1]);
            std::swap(newSizes[dimensionCount - 2], newSizes[dimensionCount - 1]);
        }

        return std::make_pair(newSizes, newStrides);
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

            // 0-sized dimensions indicate an empty tensor and shouldn't be broadcasted to higher dimensions
            if (inDimension0 == 0 || inDimension1 == 0)
            {
                inDimension0 = 0;
                inDimension1 = 0;
            }

            ML_CHECK_VALID_ARGUMENT((inDimension0 == inDimension1) || (inDimension0 == 1) || (inDimension1 == 1));
            *outDimIter = std::max(inDimension0, inDimension1);
        }

        return outputShape;
    }

    void BroadcastTensorShapeAndSetStrides(
        gsl::span<DimensionType> inputShape0,
        gsl::span<DimensionType> inputStride0,
        gsl::span<DimensionType> inputShape1,
        gsl::span<DimensionType> inputStride1
        )
    {
        if (inputShape0 != inputShape1)
        {
            ML_CHECK_VALID_ARGUMENT(
                inputShape0.size() == inputShape1.size() &&
                inputShape0.size() == inputStride0.size() &&
                inputStride0.size() == inputStride1.size(),
                "Size of inputShape0, inputStride0, inputShape1 and inputStride1 should be same while broadcasting");

            // Walk backwards through both input shapes and broadcast each dimension,
            // ignoring the last 2 dimensions (matrix dimensions).
            auto rank = inputShape0.size();
            auto inDim0Iter = inputShape0.rbegin();
            auto inDim1Iter = inputShape1.rbegin();

            auto inStride0Iter = inputStride0.rbegin();
            auto inStride1Iter = inputStride1.rbegin();

            while (rank-- > 0)
            {
                DimensionType inDimension0 = *inDim0Iter;
                DimensionType inStride0 = *inStride0Iter;

                DimensionType inDimension1 = *inDim1Iter;
                DimensionType inStride1 = *inStride1Iter;

                // 0-sized dimensions indicate an empty tensor and shouldn't be broadcasted to higher dimensions.
                if (inDimension0 == 0 || inDimension1 == 0)
                {
                    inDimension0 = 0;
                    inDimension1 = 0;
                }

                ML_CHECK_VALID_ARGUMENT((inDimension0 == inDimension1) || (inDimension0 == 1) || (inDimension1 == 1));
                auto broadcastedDimension = std::max(inDimension0, inDimension1);

                inputShape0[rank] = broadcastedDimension;
                inputShape1[rank] = broadcastedDimension;
                inputStride0[rank] = (broadcastedDimension != inDimension0) ? 0 : inStride0;
                inputStride1[rank] = (broadcastedDimension != inDimension1) ? 0 : inStride1;

                ++inDim0Iter;
                ++inStride0Iter;
                ++inDim1Iter;
                ++inStride1Iter;
            }
        }
    }

    void ConvolutionHelperBase::InitializeKernelAndShapes(const IShapeInformationAdapter& shapeInformation)
    {
        const std::vector<DimensionType> inputDimensions = shapeInformation.GetInputTensorShape(m_inputTensorIndex);
        const std::vector<DimensionType> filterDims = shapeInformation.GetInputTensorShape(m_filterTensorIndex);

        ML_CHECK_VALID_ARGUMENT(
            inputDimensions.size() >= 3 && inputDimensions.size() <= 5,
            "Input dimensions must be: 3, 4, 5."
        );

        ResolvingPadding(inputDimensions);

        m_outputShapes.resize(1);
        m_outputShapes[0] = InitializeKernelOutputDimensions(inputDimensions, m_kernel, m_isNhwc);

        if (m_isNhwc)
        {
            m_outputShapes[0].GetShape()[static_cast<uint32_t>(NhwcInputDims::C)] = filterDims[K];
        }
        else
        {
            m_outputShapes[0].GetShape()[C] = filterDims[K];
        }
    }

    void ConvolutionHelperBase::InitializeKernelAndShapesTransposed(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        bool hasDynamicPads
        )
    {
        auto& attributes = kernelInformation.GetAttributes();
        std::vector<int> outputShape = attributes.GetOptionalAttributeVectorInt32(AttrName::OutputShape);
        if (!outputShape.empty())
        {
            ML_CHECK_VALID_ARGUMENT(
                outputShape.size() >= m_kernel.spatialDimensionCount,
                "The output shape must equal the number of spatial dimensions"
            );
        }

        const std::vector<DimensionType> inputDimensions = shapeInformation.GetInputTensorShape(m_inputTensorIndex);
        const std::vector<DimensionType> filterDims = shapeInformation.GetInputTensorShape(m_filterTensorIndex);

        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() > NonspatialDimensionCount, "Input dimensions must be >= 3");

        if (hasDynamicPads)
        {
            MLOperatorTensor padsTensor = kernelInformation.GetConstantInputTensor(2);
            const std::vector<uint32_t>& padsTensorDimensions = padsTensor.GetShape();
            ML_CHECK_VALID_ARGUMENT(padsTensorDimensions.size() == 1, "Pads dimensions must equal 1");
            const size_t dimCount = padsTensorDimensions[0];
            ML_CHECK_VALID_ARGUMENT(dimCount == 2 * NchwSpatialDimensionCount, "Pads count must equal 4");
            const int64_t* padsData = padsTensor.GetData<int64_t>();

            for (size_t i = 0; i < dimCount; ++i)
            {
                ML_CHECK_VALID_ARGUMENT(padsData[i] >= 0, "Padding values must be greater than or equal to 0");
                if (i < dimCount / 2)
                {
                    m_kernel.startPadding[i] = gsl::narrow_cast<uint32_t>(padsData[i]);
                }
                else
                {
                    m_kernel.endPadding[i - dimCount/2] = gsl::narrow_cast<uint32_t>(padsData[i]);
                }
            }
        }
        else
        {
            ResolvingPadding(inputDimensions);
        }

        m_outputShapes.resize(1);
        m_outputShapes[0] = InitializeKernelOutputDimsTranspose(inputDimensions, m_kernel);
        static_assert(C < NonspatialDimensionCount);
        assert(m_outputShapes[0].GetShape().size() > C);
        m_outputShapes[0].GetShape()[C] = filterDims[C] * m_groupCount;

        if (!outputShape.empty())
        {
            // Start padding, end padding, and output padding are all ignored if output shape is set.
            std::fill(m_kernel.outputPadding, m_kernel.outputPadding + m_kernel.spatialDimensionCount, 0);

            if (outputShape.size() > 2)
            {
                ML_CHECK_VALID_ARGUMENT(outputShape[outputShape.size() - 3] == gsl::narrow_cast<int>(m_outputShapes[0].GetShape()[C]), "Output channel must be equivalent to filter channel.");
            }

            for (size_t i = 0; i < m_kernel.spatialDimensionCount; ++i)
            {
                size_t outputIndex = outputShape.size() - m_kernel.spatialDimensionCount + i;
                ML_CHECK_VALID_ARGUMENT(outputShape[outputIndex] >= gsl::narrow_cast<int>(inputDimensions[H + i]), "Output dimension cannot be smaller than input dimension.");
                m_outputShapes[0].GetShape()[H + i] = outputShape[outputIndex];
            }

            const int dimOffset = gsl::narrow_cast<int>(inputDimensions.size() - m_kernel.spatialDimensionCount);

            for (size_t i = 0; i < m_kernel.spatialDimensionCount; ++i)
            {
                int stride = m_kernel.strides[i];
                int windowSize = m_kernel.windowSize[i];

                // Compute padding such that in reverse order, the logical input (m_outputShapes below) is fully defined
                // for a convolution over the logical output region (inputDimensions below).
                //
                // The padding required is the first windowSize element (for the first logical output element),
                // plus (logicalOutput - 1) steps of stride (the distance between each windowed set of logical
                // input elements), minus the actual logical input size.
                int paddings = gsl::narrow_cast<int>((inputDimensions[i + dimOffset] - 1) * stride + windowSize - m_outputShapes[0].GetShape()[i + dimOffset]);
                paddings = std::max<int>(0, paddings);

                m_kernel.startPadding[i] = m_kernel.autoPadSameUpper ? paddings / 2 : (paddings + 1) / 2;
                m_kernel.endPadding[i] = paddings - m_kernel.startPadding[i];
            }
        }
    }

    std::vector<EdgeShapes> ConvolutionHelperBase::GetOutputShapes(const MLShapeInferenceContext& shapeInformation) const
    {
        ORT_UNUSED_PARAMETER(shapeInformation);
        return m_outputShapes;
    }

    void ConvolutionHelperBase::ResolvingPadding(gsl::span<const DimensionType> inputDimensions)
    {
        ResolveAutoPadding(m_kernel, inputDimensions, m_isNhwc);
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
        IKernelInformationAdapter const& kernelInformation,
        IShapeInformationAdapter const& shapeInformation,
        uint32_t opsetVersion
        )
    {
        auto& operatorAttributes = kernelInformation.GetAttributes();
        if (opsetVersion >= 13) // Axes are a dynamic input parameter.
        {
            // The tensor is optional, which if empty, means to default to equal splits.
            if (kernelInformation.IsInputValid(1))
            {
                ReadCpuLocalTensorIntoInt32(kernelInformation.GetConstantInputTensor(1), /*out*/ m_split);
            }
        }
        else // Axes were a constant attribute parameter.
        {
            m_split = operatorAttributes.GetOptionalAttributeVectorInt32(AttrName::Split);
        }

        const std::vector<DimensionType> inputDimensions = shapeInformation.GetInputTensorShape(0);

        const uint32_t inputDimCount = gsl::narrow_cast<int32_t>(inputDimensions.size());
        const uint32_t axis = operatorAttributes.GetOptionalAttribute<int32_t>(AttrName::Axis, 0);
        m_axis = static_cast<int>(HandleNegativeAxis(axis, inputDimCount));

        if (opsetVersion >= 18) // num_outputs attribute is only defined in opset18.
        {
            const uint32_t numOutputs = operatorAttributes.GetOptionalAttribute<int32_t>(AttrName::NumOutputs, 0);
            if (numOutputs > 0)
            {
                ML_CHECK_VALID_ARGUMENT(m_split.size() == 0);
                auto inputSizeAlongAxis = inputDimensions.at(m_axis);
                auto outputSizeAlongAxis = DivideRoundUp(inputSizeAlongAxis, numOutputs);
                m_split.resize(numOutputs, outputSizeAlongAxis);
                // Every output has the same size except potentially the last one, which may be smaller.
                m_split.back() = static_cast<int>(inputSizeAlongAxis - (numOutputs - 1) * outputSizeAlongAxis);
            }
            else
            {
                // There is no num_outputs attribute set, so splits must be set.
                ML_CHECK_VALID_ARGUMENT(m_split.size() > 0);
            }
        }
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

    void SliceHelper::Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        uint32_t opsetVersion
        )
    {
        auto& attributes = kernelInformation.GetAttributes();
        std::vector<DimensionType> inputDimensions = shapeInformation.GetInputTensorShape(0);

        std::vector<int32_t> starts;
        std::vector<int32_t> ends;
        std::vector<int32_t> axes;
        std::vector<int32_t> steps;

        if (opsetVersion >= 10)
        {
            // Get starts, ends, optional axes, and optional steps from constant tensor inputs.
            ReadCpuLocalTensorIntoInt32(kernelInformation.GetConstantInputTensor(1), /*out*/ starts);
            ReadCpuLocalTensorIntoInt32(kernelInformation.GetConstantInputTensor(2), /*out*/ ends);
            if (kernelInformation.IsInputValid(3))
            {
                ReadCpuLocalTensorIntoInt32(kernelInformation.GetConstantInputTensor(3), /*out*/ axes);
            }
            if (kernelInformation.IsInputValid(4))
            {
                ReadCpuLocalTensorIntoInt32(kernelInformation.GetConstantInputTensor(4), /*out*/ steps);
            }
        }
        else if (opsetVersion >= 7)
        {
            // Read starts, ends, and axes from attributes.
            starts = attributes.GetOptionalAttributeVectorInt32(AttrName::Starts);
            ends = attributes.GetOptionalAttributeVectorInt32(AttrName::Ends);
            axes = attributes.GetOptionalAttributeVectorInt32(AttrName::Axes);
        }

        const uint32_t inputDimensionCount = gsl::narrow_cast<int32_t>(inputDimensions.size());
        HandleNegativeAxes(/*inout*/ axes, inputDimensionCount);

        ML_CHECK_VALID_ARGUMENT(starts.size() == ends.size(), "'starts' must equal 'ends' in size.");
        ML_CHECK_VALID_ARGUMENT(steps.empty() || steps.size() == axes.size(), "'steps' must equal 'axes' in size, or 'steps' must be empty.");
        ML_CHECK_VALID_ARGUMENT(axes.empty() || starts.size() == axes.size(), "'axes' must equal 'starts' in size, or 'axes' must be empty.");

        m_outputDimensions.assign(inputDimensions.begin(), inputDimensions.end());
        m_offsets.resize(m_outputDimensions.size());
        m_sizes.resize(m_outputDimensions.size());
        m_strides.resize(m_outputDimensions.size(), 1); // Default initialize to all steps to 1's.

        // Set initial defaults lest 'starts' and 'ends' arrays are shorter than the dimension count.
        std::copy(inputDimensions.begin(), inputDimensions.begin() + m_outputDimensions.size(), m_sizes.begin());

        // Clamp selected dimensions to given 'starts' and 'ends'.
        for (int i = 0, ci = gsl::narrow_cast<int>(starts.size()); i < ci; ++i)
        {
            int dimIndex = axes.empty() ? i : axes[i];
            int stride = steps.empty() ? 1 : steps[i];
            ML_CHECK_VALID_ARGUMENT(static_cast<size_t>(dimIndex) < static_cast<size_t>(inputDimensions.size()), "'axes' must be valid with within actual input dimensions.");
            ML_CHECK_VALID_ARGUMENT(stride != 0, "'steps' must not be 0.");

            // Positive values are offsets from 0.
            // Negative values are offsets from back of the dimension's size.
            // INT_MIN is a special value in ONNX which means to treat it as the smallest
            // possible value, rather than the usual reversed from-the-back semantics.
            int dim = gsl::narrow_cast<int>(inputDimensions[dimIndex]);
            int start = (starts[i] < 0 && starts[i] > INT_MIN) ? (starts[i] + dim) : starts[i];
            int end = (ends[i] < 0 && starts[i] > INT_MIN) ? (ends[i] + dim) : ends[i];

            // For negative strides, the ONNX start and end values are off-by-one.
            // So fix them such that the start value remains the minimum extent
            // of the slice window, and end remains the maximum exclusive extent.
            if (stride < 0)
            {
                std::swap(start, end);
                start += (start < INT_MAX) ? 1 : 0; // Avoid overflow wrap.
                end += (end < INT_MAX) ? 1 : 0;
            }

            // Clamp the dimensions to the slice extents.
            // Clamp negative numbers to 0, per case test_slice_start_out_of_bounds.
            start = std::max(start, 0);
            end = std::min(end, dim);
            int size = std::max(end - start, 0);

            // Set the input window offsets/sizes, and compute output size based on input
            // window size (rounding up).
            // e.g. a window size 13 and step 3 yields 5 output elements.
            int absoluteStride = abs(stride);
            m_outputDimensions[dimIndex] = (size / absoluteStride) + (size % absoluteStride != 0);
            m_offsets[dimIndex] = start;
            m_strides[dimIndex] = stride;
            m_sizes[dimIndex] = gsl::narrow_cast<uint32_t>(size);
        }
    }

    std::vector<EdgeShapes> SliceHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return { m_outputDimensions };
    }

    void PaddingHelper::Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        uint32_t opsetVersion
        )
    {
        auto& attributes = kernelInformation.GetAttributes();

        std::vector<int32_t> padding;
        if (opsetVersion >= 11)
        {
            MLOperatorTensor padsTensor = kernelInformation.GetConstantInputTensor(1);
            ReadCpuLocalTensorIntoInt32(padsTensor, /*out*/ padding);
        }
        else
        {
            padding = attributes.GetOptionalAttributeVectorInt32(AttrName::Pads);
        }

        ML_CHECK_VALID_ARGUMENT(padding.size() % 2 == 0, "Padding must be even count, including begin/end pairs.");
        std::vector<uint32_t> inputShape = shapeInformation.GetInputTensorShape(0);
        uint32_t dimCount = gsl::narrow_cast<uint32_t>(inputShape.size());
        m_startPadding.resize(dimCount, 0);
        m_endPadding.resize(dimCount, 0);
        std::vector<int32_t> axes;

        // Handle possible axes input
        if (opsetVersion >= 18)
        {
            if (kernelInformation.IsInputValid(3))
            {
                ReadCpuLocalTensorIntoInt32(kernelInformation.GetConstantInputTensor(3), /*out*/ axes);
            }
            HandleEmptyAxes(axes, inputShape, false);
            ML_CHECK_VALID_ARGUMENT(axes.size() * 2 == padding.size(), "The number of elements in padding should be 2 times the number of axes.");
            HandleNegativeAxes(axes, dimCount);
        }
        else
        {
            HandleEmptyAxes(axes, inputShape, false);
        }

        uint32_t numAxes = gsl::narrow_cast<uint32_t>(axes.size());
        for (int32_t i = 0; i < axes.size(); i++)
        {
            auto xi_begin = padding[i];
            auto xi_end = padding[i+axes.size()];
            m_startPadding[axes[i]] = xi_begin;
            m_endPadding[axes[i]] = xi_end;
        }
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
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation
        )
    {
        auto& operatorAttributes = kernelInformation.GetAttributes();
        std::vector<DimensionType> inputDimensions = shapeInformation.GetInputTensorShape(0);

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
        ML_CHECK_VALID_ARGUMENT(outDimCount >= 0);

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
        int32_t batchCount = m_batchCount;

        // Determine the number of output dimensions.
        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() >= 1);
        ML_CHECK_VALID_ARGUMENT(indicesDimensions.size() >= 1);
        ML_CHECK_VALID_ARGUMENT(static_cast<int64_t>(inputDimensions.size()) > static_cast<int64_t>(batchCount));
        ML_CHECK_VALID_ARGUMENT(static_cast<int64_t>(indicesDimensions.size()) > static_cast<int64_t>(batchCount));
        const uint32_t numberOfCoordinatesPerIndex = indicesDimensions.back();
        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() >= batchCount + numberOfCoordinatesPerIndex);
        const uint32_t numberOfOutputDimensionsFromInput = static_cast<uint32_t>(inputDimensions.size()) - batchCount - numberOfCoordinatesPerIndex;
        const uint32_t numberOfOutputDimensionsFromIndices = static_cast<uint32_t>(indicesDimensions.size()) - batchCount - 1; // Strip off last dimension.
        uint32_t outputDimensionCount = gsl::narrow_cast<uint32_t>(batchCount + numberOfOutputDimensionsFromIndices + numberOfOutputDimensionsFromInput);
        ML_CHECK_VALID_ARGUMENT(outputDimensionCount > 0);

        // Form the full expected size by concatenating fragments:
        // 1 - batch count
        // 2 - prefix part of the indices tensor shape
        // 3 - suffix of the input tensor shape.
        std::vector<DimensionType> outputDimensions;
        outputDimensions.assign(inputDimensions.begin(), inputDimensions.begin() + batchCount);
        outputDimensions.insert(outputDimensions.end(), indicesDimensions.begin() + batchCount, indicesDimensions.end() - 1);
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

    void ReduceHelperBase::Initialize(
        IKernelInformationAdapter const& kernelInformation,
        IShapeInformationAdapter const& shapeInformation,
        bool usingMultipleAxes
        )
    {
        auto& attributes = kernelInformation.GetAttributes();
        m_keepDims = attributes.GetOptionalAttribute<int32_t>(AttrName::KeepDims, 1);
        m_selectLastIndex = attributes.GetOptionalAttribute<int32_t>(AttrName::SelectLastIndex, 0);
        m_noopWithEmptyAxes = attributes.GetOptionalAttribute<int32_t>(AttrName::NoopWithEmptyAxes, 0);

        if (usingMultipleAxes) // Read full axis list. e.g. ReduceSum.
        {
            if (kernelInformation.IsInputValid(1)) // Axes are from a dynamic input parameter.
            {
                ReadCpuLocalTensorIntoInt32(kernelInformation.GetConstantInputTensor(1), /*out*/ m_axes);
            }
            else // Axes were a constant attribute parameter.
            {
                m_axes = attributes.GetOptionalAttributeVectorInt32(AttrName::Axes);
            }
        }
        else // Only read a single axis. e.g. ArgMin/ArgMax.
        {
            int axis = attributes.GetOptionalAttribute<int32_t>(AttrName::Axis, 0);
            m_axes.push_back(axis);
        }

        std::vector<uint32_t> inputShape = shapeInformation.GetInputTensorShape(0);
        HandleNegativeAxes(/*inout*/ m_axes, gsl::narrow_cast<uint32_t>(inputShape.size()));
        HandleEmptyAxes(/*inout*/ m_axes, inputShape, bool(m_noopWithEmptyAxes));
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

    void EinSumHelper::Initialize()
    {
        ParseEquationComponents();
        m_recognizedOperatorType = DetermineRecognizedOperatorType();
    }

    void EinSumHelper::ParseEquationComponents()
    {
        // Parse an equation like 'ij,jk->ik' into components {ij, jk, ik} mapping letters to
        // numeric indices {(0,1}, {1,2}, {0,2}}. The last component is the output.

        std::map<char, uint32_t> labelMap;
        std::set<char> repeatedLabels;

        uint32_t currentLabelIndex = 0;
        Component currentComponent = {};
        bool foundOutput = false;
        bool reachedEnd = false;

        // Read first to last character in equation, looking for letters, commas, and one arrow.
        for (char* token = m_equation.data(); !reachedEnd; ++token)
        {
            char ch = *token;

            // Only ASCII letters are valid subscript symbols in numpy.einsum().
            if ((ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z'))
            {
                // Check whether label already has an index.
                const auto [i, inserted] = labelMap.insert({ch, currentLabelIndex});
                if (inserted)
                {
                    ML_CHECK_VALID_ARGUMENT(!foundOutput, "Found label in equation output not matching any label from inputs.")
                    ++currentLabelIndex; // New label found.
                }
                else if (!foundOutput)
                {
                    // If label in input already found earlier, then keep track of this later
                    // to generate the default output in case one is not specified.
                    repeatedLabels.insert(ch);
                }
                m_labelIndices.push_back(i->second);
            }
            else if (ch == ' ')
            {
                // Ignore spaces.
            }
            else
            {
                currentComponent.labelIndexEnd = static_cast<uint32_t>(m_labelIndices.size());
                m_components.push_back(currentComponent);
                currentComponent.labelIndexBegin = currentComponent.labelIndexEnd;

                switch (ch)
                {
                case ',':
                    // Note it's valid for 2 commas be adjacent, which indicates a scalar and generates
                    // an empty component.
                    break;

                case '-': // Start of "->" (must be atomic, no space between them).
                    ++token; // Skip '-'.
                    ML_CHECK_VALID_ARGUMENT(*token == '>', "Expected '->' for output.")
                    ML_CHECK_VALID_ARGUMENT(foundOutput == false, "Only one output arrow '->' is valid.")
                    foundOutput = true;
                    break;

                case '.':
                    // Ellipsis is unsupported. Leave recognized operator as None, deferring to another EP.
                    m_components.clear();
                    return;

                case '\0':
                    reachedEnd = true;
                    break; // End of string.

                default:
                    ML_INVALID_ARGUMENT("Unsupported character in equation string. Must be a-z, A-Z, ',', or '->'.");
                }
            }
        }

        if (!foundOutput)
        {
            // If no explicit output was given, generate an implicit output by ordering all the
            // labels in alphabetic order (by ASCII value consistent with numpy, so Z < a).
            // Exclude any labels that occurred more than once, as these cancel out.

            for (auto i : labelMap)
            {
                if (repeatedLabels.count(i.first) == 0)
                {
                    m_labelIndices.push_back(i.second);
                }
            }

            // Push the final component, which is the output.
            currentComponent.labelIndexEnd = static_cast<uint32_t>(m_labelIndices.size());
            m_components.push_back(currentComponent);
        }
    }

    EinSumHelper::RecognizedOperatorType EinSumHelper::DetermineRecognizedOperatorType()
    {
        if (m_components.empty())
        {
            return RecognizedOperatorType::None; // Parsing may have found unsupported components - treating as unknown.
        }

        // std::ranges::equal is not supported yet.
        auto equals = [](gsl::span<const uint32_t> a, gsl::span<const uint32_t> b)
        {
            return std::equal(a.begin(), a.end(), b.begin(), b.end());
        };

        auto as_span = [](std::initializer_list<uint32_t> il) {
            return gsl::make_span(il.begin(), il.size());
        };

        std::array<uint32_t, 3> componentRanks;
        if (m_components.size() > componentRanks.size())
        {
            // No recognized operator takes more than 2 inputs and 1 output.
            // EinSum itself is generic and can handle any variable number of inputs,
            // but DML's operators expect fixed counts.
            return RecognizedOperatorType::None;
        }
        else if (m_components.size() == 2)
        {
            auto inputLabels = m_components[0].GetLabels(m_labelIndices);
            auto outputLabels = m_components[1].GetLabels(m_labelIndices);
            if (inputLabels.size() == outputLabels.size())
            {
                // Check identity.
                if (equals(inputLabels, outputLabels))
                {
                    // Handles: "->", "i->i", "ij->ij", "ijk->ijk", "ijkl->ijkl" ...
                    return RecognizedOperatorType::Identity;
                }
                else // Transpose since a permutation exists.
                {
                    // Handles: "ij->ji", "ijk->kji", "ijkl->lkji", "ijkl->ijkl" ...
                    return RecognizedOperatorType::Transpose;
                }
            }
            else if (outputLabels.empty()) // Scalar output, with all inputs reduced.
            {
                // Handles: "i->", "ij->", "ijk->", "ijkl->" ...
                return RecognizedOperatorType::ReduceSum;
            }
        }
        else if (m_components.size() == 3)
        {
            // If all components have the same size and label order, then apply elementwise multiplication.
            auto inputALabels = m_components[0].GetLabels(m_labelIndices);
            auto inputBLabels = m_components[1].GetLabels(m_labelIndices);
            auto outputLabels = m_components[2].GetLabels(m_labelIndices);
            if (equals(inputALabels, outputLabels) && equals(inputBLabels, outputLabels))
            {
                // Handles: "i,i->i", "ij,ij->ij", "ijk,ijk->ijk", "ijkl,ijkl->ijkl" ...
                return RecognizedOperatorType::Multiply;
            }
        }

        // Otherwise check for special cases of dedicated operators...

        struct RecognizedOperatorInfo
        {
            RecognizedOperatorType recognizedOperatorType;
            std::initializer_list<uint32_t> componentRanks;
            std::initializer_list<uint32_t> labelIndices;
        };

        const RecognizedOperatorInfo recognizedOperators[] = {
            {RecognizedOperatorType::MatMul,               {2,2,2},{0,1, 1,2, 0,2}}, // ij,jk->ik
            {RecognizedOperatorType::MatMul,               {3,3,3},{0,1,2, 0,2,3, 0,1,3}}, // bij,bjk->bik
            {RecognizedOperatorType::MatMul,               {4,4,4},{0,1,2,3, 0,1,3,4, 0,1,2,4}}, // abij,abjk->abik
            {RecognizedOperatorType::OuterProduct,         {1,1,2},{0, 1, 0,1}}, // i,j->ij
            {RecognizedOperatorType::MatMulTransposeA,     {2,2,2},{0,1, 0,2, 1,2}}, // ji,jk->ik
            {RecognizedOperatorType::MatMulTransposeA,     {3,3,3},{0,1,2, 0,1,3, 0,2,3}}, // bji,bjk->bik
            {RecognizedOperatorType::MatMulTransposeA,     {4,4,4},{0,1,2,3, 0,1,2,4, 0,1,3,4}}, // abji,abjk->abik
            {RecognizedOperatorType::MatMulTransposeB,     {2,2,2},{0,1, 2,1, 0,2}}, // ij,kj->ik
            {RecognizedOperatorType::MatMulTransposeB,     {3,3,3},{0,1,2, 0,3,2, 0,1,3}}, // bij,bkj->bik
            {RecognizedOperatorType::MatMulTransposeB,     {4,4,4},{0,1,2,3, 0,1,4,3, 0,1,2,4}}, // abij,abkj->abik
            {RecognizedOperatorType::MatMulTransposeB,     {1,1,0},{0,0,}}, // i,i-> (1D inner_prod)
            {RecognizedOperatorType::MatMulNhcw,           {4,4,4},{0,1,2,3, 0,3,2,4, 0,1,2,4}}, // aibj,ajbk->aibk
            {RecognizedOperatorType::MatMulNhcwTransposeA, {4,4,4},{0,1,2,3, 0,1,2,4, 0,3,2,4}}, // ajbi,ajbk->aibk
            {RecognizedOperatorType::MatMulNhcwTransposeB, {4,4,4},{0,1,2,3, 0,4,2,3, 0,1,2,4}}, // aibj,akbj->aibk
            {RecognizedOperatorType::ReduceSum,            {2,1  },{0,1, 0}}, // ij->i
            {RecognizedOperatorType::ReduceSum,            {2,1  },{0,1, 1}}, // ij->j
        };

        // For each recognized operator, compare the labels-per-component and label indices.
        for (auto& recognizedOperator : recognizedOperators)
        {
            if (equals(m_labelIndices, as_span(recognizedOperator.labelIndices))
            &&  m_components.size() == recognizedOperator.componentRanks.size())
            {
                for (size_t i = 0; i < m_components.size(); ++i)
                {
                    componentRanks[i] = m_components[i].GetDimensionCount();
                }

                if (equals(gsl::make_span(componentRanks.data(), m_components.size()), as_span(recognizedOperator.componentRanks)))
                {
                    return recognizedOperator.recognizedOperatorType;
                }
            }
        }

        return RecognizedOperatorType::None;
    }

    std::vector<EdgeShapes> EinSumHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        assert(!m_components.empty()); // Should have already parsed components.

        uint32_t inputCount  = shapeInfo.GetInputCount();
        uint32_t outputCount = shapeInfo.GetOutputCount();
        ML_CHECK_VALID_ARGUMENT(inputCount + 1 == m_components.size(), "Mismatch between input tensor count and string equation component count.");
        ML_CHECK_VALID_ARGUMENT(outputCount == 1, "EinSum expects exactly 1 output tensor.");

        std::vector<uint32_t> labelSizes(m_labelIndices.size(), UINT_MAX);

        // Read every input tensor, comparing labels to ensure consistent sizes from the equation parsed earlier.
        for (uint32_t i = 0; i < inputCount; ++i)
        {
            auto inputShape = shapeInfo.GetInputTensorShape(i);
            auto& component = m_components[i];
            auto labelIndices = component.GetLabels(m_labelIndices);
            uint32_t dimensionCount = component.GetDimensionCount();

            ML_CHECK_VALID_ARGUMENT(inputShape.size() == dimensionCount, "Mismatch between input tensor shape and string equation label count.");

            for (uint32_t j = 0; j < dimensionCount; ++j)
            {
                // If this is the first time seeing this label, then record the size.
                // Otherwise any following occurrences of the label must match sizes.
                // e.g. Given "ij,ji", both i's and both j's must match dimension sizes.
                uint32_t dimensionSize = inputShape[j];
                uint32_t labelIndex = labelIndices[j];
                assert(labelIndex < labelSizes.size());

                if (labelSizes[labelIndex] == UINT_MAX)
                {
                    labelSizes[labelIndex] = dimensionSize;
                }
                else
                {
                    ML_CHECK_VALID_ARGUMENT(labelSizes[labelIndex] == dimensionSize, "All labels must have the same dimension sizes.");
                }
            }
        }

        // Generate output dimensions from corresponding input tensor labels.
        // e.g. Given ij,jk->ij with [2,3] and [3,5], the output is [2,5].
        std::vector<uint32_t> outputDimensions;
        auto outputLabelIndices = m_components.back().GetLabels(m_labelIndices);
        for (auto labelIndex : outputLabelIndices)
        {
            outputDimensions.push_back(labelSizes[labelIndex]);
        }

        return { EdgeShapes(outputDimensions) };
    }

    bool EinSumHelper::IsMatMulOperatorType() const noexcept
    {
        return m_recognizedOperatorType == RecognizedOperatorType::OuterProduct ||
            m_recognizedOperatorType == RecognizedOperatorType::MatMul ||
            m_recognizedOperatorType == RecognizedOperatorType::MatMulTransposeA ||
            m_recognizedOperatorType == RecognizedOperatorType::MatMulTransposeB ||
            m_recognizedOperatorType == RecognizedOperatorType::MatMulNhcw ||
            m_recognizedOperatorType == RecognizedOperatorType::MatMulNhcwTransposeA ||
            m_recognizedOperatorType == RecognizedOperatorType::MatMulNhcwTransposeB;
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

    std::vector<EdgeShapes> FusedMatMulHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputCount() == 2);

        auto inputShape0 = shapeInfo.GetInputTensorShape(0);
        auto inputShape1 = shapeInfo.GetInputTensorShape(1);
        ML_CHECK_VALID_ARGUMENT(inputShape0.size() >= 1);
        ML_CHECK_VALID_ARGUMENT(inputShape1.size() >= 1);

        std::vector<int64_t> aSizes(inputShape0.begin(), inputShape0.end());
        std::vector<int64_t> bSizes(inputShape1.begin(), inputShape1.end());
        auto transAAttr = shapeInfo.GetOptionalAttribute<int64_t>(AttrName::TransA, 0);
        auto transBAttr = shapeInfo.GetOptionalAttribute<int64_t>(AttrName::TransB, 0);

        const bool transA = transAAttr && aSizes.size() != 1;
        const bool transB = transBAttr && bSizes.size() != 1;
        auto transBatchA = shapeInfo.GetOptionalAttribute<int64_t>(AttrName::TransBatchA, 0);
        auto transBatchB = shapeInfo.GetOptionalAttribute<int64_t>(AttrName::TransBatchB, 0);

        onnxruntime::MatMulComputeHelper helper;
        ML_CHECK_VALID_ARGUMENT(helper.Compute(onnxruntime::TensorShape(aSizes), onnxruntime::TensorShape(bSizes), transA, transB, transBatchA, transBatchB, false).IsOK());

        auto outputDims = helper.OutputShape().AsShapeVector();

        std::vector<uint32_t> outputShape;
        outputShape.reserve(outputDims.size());
        std::transform(outputDims.begin(), outputDims.end(), std::back_inserter(outputShape), [](int64_t dimSize){ return static_cast<uint32_t>(dimSize); });

        return {std::move(outputShape)};
    }

    void TopKHelper::Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        uint32_t opsetVersion
        )
    {
        auto& attributes = kernelInformation.GetAttributes();
        int32_t k;
        if (opsetVersion >= 10)
        {
            MLOperatorTensor kTensor = kernelInformation.GetConstantInputTensor(1);
            k = gsl::narrow_cast<int32_t>(ReadScalarTensorCastToInt64(kTensor));
        }
        else
        {
            k = attributes.template GetOptionalAttribute<int32_t>(AttrName::K, -1);
        }
        ML_CHECK_VALID_ARGUMENT(k >= 0, "Attribute k is missing or negative.");
        m_k = k;

        auto inputShape = shapeInformation.GetInputTensorShape(0);
        int32_t axis = attributes.template GetOptionalAttribute<int32_t>(AttrName::Axis, -1);
        m_axis = HandleNegativeAxis(axis, gsl::narrow_cast<uint32_t>(inputShape.size()));
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
            return { EdgeShapes(outputDimensionsSequence) };
        }
        case 2:
        {
            std::vector<EdgeShapes> outputShapes = {
                EdgeShapes(outputDimensionsSequence),
                EdgeShapes(outputDimensionsSingle)
            };

            return outputShapes;
        }
        case 3:
        {
            std::vector<EdgeShapes> outputShapes = {
                EdgeShapes(outputDimensionsSequence),
                EdgeShapes(outputDimensionsSingle),
                EdgeShapes(cellOutputDimensionsSingle)
            };

            return outputShapes;
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

    void Col2ImHelper::Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation)
    {
        std::vector<int> tensor;
        ReadCpuLocalTensorIntoInt32(kernelInformation.GetConstantInputTensor(1), /*out*/ tensor);
        m_imageShape.resize(tensor.size());
        DowncastDimensions(gsl::make_span<const int>(tensor.data(), tensor.size()), /*out*/ m_imageShape);
        ReadCpuLocalTensorIntoInt32(kernelInformation.GetConstantInputTensor(2), /*out*/ tensor);
        m_blockShape.resize(tensor.size());
        DowncastDimensions(gsl::make_span<const int>(tensor.data(), tensor.size()), /*out*/ m_blockShape);

        const uint32_t dimCount = gsl::narrow_cast<uint32_t>(m_blockShape.size());
        m_dilations = {dimCount, 1};
        m_pads = {dimCount * 2, 0};
        m_strides = {dimCount, 1};

        if (kernelInformation.HasAttribute(AttrName::Dilations, MLOperatorAttributeType::IntArray))
        {
            tensor = kernelInformation.GetAttributes().GetOptionalAttributeVectorInt32(AttrName::Dilations);
            m_dilations.resize(tensor.size());
            DowncastDimensions(gsl::make_span<const int>(tensor.data(), tensor.size()), /*out*/ m_dilations);
            ML_CHECK_VALID_ARGUMENT(m_dilations.size() == dimCount);
        }

        if (kernelInformation.HasAttribute(AttrName::Pads, MLOperatorAttributeType::IntArray))
        {
            tensor = kernelInformation.GetAttributes().GetOptionalAttributeVectorInt32(AttrName::Pads);
            m_pads.resize(tensor.size());
            DowncastDimensions(gsl::make_span<const int>(tensor.data(), tensor.size()), /*out*/ m_pads);
            ML_CHECK_VALID_ARGUMENT(m_pads.size() == dimCount * 2);
        }

        if (kernelInformation.HasAttribute(AttrName::Strides, MLOperatorAttributeType::IntArray))
        {
            tensor = kernelInformation.GetAttributes().GetOptionalAttributeVectorInt32(AttrName::Strides);
            m_strides.resize(tensor.size());
            DowncastDimensions(gsl::make_span<const int>(tensor.data(), tensor.size()), /*out*/ m_strides);
            ML_CHECK_VALID_ARGUMENT(m_strides.size() == dimCount);
        }

        m_inputShape = shapeInformation.GetInputTensorShape(0);

        auto blockShapeProduct = ComputeElementCountFromDimensions(m_blockShape);
        m_outputShape.resize(2 + m_imageShape.size());
        m_outputShape[0] = m_inputShape[0];                     // N
        m_outputShape[1] = m_inputShape[1] / blockShapeProduct; // C
        for (int i = 2; i < m_outputShape.size(); i++)
        {
            m_outputShape[i] = m_imageShape[i - 2];
        };
    }

    std::vector<EdgeShapes> Col2ImHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return { EdgeShapes(m_outputShape) };
    }

    void ConcatHelperBase::Initialize(
        const MLOperatorAttributes& operatorAttributes,
        gsl::span<const DimensionType> inputDimensions
        )
    {
        uint32_t inputDimCount = gsl::narrow_cast<uint32_t>(inputDimensions.size());
        m_axis = static_cast<int>(HandleNegativeAxis(operatorAttributes.GetOptionalAttribute<int>(AttrName::Axis, -1), inputDimCount));
        ML_CHECK_VALID_ARGUMENT(m_axis < static_cast<int>(inputDimensions.size()));
    }

    std::vector<EdgeShapes> ConcatHelperBase::GetOutputShapes(const MLShapeInferenceContext& shapeInfo, uint32_t firstInputIndex, uint32_t step) const
    {
        auto outputShape = shapeInfo.GetInputTensorShape(firstInputIndex);

        uint32_t inputCount = shapeInfo.GetInputCount();

        for (uint32_t i = firstInputIndex + step; i < inputCount; i += step)
        {
            auto inputShape = shapeInfo.GetInputTensorShape(i);
            for (size_t j = 0; j < outputShape.size(); ++j)
            {
                if (static_cast<size_t>(m_axis) == j)
                {
                    outputShape[j] += inputShape[j];
                }
            }
        }

        return { EdgeShapes(outputShape) };
    }

    std::vector<EdgeShapes> ConcatHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return ConcatHelperBase::GetOutputShapes(shapeInfo, 0, 1);
    }

    std::vector<EdgeShapes> QLinearConcatHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return ConcatHelperBase::GetOutputShapes(shapeInfo, 2, 3);
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

        return { EdgeShapes(outputDimensions) };
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

        return { EdgeShapes(outputDimensions) };
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

        return { outputDimensions };
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

    std::vector<EdgeShapes> QLinearAveragePoolingHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto inputShape = shapeInfo.GetInputTensorShape(0);
        std::vector<DimensionType> outputDimensions = InitializeKernelOutputDimensions(inputShape, m_kernel, m_kernel.channelsLast);

        const uint32_t outputCount = shapeInfo.GetOutputCount();

        std::vector<EdgeShapes> outputShapes;
        for (uint32_t i = 0; i < outputCount; ++i)
        {
            outputShapes.push_back(outputDimensions);
        }
        return outputShapes;
    }

    std::vector<EdgeShapes> QLinearGlobalAveragePoolingHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto inputShape = shapeInfo.GetInputTensorShape(0);
        std::vector<DimensionType> outputDimensions = InitializeKernelOutputDimensions(inputShape, m_kernel, m_kernel.channelsLast);

        const uint32_t outputCount = shapeInfo.GetOutputCount();

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
            static_cast<DimensionType>(m_outputSizeH),
            static_cast<DimensionType>(m_outputSizeW),
        };

        return { EdgeShapes(outputDimensions) };
    }

    std::vector<EdgeShapes> RoiAlignHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        auto roiShape = shapeInfo.GetInputTensorShape(InputTensors::ROIS);
        auto inputShape = shapeInfo.GetInputTensorShape(InputTensors::INPUT);
        ML_CHECK_VALID_ARGUMENT(inputShape.size() >= 4, "inputShape must be >= 4.");

        DimensionType outputDimensions[4] =
        {
            roiShape[0], // number of ROIs
            inputShape[C], // number of channels
            static_cast<DimensionType>(m_outputSizeH),
            static_cast<DimensionType>(m_outputSizeW),
        };

        return { EdgeShapes(outputDimensions) };
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
            DowncastDimensions(gsl::make_span<const int64_t>(outputDimensions64bit.data(), outputDimensions64bit.size()), /*out*/ outputDimensions);
        }
        else
        {
            outputDimensions = m_inferredOutputDimensions;
        }

        return { outputDimensions };
    }

    void SqueezeHelper::Initialize(
        IKernelInformationAdapter const& kernelInformation,
        IShapeInformationAdapter const& shapeInformation,
        uint32_t opsetVersion
        )
    {
        if (opsetVersion >= 13) // Axes are a dynamic input parameter.
        {
            if (kernelInformation.IsInputValid(1))
            {
                ReadCpuLocalTensorIntoInt32(kernelInformation.GetConstantInputTensor(1), /*out*/ m_axes);
            }
        }
        else // Axes were a constant attribute parameter.
        {
            m_axes = kernelInformation.GetAttributes().GetOptionalAttributeVectorInt32(AttrName::Axes);
        }
        std::vector<DimensionType> inputDimensions = shapeInformation.GetInputTensorShape(0);

        HandleNegativeAxes(/*inout*/ m_axes, gsl::narrow_cast<uint32_t>(inputDimensions.size()));
        std::sort(m_axes.begin(), m_axes.end());
        if (m_axes.empty())
        {
            m_axes.resize(inputDimensions.size());
            std::iota(m_axes.begin(), m_axes.end(), 0u);
        }
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

        return { outputDimensions };
    }

    void UnsqueezeHelper::Initialize(
        IKernelInformationAdapter const& kernelInformation,
        IShapeInformationAdapter const& shapeInformation,
        uint32_t opsetVersion
        )
    {
        if (opsetVersion >= 13) // Axes are a dynamic input parameter.
        {
            ReadCpuLocalTensorIntoInt32(kernelInformation.GetConstantInputTensor(1), /*out*/ m_axes);
        }
        else // Axes were a constant attribute parameter.
        {
            m_axes = kernelInformation.GetAttributes().GetOptionalAttributeVectorInt32(AttrName::Axes);
        }
        std::vector<DimensionType> inputDimensions = shapeInformation.GetInputTensorShape(0);

        const uint32_t outputDimensionCount = gsl::narrow_cast<uint32_t>(inputDimensions.size() + m_axes.size());
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

        return { outputDimensions };
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

        return { EdgeShapes(outputDimensions) };
    }

    void ReshapeHelper::Initialize(IKernelInformationAdapter const& kernelInformation)
    {
        ML_CHECK_VALID_ARGUMENT(kernelInformation.GetInputCount() >= 2);
        ML_CHECK_VALID_ARGUMENT(kernelInformation.GetOutputCount() >= 1);

        // The 'shape' tensor is a 1D tensor holding the new shape to reshape to,
        // and the first element of its own shape holds how many dimensions there
        // will be for the output.
        MLOperatorTensor shapeTensor = kernelInformation.GetConstantInputTensor(1);
        ReadCpuLocalTensorIntoInt32(shapeTensor, /*out*/ m_shapeDims);
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
        bool allowZero = shapeInfo.template GetOptionalAttribute<int32_t>(AttrName::AllowZero, 0);

        if (allowZero)
        {
            // Just take the shape directly (no special handling for 0).
            for (int i = 0, ci = gsl::narrow_cast<int>(m_shapeDims.size()); i < ci; ++i)
            {
                outputDimensions[i] = m_shapeDims[i];
            }
        }
        else
        {
            // Special handling where 0 size means to copy the corresponding input tensor dimension.
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
            }
        }

        return { EdgeShapes(outputDimensions) };
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

        return { EdgeShapes(outputDimensions) };
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

        return { EdgeShapes(desiredTensorShape) };
    }

    void TileHelper::Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation
        )
    {
        m_inputDimensions = shapeInformation.GetInputTensorShape(0);

        // Read the repeats tensor.
        const std::vector<uint32_t> repeatsTensorDimensions = shapeInformation.GetInputTensorShape(1);
        ML_CHECK_VALID_ARGUMENT(repeatsTensorDimensions.size() == 1, "Tile's repeats tensor must be 1D.");
        const size_t dimCount = repeatsTensorDimensions[0];

        MLOperatorTensor repeatsTensor = kernelInformation.GetConstantInputTensor(1);
        const int64_t* repeatsData = repeatsTensor.GetData<int64_t>();
        ML_CHECK_VALID_ARGUMENT(m_inputDimensions.size() == dimCount, "Tile's repeats tensor must be the same dimension count as the input tensor.");
        ML_CHECK_VALID_ARGUMENT(repeatsTensor.IsCpuData(), "Tile's repeats tensor must be CPU Tensor.");

        for (size_t i = 0; i < dimCount; ++i)
        {
            ML_CHECK_VALID_ARGUMENT(repeatsData[i] >= 0, "Repeat values should be >= 0.");
            m_repeatsData.push_back(gsl::narrow_cast<uint32_t>(repeatsData[i]));
        }

        // Update the computed output shape accordingly, repeat every axis's length by the repeat count.
        m_outputDimensions.assign(m_inputDimensions.begin(), m_inputDimensions.end());

        for (size_t dimIndex = 0; dimIndex < dimCount; ++dimIndex)
        {
            m_outputDimensions[dimIndex] *= m_repeatsData[dimIndex];
        }
    }

    std::vector<EdgeShapes> TileHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return { EdgeShapes(m_outputDimensions) };
    }

    void ResizeHelper::Initialize(
        IKernelInformationAdapter const& kernelInformation,
        IShapeInformationAdapter const& shapeInformation,
        uint32_t opsetVersion
        )
    {
        auto& attributes = kernelInformation.GetAttributes();
        m_inputDimensions = shapeInformation.GetInputTensorShape(0);
        std::vector<int32_t> outputSizes;

        if (opsetVersion >= 11)
        {
            if (kernelInformation.IsInputValid(1))
            {
                MLOperatorTensor regionOfInterestTensor = kernelInformation.GetConstantInputTensor(1);
                ReadCpuLocalTensorIntoFloat32(regionOfInterestTensor, /*out*/ m_regionOfInterest);
            }
            if (kernelInformation.IsInputValid(2))
            {
                MLOperatorTensor scalesTensor = kernelInformation.GetConstantInputTensor(2);
                ReadCpuLocalTensorIntoFloat32(scalesTensor, /*out*/ m_scales);
            }
            if (kernelInformation.IsInputValid(3))
            {
                MLOperatorTensor outputSizesTensor = kernelInformation.GetConstantInputTensor(3);
                ReadCpuLocalTensorIntoInt32(outputSizesTensor, /*out*/ outputSizes);
            }
        }
        else if (opsetVersion >= 9)
        {
            // Read the scales from the 2nd tensor.
            // Compatible with Upsample-9/Upsample-10 and Resize-10.
            MLOperatorTensor scalesTensor = kernelInformation.GetConstantInputTensor(1);
            ReadCpuLocalTensorIntoFloat32(scalesTensor, /*out*/ m_scales);
        }
        else
        {
            // From attribute, compatible with Upsample-7.
            m_scales = attributes.template GetOptionalAttribute<std::vector<float>>(AttrName::Scales, std::vector<float>());
        }

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
                m_outputDimensions.push_back(gsl::narrow_cast<uint32_t>(floor(m_inputDimensions[i] * scale)));
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

    void OneHotHelper::Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation
        )
    {
        ML_CHECK_VALID_ARGUMENT(kernelInformation.GetInputCount() == 3);
        ML_CHECK_VALID_ARGUMENT(kernelInformation.GetOutputCount() == 1);

        auto& attributes = kernelInformation.GetAttributes();
        const std::vector<DimensionType> inputDimensions = shapeInformation.GetInputTensorShape(0);
        std::vector<uint32_t> outputDimensions;

        m_onnxAxis = attributes.template GetOptionalAttribute<int32_t>(AttrName::Axis, -1);

        // Get 'depth' tensor, which is really a scalar for the output size along the given axis.
        MLOperatorTensor shapeTensor = kernelInformation.GetConstantInputTensor(1);

        auto indicesShape = shapeInformation.GetInputTensorShape(0);
        m_absoluteAxis = HandleNegativeAxis(m_onnxAxis, gsl::narrow_cast<uint32_t>(indicesShape.size() + 1));

        // The shape tensor ('depth') is a 0D tensor holding the size for the output tensor along the specified axis.
        // It must be registered as OrtMemType::OrtMemTypeCPUInput for CPU read access.
        const int64_t depth64 = ReadScalarTensorCastToInt64(shapeTensor);
        ML_CHECK_VALID_ARGUMENT(depth64 > 0, "Negative or zero 'depth' values for OneHot are illegal.");
        const uint32_t depth = gsl::narrow_cast<uint32_t>(depth64);
        m_outputDimensions.assign(indicesShape.begin(), indicesShape.end());
        m_outputDimensions.insert(m_outputDimensions.begin() + m_absoluteAxis, depth);
    }

    std::vector<EdgeShapes> OneHotHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return { EdgeShapes(m_outputDimensions) };
    }

    void BatchNormalizationHelper::Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation
        )
    {
        ML_CHECK_VALID_ARGUMENT(kernelInformation.GetInputCount() == 5);
        ML_CHECK_VALID_ARGUMENT(kernelInformation.GetOutputCount() >= 1 && kernelInformation.GetOutputCount() <= 3);
    }

    std::vector<EdgeShapes> BatchNormalizationHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInformation) const
    {
        std::vector<EdgeShapes> outputDimensionsList;

        outputDimensionsList.push_back(EdgeShapes(shapeInformation.GetInputTensorShape(0))); // output.shape = input.shape
        int32_t trainingMode = shapeInformation.GetOptionalAttribute<int32_t>(AttrName::TrainingMode, 0);
        if (trainingMode && shapeInformation.GetOutputCount() >= 3)
        {
            outputDimensionsList.push_back(EdgeShapes(shapeInformation.GetInputTensorShape(3))); // running_mean.shape = input_mean.shape
            outputDimensionsList.push_back(EdgeShapes(shapeInformation.GetInputTensorShape(4))); // running_variance.shape = input_variance.shape
        }
        return outputDimensionsList;
    }

    void ShapeHelper::Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation
        )
    {
        ML_CHECK_VALID_ARGUMENT(kernelInformation.GetInputCount() == 1);
        ML_CHECK_VALID_ARGUMENT(kernelInformation.GetOutputCount() == 1);

        std::vector<uint32_t> inputShape = shapeInformation.GetInputTensorShape(0);
        const uint32_t inputDimCount = static_cast<uint32_t>(inputShape.size());

        const auto& attributes = kernelInformation.GetAttributes();
        const int64_t startIndex = attributes.template GetOptionalAttribute<int64_t>(AttrName::Start, 0);
        const int64_t endIndex = attributes.template GetOptionalAttribute<int64_t>(AttrName::End, inputDimCount);

        // Compute the starting and ending indices for the slice
        int64_t trueStart = startIndex < 0 ? startIndex + inputDimCount : startIndex;
        trueStart = std::clamp<int64_t>(trueStart, 0, inputDimCount);

        int64_t trueEnd = endIndex < 0 ? endIndex + inputDimCount : endIndex;
        trueEnd = std::clamp<int64_t>(trueEnd, 0, inputDimCount);

        m_sliceStart = static_cast<uint32_t>(trueStart);
        m_sliceEnd = std::max<uint32_t>(static_cast<uint32_t>(trueEnd), m_sliceStart);
    }

    std::vector<EdgeShapes> ShapeHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return { EdgeShapes({m_sliceEnd - m_sliceStart}) };
    }

    std::vector<EdgeShapes> SizeHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        return { EdgeShapes({}) };
    }

    std::vector<EdgeShapes> EmbedLayerNormalizationHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputCount() >= 3);

        auto inputIdsShape = shapeInfo.GetInputTensorShape(0);
        auto wordEmbeddingShape = shapeInfo.GetInputTensorShape(2);

        // input_ids and word_embedding are 2D tensors
        ML_CHECK_VALID_ARGUMENT(inputIdsShape.size() == 2);
        ML_CHECK_VALID_ARGUMENT(wordEmbeddingShape.size() == 2);

        uint32_t batchSize = inputIdsShape[0];
        uint32_t sequenceLength = inputIdsShape[1];
        uint32_t hiddenSize = wordEmbeddingShape[1];

        std::vector<EdgeShapes> outputShapes;
        outputShapes.reserve(3);

        outputShapes.push_back(EdgeShapes({batchSize, sequenceLength, hiddenSize}));
        outputShapes.push_back(EdgeShapes({batchSize}));

        if (shapeInfo.GetOutputCount() == 3)
        {
            outputShapes.push_back(EdgeShapes({batchSize, sequenceLength, hiddenSize}));
        }

        return outputShapes;
    }

    std::vector<EdgeShapes> MultiHeadAttentionHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputCount() >= 1);

        auto queryShape = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(queryShape.size() == 3 || queryShape.size() == 5);

        const uint32_t batchSize = queryShape[0];
        const uint32_t sequenceLength = queryShape[1];
        uint32_t kvSequenceLength = 0;
        uint32_t vHiddenSize = 0;
        uint32_t headSize = 0;

        if (shapeInfo.IsInputValid(2))
        {
            auto valueShape = shapeInfo.GetInputTensorShape(2);
            ML_CHECK_VALID_ARGUMENT(queryShape.size() == 3);
            headSize = queryShape[2] / m_numHeads;

            if (valueShape.size() == 3)
            {
                kvSequenceLength = valueShape[1];
                vHiddenSize = valueShape[2];
            }
            else
            {
                ML_CHECK_VALID_ARGUMENT(valueShape.size() == 4);
                const uint32_t vHeadSize = valueShape[3];
                kvSequenceLength = valueShape[2];
                vHiddenSize = vHeadSize * m_numHeads;
            }
        }
        else if (shapeInfo.IsInputValid(1))
        {
            auto keyShape = shapeInfo.GetInputTensorShape(1);
            ML_CHECK_VALID_ARGUMENT(keyShape.size() == 5);
            kvSequenceLength = keyShape[1];
            vHiddenSize = queryShape[2];
            headSize = keyShape[4];
        }
        else
        {
            ML_CHECK_VALID_ARGUMENT(queryShape.size() == 5);
            kvSequenceLength = queryShape[1];
            headSize = queryShape[4];
            vHiddenSize = headSize * m_numHeads;
        }

        std::vector<EdgeShapes> outputShapes(3);
        outputShapes[0] = EdgeShapes({batchSize, sequenceLength, vHiddenSize});

        uint32_t totalSequenceLength = kvSequenceLength;
        if (shapeInfo.IsInputValid(6))
        {
            ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputTensorDimensionCount(6) == 4);
            const uint32_t pastSequenceLength = shapeInfo.GetInputTensorShape(6)[2];
            totalSequenceLength += pastSequenceLength;
        }

        if (shapeInfo.IsOutputValid(1))
        {
            outputShapes[1] = EdgeShapes({batchSize, m_numHeads, totalSequenceLength, headSize});
        }

        if (shapeInfo.IsOutputValid(2))
        {
            outputShapes[2] = EdgeShapes({batchSize, m_numHeads, totalSequenceLength, headSize});
        }

        return outputShapes;
    }

    void MultiHeadAttentionHelper::Initialize(const IKernelInformationAdapter& kernelInformation)
    {
        m_numHeads = gsl::narrow_cast<uint32_t>(kernelInformation.GetAttributes().GetAttribute<int64_t>(AttrName::NumHeads));
    }

    std::vector<EdgeShapes> AttentionHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputCount() >= 2);

        auto queryShape = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(queryShape.size() == 3);

        auto weightShape = shapeInfo.GetInputTensorShape(1);
        ML_CHECK_VALID_ARGUMENT(weightShape.size() == 2);

        if (m_qkvHiddenSizes.empty())
        {
            ML_CHECK_VALID_ARGUMENT(weightShape[1] % 3 == 0);
        }
        else
        {
            ML_CHECK_VALID_ARGUMENT(m_qkvHiddenSizes.size() == 3);
        }

        const uint32_t batchSize = queryShape[0];
        const uint32_t sequenceLength = queryShape[1];
        const uint32_t vHiddenSize = m_qkvHiddenSizes.empty() ? weightShape[1] / 3 : m_qkvHiddenSizes[2];

        return { EdgeShapes({batchSize, sequenceLength, vHiddenSize}) };
    }

    void AttentionHelper::Initialize(const IKernelInformationAdapter& kernelInformation)
    {
        m_qkvHiddenSizes = kernelInformation.GetAttributes().GetOptionalAttributeVectorInt32(AttrName::QkvHiddenSizes);
    }

    std::vector<EdgeShapes> QAttentionHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputCount() >= 5);

        auto queryShape = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(queryShape.size() == 3);

        auto weightShape = shapeInfo.GetInputTensorShape(1);
        ML_CHECK_VALID_ARGUMENT(weightShape.size() == 2);
        ML_CHECK_VALID_ARGUMENT(weightShape[1] % 3 == 0);

        const uint32_t batchSize = queryShape[0];
        const uint32_t sequenceLength = queryShape[1];
        const uint32_t hiddenSize = weightShape[1] / 3;
        const uint32_t headSize = hiddenSize / m_numHeads;

        std::vector<EdgeShapes> outputShapes(2);

        outputShapes[0] = EdgeShapes({batchSize, sequenceLength, hiddenSize});

        uint32_t totalSequenceLength = sequenceLength;
        if (shapeInfo.IsInputValid(8))
        {
            ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputTensorDimensionCount(8) == 5);
            const uint32_t pastSequenceLength = shapeInfo.GetInputTensorShape(8)[3];
            totalSequenceLength += pastSequenceLength;
        }

        if (shapeInfo.IsOutputValid(1))
        {
            ML_CHECK_VALID_ARGUMENT(shapeInfo.IsInputValid(8));
            outputShapes[1] = EdgeShapes({2, batchSize, m_numHeads, totalSequenceLength, headSize});
        }

        return outputShapes;
    }

    void QAttentionHelper::Initialize(const IKernelInformationAdapter& kernelInformation)
    {
        m_numHeads = gsl::narrow_cast<uint32_t>(kernelInformation.GetAttributes().GetAttribute<int64_t>(AttrName::NumHeads));
    }

    std::vector<EdgeShapes> SkipLayerNormHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputCount() >= 3);

        auto inputShape = shapeInfo.GetInputTensorShape(0);

        std::vector<EdgeShapes> outputShapes(4);
        outputShapes[0] = EdgeShapes(inputShape);

        if (shapeInfo.IsOutputValid(3))
        {
            outputShapes[3] = EdgeShapes(inputShape);
        }

        return outputShapes;
    }

    std::vector<EdgeShapes> BiasSplitGeluHelper::GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const
    {
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetInputCount() == 2);
        ML_CHECK_VALID_ARGUMENT(shapeInfo.GetOutputCount() == 1);
        auto outputShape = shapeInfo.GetInputTensorShape(0);
        ML_CHECK_VALID_ARGUMENT(outputShape.size() >= 1);
        ML_CHECK_VALID_ARGUMENT(outputShape.back() % 2 == 0);
        outputShape.back() /= 2;

        return { EdgeShapes(std::move(outputShape)) };
    }

} // namespace OperatorHelper

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Common.h"
#include "Attributes.h"
#include "core/common/common.h"
#include "../DmlExecutionProvider/src/ErrorHandling.h"
#include "MLOperatorAuthorHelper.h"

namespace OperatorHelper
{

bool ContainsEmptyDimensions(gsl::span<const DimensionType> dimensions);

std::vector<DimensionType> BroadcastTensorShape(
    gsl::span<const DimensionType> inputShape0,
    gsl::span<const DimensionType> inputShape1);

// This won't allocate extra memory, if required. This expects
// caller to make the size of all containers to be same
void BroadcastTensorShapeAndSetStrides(
    gsl::span<DimensionType> inputShape0,
    gsl::span<DimensionType> inputStride0,
    gsl::span<DimensionType> inputShape1,
    gsl::span<DimensionType> inputStride1);

// Find all the occurrences of a value, and return the array indices (in ascending order).
//
// e.g. input values = {2,1,3,1,1,5}
//      value = 1
//      output indices = {1,3,4}
#ifndef __clang__
#pragma optimize("", off)
#endif
template <typename T>
void FindValueIndices(gsl::span<const T> values, T value, /*out*/ std::vector<uint32_t>& indices)
{
    indices.clear();
    for (size_t i = 0, valuesCount = values.size(); i < valuesCount; ++i)
    {
        // Work around compiler bug on x86 release by using data() rather than operator [] directly.
        // cl.exe 19.20.27412.4 for x86
        if (values.data()[i] == value)
        {
            indices.push_back(gsl::narrow_cast<uint32_t>(i));
        }
    }
}
#ifndef __clang__
#pragma optimize("", on)
#endif

// Convert any negative axis into an absolute axis relative to the back end.
// So given 3 dimensions, a -1 refers to axis 2, and -3 to axis 0.
uint32_t HandleNegativeAxis(int32_t signedOnnxAxis, uint32_t dimCount, bool validateAxis = true);

void HandleNegativeAxes(gsl::span<int32_t> onnxAxes, uint32_t dimCount);

// Remove array entries of the given indices (in ascending order), shifting them toward the front.
// There is a special check to avoid removing all the values, since returning a completely
// empty array would frequently causes errors later in many uses (such as with dimensions).
//
// e.g. input values = {2,1,3,1,1,5}
//      ellidable input indices = {1,3,4}
//      output values = {2,3,5}
template <typename T>
void RemoveValuesByIndex(gsl::span<const uint32_t> indices, bool keepOneValue, /*inout*/ std::vector<T>& values)
{
    assert(std::is_sorted(indices.begin(), indices.end()));

    // Keep the last value at least, if all values would otherwise be removed.
    if (keepOneValue && !indices.empty() && indices.size() == values.size())
    {
        indices = indices.first(indices.size() - 1);
    }

    auto indicesIterator = indices.begin();
    auto indicesEnd = indices.end();
    size_t oldValuesCount = values.size();
    size_t newValuesCount = 0;
    size_t nextIndex = (indicesIterator == indicesEnd) ? SIZE_MAX : *(indicesIterator++);

    // For every value, either skip the entry, or copy it to the output.
    for (size_t i = 0; i < oldValuesCount; ++i)
    {
        if (i == nextIndex)  // Skip and remove entry.
        {
            nextIndex = (indicesIterator == indicesEnd) ? SIZE_MAX : *(indicesIterator++);
        }
        else  // Keep and copy entry.
        {
            values[newValuesCount++] = values[i];
        }
    }
    values.resize(newValuesCount);
}

template <typename T>
void FillWithLeadingValues(/*inout*/ std::vector<T>& values, uint32_t minimumElementCount, T fillValue)
{
    // e.g.
    // input = [6,7]
    // elementCount = 4
    // fillValue = 1
    // output = [1,1,6,7]

    const size_t oldElementCount = values.size();
    const size_t newElementCount = std::max(size_t(minimumElementCount), oldElementCount);
    const size_t fillCount = newElementCount - oldElementCount;

    values.resize(newElementCount);
    std::copy_backward(values.begin(), values.begin() + oldElementCount, values.end());
    std::fill_n(values.data(), fillCount, fillValue);
}

int64_t CastToInt64(MLOperatorTensorDataType tensorDataType, const void* p);
double CastToFloat64(MLOperatorTensorDataType tensorDataType, const void* p);
void ReadScalarTensorData(const MLOperatorTensor& tensor, /*out*/ void* data, size_t dataByteSize);
int64_t ReadScalarTensorCastToInt64(const MLOperatorTensor& tensor);
double ReadScalarTensorCastToFloat64(const MLOperatorTensor& tensor);

void ReadCpuLocalTensorIntoInt32(const MLOperatorTensor& tensor, std::vector<int32_t>& result);
void ReadCpuLocalTensorIntoFloat32(const MLOperatorTensor& tensor, std::vector<float>& result);

class EdgeShapes
{
public:
    EdgeShapes() = default;
    EdgeShapes(const std::vector<uint32_t>& dim) { m_shapes = dim; }
    EdgeShapes(const std::initializer_list<uint32_t>& dim) { m_shapes.assign(dim.begin(), dim.end()); }
    EdgeShapes(const gsl::span<const DimensionType> dim) { m_shapes.assign(dim.begin(), dim.end()); }

    bool IsTensor() { return true; }
    bool IsUnused() { return m_shapes.empty(); }

    std::vector<uint32_t>& GetShape() { return m_shapes; }

private:
    std::vector<uint32_t> m_shapes;
};

struct KernelArgs
{
    // Initialize arrays up to NcdhwSpatialDimensionCount to avoid vector allocations,
    // but it's important to use .spatialDimensionCount when accessing them because
    // values beyond that may be bogus.
    uint32_t strides[NcdhwSpatialDimensionCount];
    uint32_t dilations[NcdhwSpatialDimensionCount];
    uint32_t windowSize[NcdhwSpatialDimensionCount]; // The filter kernel dimensions.
    uint32_t startPadding[NcdhwSpatialDimensionCount];
    uint32_t endPadding[NcdhwSpatialDimensionCount];
    uint32_t outputPadding[NcdhwSpatialDimensionCount];

    // This is true if padding must be automatically computed based on input sizes.
    // ResolveAutoPadding must happen during Compute rather than initialization.
    // This is temporary until kernel initialization routine once Lotus can provide
    // sizes at operator initialization.
    bool autoPad = false;
    bool autoPadSameUpper = false;
    bool useCeilingOutputShape = false;
    bool channelsLast = false;
    uint32_t spatialDimensionCount = 0;

    KernelArgs(uint32_t spatialDimensionCount) : spatialDimensionCount(spatialDimensionCount)
    {
        ML_CHECK_VALID_ARGUMENT(spatialDimensionCount <= NcdhwSpatialDimensionCount);
    }

    void FillWithLeadingValues(gsl::span<const uint32_t> input, gsl::span<uint32_t> output, uint32_t fillCount, uint32_t value)
    {
        // e.g.
        // input = [5,6,7,8]
        // fillcount = 2
        // value = 1
        // output = [1,1,5,6,7,8]

        const size_t inputCount = input.size();
        const size_t outputCount = output.size();
        const size_t copyCount = std::min(outputCount - fillCount, inputCount);

        std::fill_n(output.data(), fillCount, value);
        std::copy_n(input.data(), copyCount, output.data() + fillCount);
    }

    // Create a copy of an existing kernel args with a minimum dimension count,
    // filling the leading attribute values with 1's or 0's respectively.
    KernelArgs(KernelArgs const& kernelArgs, uint32_t minimumDimensionCount)
    :   autoPad(kernelArgs.autoPad),
        autoPadSameUpper(kernelArgs.autoPadSameUpper),
        channelsLast(kernelArgs.channelsLast),
        spatialDimensionCount(std::max(kernelArgs.spatialDimensionCount, minimumDimensionCount))
    {
        ML_CHECK_VALID_ARGUMENT(spatialDimensionCount <= NcdhwSpatialDimensionCount);

        uint32_t fillCount = (minimumDimensionCount > kernelArgs.spatialDimensionCount) ? minimumDimensionCount - kernelArgs.spatialDimensionCount : 0;
        FillWithLeadingValues(kernelArgs.strides, this->strides, fillCount, 1u);
        FillWithLeadingValues(kernelArgs.dilations, this->dilations, fillCount, 1u);
        FillWithLeadingValues(kernelArgs.windowSize, this->windowSize, fillCount, 1u);
        FillWithLeadingValues(kernelArgs.startPadding, this->startPadding, fillCount, 0u);
        FillWithLeadingValues(kernelArgs.endPadding, this->endPadding, fillCount, 0u);
        FillWithLeadingValues(kernelArgs.outputPadding, this->outputPadding, fillCount, 0u);
    }
};

std::vector<DimensionType> InitializeKernelOutputDimensions(
    gsl::span<const DimensionType> inputDimensions,
    const KernelArgs& args,
    bool isNhwc = false);

std::vector<DimensionType> InitializeKernelOutputDimsTranspose(
    gsl::span<const DimensionType> inputDimensions,
    const KernelArgs& args);

KernelArgs InitializeGlobalKernel(
        const MLOperatorAttributes& kernelInfo,
        gsl::span<const DimensionType> inputDimensions);

KernelArgs InitializeKernel(
    const MLOperatorAttributes& kernelInfo,
    uint32_t inputDimensionCount,
    gsl::span<const uint32_t> filterTensorShape);

void ResolveAutoPadding(
    KernelArgs& args,
    gsl::span<const DimensionType> inputDimensions,
    bool isNhwc = false);

void MatMulShapeMapping(
    std::vector<DimensionType>& inputShape0,
    std::vector<DimensionType>& inputShape1,
    std::vector<DimensionType>& outputShape);

void FusedMatMulShapeMapping(
    std::vector<DimensionType>& inputShape0,
    std::vector<DimensionType>& inputStride0,
    std::vector<DimensionType>& inputShape1,
    std::vector<DimensionType>& inputStride1,
    std::vector<DimensionType>& outputShape);

std::pair<std::vector<uint32_t>, std::vector<uint32_t>> GetFusedMatMulSizesAndStrides(
    gsl::span<const uint32_t> sizes,
    int32_t transBatch = 0,
    int32_t transpose = 0);

class GetOutputShapeAsInputShapeHelper
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    // Default to first input tensor.
    template <typename Info_t, typename Shape_t>
    GetOutputShapeAsInputShapeHelper(const Info_t& info, const Shape_t& shape)
    {
        ORT_UNUSED_PARAMETER(info);
        ORT_UNUSED_PARAMETER(shape);
    };

    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    // Pass specific tensor index.
    template <typename Info_t, typename Shape_t>
    GetOutputShapeAsInputShapeHelper(const Info_t& info, const Shape_t& shape, uint32_t inputTensorIndex)
        : m_inputTensorIndex(inputTensorIndex)
    {
        ORT_UNUSED_PARAMETER(info);
        ORT_UNUSED_PARAMETER(shape);
    };

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

    uint32_t m_inputTensorIndex = 0;
};

template <uint32_t InputTensorIndex>
class GetOutputShapeAsSpecificInputShapeHelper : public GetOutputShapeAsInputShapeHelper
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    GetOutputShapeAsSpecificInputShapeHelper(const Info_t& info, const Shape_t& shape)
        : GetOutputShapeAsInputShapeHelper(info, shape, InputTensorIndex)
    {}
};

class GetBroadcastedOutputShapeHelper
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    GetBroadcastedOutputShapeHelper(const Info_t& info, const Shape_t& shape) {};

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

struct IKernelInformationAdapter
{
    virtual uint32_t GetInputCount() const noexcept = 0;
    virtual uint32_t GetOutputCount() const noexcept = 0;
    virtual bool IsInputValid(uint32_t inputIndex) const noexcept = 0;
    virtual bool IsOutputValid(uint32_t outputIndex) const noexcept = 0;
    virtual MLOperatorTensor GetConstantInputTensor(uint32_t inputIndex) const = 0;
    virtual bool HasAttribute(_In_z_ MLConstStringParam name, MLOperatorAttributeType type) const noexcept = 0;
    virtual MLOperatorAttributes const& GetAttributes() const noexcept = 0;
    virtual MLOperatorEdgeDescription GetInputEdgeDescription(uint32_t inputIndex) const = 0;

    virtual ~IKernelInformationAdapter() {}
};

struct IShapeInformationAdapter
{
    virtual uint32_t GetInputTensorDimensionCount(uint32_t inputIndex) const = 0;
    virtual std::vector<uint32_t> GetInputTensorShape(uint32_t inputIndex) const = 0;
    virtual uint32_t GetSequenceInputCount(uint32_t inputIndex) const = 0;
    virtual MLOperatorTensorDataType GetSequenceInputDataType(uint32_t inputIndex) const = 0;
    virtual uint32_t GetSequenceInputTensorDimensionCount(uint32_t inputIndex, uint32_t sequenceIndex) const = 0;
    virtual std::vector<uint32_t> GetSequenceInputTensorShape(uint32_t inputIndex, uint32_t sequenceIndex) const = 0;
    virtual ~IShapeInformationAdapter() {}
};

// To avoid duplicating dozens of templated functions that vary only on the source of the kernel
// information, provide a thin abstraction.
//
// InformationSourceType may be MLOperatorKernelCreationContext or MLShapeInferenceContext.
template <typename InformationSourceType>
struct KernelInformationAdapter : IKernelInformationAdapter
{
    KernelInformationAdapter(InformationSourceType& informationSource) : m_informationSource(informationSource) {}

    virtual uint32_t GetInputCount() const noexcept { return m_informationSource.GetInputCount(); }
    virtual uint32_t GetOutputCount() const noexcept { return m_informationSource.GetOutputCount(); }
    virtual bool IsInputValid(uint32_t inputIndex) const noexcept { return m_informationSource.IsInputValid(inputIndex); }
    virtual bool IsOutputValid(uint32_t outputIndex) const noexcept { return m_informationSource.IsOutputValid(outputIndex); }
    virtual MLOperatorTensor GetConstantInputTensor(uint32_t inputIndex) const { return m_informationSource.GetConstantInputTensor(inputIndex); }
    virtual bool HasAttribute(_In_z_ MLConstStringParam name, MLOperatorAttributeType type) const noexcept { return m_informationSource.HasAttribute(name, type); }
    virtual MLOperatorAttributes const& GetAttributes() const noexcept { return m_informationSource; }
    virtual MLOperatorEdgeDescription GetInputEdgeDescription(uint32_t inputIndex) const { return m_informationSource.GetInputEdgeDescription(inputIndex); }
    virtual ~KernelInformationAdapter() {}

    InformationSourceType& m_informationSource;
};

// To avoid duplicating dozens of templated functions that vary only on the source of the kernel
// information, provide a thin abstraction (light enough to just be passed by value).
//
// InformationSourceType may be MLOperatorKernelCreationContext or MLShapeInferenceContext.
template <typename InformationSourceType>
struct ShapeInformationAdapter : IShapeInformationAdapter
{
    ShapeInformationAdapter(InformationSourceType& informationSource) : m_informationSource(informationSource) {}

    virtual uint32_t GetInputTensorDimensionCount(uint32_t inputIndex) const { return m_informationSource.GetInputTensorDimensionCount(inputIndex); }
    virtual std::vector<uint32_t> GetInputTensorShape(uint32_t inputIndex) const { return m_informationSource.GetInputTensorShape(inputIndex); }
    virtual uint32_t GetSequenceInputCount(uint32_t inputIndex) const { return m_informationSource.GetSequenceInputCount(inputIndex); }
    virtual MLOperatorTensorDataType GetSequenceInputDataType(uint32_t inputIndex) const { return m_informationSource.GetSequenceInputDataType(inputIndex); }
    virtual uint32_t GetSequenceInputTensorDimensionCount(uint32_t inputIndex, uint32_t sequenceIndex) const { return m_informationSource.GetSequenceInputTensorDimensionCount(inputIndex, sequenceIndex); }
    virtual std::vector<uint32_t> GetSequenceInputTensorShape(uint32_t inputIndex, uint32_t sequenceIndex) const { return m_informationSource.GetSequenceInputTensorShape(inputIndex, sequenceIndex); }
    virtual ~ShapeInformationAdapter() {}

    InformationSourceType& m_informationSource;
};

class RandomUniformHelperBase
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    template <typename Info_t>
    RandomUniformHelperBase(const Info_t& info)
    {
        m_high = info.GetOptionalAttribute<float>(AttrName::High, 1.0f);
        m_low = info.GetOptionalAttribute<float>(AttrName::Low, 0.0f);

        if (info.HasAttribute(AttrName::Seed, MLOperatorAttributeType::Float))
        {
            m_seed = info.GetAttribute<float>(AttrName::Seed);
        }
        else
        {
            m_seed = static_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        }
    }

protected:
    float m_high;
    float m_low;
    float m_seed;
};

class RandomUniformHelper : public RandomUniformHelperBase
{
public:
    template <typename Info_t, typename Shape_t>
    RandomUniformHelper(const Info_t& info, const Shape_t& shape) : RandomUniformHelperBase(info)
    {
        auto shapeAttribute = info.GetOptionalAttributeVectorInt32(AttrName::Shape);
        ML_CHECK_VALID_ARGUMENT(!shapeAttribute.empty(), "Attribute shape is missing.");
        m_tensorShape.assign(shapeAttribute.begin(), shapeAttribute.end());
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

private:
    // Returns an empty vector if the optional attribute is missing.
    std::vector<uint32_t> m_tensorShape;
};

class RandomNormalHelperBase
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    template <typename Info_t>
    RandomNormalHelperBase(const Info_t& info)
    {
        m_mean = info.GetOptionalAttribute<float>(AttrName::Mean, 0.0f);
        m_scale = info.GetOptionalAttribute<float>(AttrName::Scale, 1.0f);

        if (info.HasAttribute(AttrName::Seed, MLOperatorAttributeType::Float))
        {
            m_seed = info.GetAttribute<float>(AttrName::Seed);
        }
        else
        {
            m_seed = static_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
        }
    }

protected:
    float m_mean;
    float m_scale;
    float m_seed;
};

class RandomNormalHelper : public RandomNormalHelperBase
{
public:
    template <typename Info_t, typename Shape_t>
    RandomNormalHelper(const Info_t& info, const Shape_t& shape) : RandomNormalHelperBase(info)
    {
        auto shapeAttribute = info.GetOptionalAttributeVectorInt32(AttrName::Shape);
        ML_CHECK_VALID_ARGUMENT(!shapeAttribute.empty(), "Attribute shape is missing.");
        m_tensorShape.assign(shapeAttribute.begin(), shapeAttribute.end());
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

private:
    // Returns an empty vector if the optional attribute is missing.
    std::vector<uint32_t> m_tensorShape;
};

class ConvolutionHelperBase
{
public:
    enum FilterDims { K };
    enum InputDims { N, C, H, W };
    enum class NhwcInputDims { N, H, W, C };

public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    template<typename Info_t, typename Shape_t>
    ConvolutionHelperBase(const Info_t& info, const Shape_t& shape, bool transpose, bool hasDynamicPads, bool isNhwc, uint32_t inputTensorIndex, uint32_t filterTensorIndex) :
        m_inputTensorIndex(inputTensorIndex),
        m_filterTensorIndex(filterTensorIndex),
        m_isNhwc(isNhwc),
        m_kernel(InitializeKernel(info, shape.GetInputTensorDimensionCount(inputTensorIndex), shape.GetInputTensorShape(filterTensorIndex)))
    {
        m_groupCount = info.template GetOptionalAttribute<uint32_t>(AttrName::Group, 1);

        if (!transpose)
        {
            InitializeKernelAndShapes(ShapeInformationAdapter(shape));
        }
        else
        {
            InitializeKernelAndShapesTransposed(KernelInformationAdapter(info), ShapeInformationAdapter(shape), hasDynamicPads);
        }
    }

    void ResolvingPadding(gsl::span<const DimensionType> inputDimensions);

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

    void InitializeKernelAndShapes(const IShapeInformationAdapter& shapeInfo);

    void InitializeKernelAndShapesTransposed(
        const IKernelInformationAdapter& info,
        const IShapeInformationAdapter& shapeInfo,
        bool hasDynamicPads
    );

protected:
    uint32_t m_groupCount;
    uint32_t m_inputTensorIndex;
    uint32_t m_filterTensorIndex;
    bool m_isNhwc;
    KernelArgs m_kernel;
    std::vector<EdgeShapes> m_outputShapes;
};

class ConvHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    ConvHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, false, false, false, 0, 1) {}
};

class NhwcConvHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    NhwcConvHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, false, false, true, 0, 1) {}
};

class ConvTransposeHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    ConvTransposeHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, true, false, false, 0, 1) {}
};

class ConvTransposeWithDynamicPadsHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    ConvTransposeWithDynamicPadsHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, true, true, false, 0, 1) {}
};

class QLinearConvHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    QLinearConvHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, false, false, false, 0, 3) {}
};

class GemmHelper
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template<typename Info_t, typename Shape_t>
    GemmHelper(const Info_t& info, const Shape_t& shape)
    {
        ORT_UNUSED_PARAMETER(shape);
        m_transA = info.template GetOptionalAttribute<int>(AttrName::TransA, 0) != 0;
        m_transB = info.template GetOptionalAttribute<int>(AttrName::TransB, 0) != 0;
        m_broadcast = info.template GetOptionalAttribute<int>(AttrName::Broadcast, 0) != 0;
        m_alpha = info.template GetOptionalAttribute<float>(AttrName::Alpha, 1.0f);
        m_beta = info.template GetOptionalAttribute<float>(AttrName::Beta, 0.0f);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

    enum InputTensors
    {
        IN_A,
        IN_B,
        IN_C
    };

protected:
    bool m_transA = false;
    bool m_transB = false;
    bool m_broadcast = false;
    float m_alpha = 0.0f;
    float m_beta = 0.0f;
};

class TransposeHelper
{
public:
    void Initialize(
        const MLOperatorAttributes& operatorAttributes,
        gsl::span<const DimensionType> inputDimensions);

    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    TransposeHelper(const Info_t& info, const Shape_t& shape)
    {
        Initialize(info, shape.GetInputTensorShape(0));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    std::vector<int> m_permutations;
};

class SplitHelper
{
public:
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        uint32_t opsetVersion
    );

    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    SplitHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shape), opsetVersion);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    int m_axis = 0;
    std::vector<int> m_split;
};

class SliceHelper
{
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        uint32_t opsetVersion
    );

public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template<typename Info_t, typename Shape_t>
    SliceHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shape), opsetVersion);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    std::vector<DimensionType> m_outputDimensions;
    std::vector<uint32_t> m_offsets;
    std::vector<uint32_t> m_sizes;
    std::vector<int32_t> m_strides;
};

class PaddingHelper
{
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        uint32_t opsetVersion
    );

public:

    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    PaddingHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shape), opsetVersion);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    std::vector<uint32_t> m_startPadding;
    std::vector<uint32_t> m_endPadding;
};

template <typename OpsetHelper, uint32_t OpsetVersion>
class VersionedOpsetHelper : public OpsetHelper
{
public:
    template<typename Info_t, typename Shape_t>
    VersionedOpsetHelper(const Info_t& info, const Shape_t& shape) : OpsetHelper(info, shape, OpsetVersion) {}
};

class ReduceHelperBase
{
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        bool usingMultipleAxes
    );

public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    ReduceHelperBase(const Info_t& info, const Shape_t& shape, bool usingMultipleAxes)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shape), usingMultipleAxes);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    std::vector<int32_t> m_axes;
    int m_keepDims = 0; // Keep the dimensions rather than removing size 1 dimensions.
    int m_selectLastIndex = 0; // Prefer the higher index if there is a tie between element values.
    int m_noopWithEmptyAxes = 0; // Reduce nothing if axis list is empty.
};

class ArgMinArgMaxHelper : public ReduceHelperBase
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    ArgMinArgMaxHelper(const Info_t& info, const Shape_t& shape) : ReduceHelperBase(info, shape, false) {}
};

class ReduceHelper : public ReduceHelperBase
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    ReduceHelper(const Info_t& info, const Shape_t& shape) : ReduceHelperBase(info, shape, true) {}
};

class EinSumHelper
{
public:
    void Initialize();

    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    EinSumHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion)
    {
        m_equation = info.GetAttribute(AttrName::Equation);
        Initialize();
    }

    EinSumHelper(const MLOperatorAttributes& info)
    {
        m_equation = info.GetAttribute(AttrName::Equation);
        Initialize();
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

    enum class RecognizedOperatorType
    {
        None,
        Identity,
        Multiply,
        OuterProduct,
        MatMul,
        MatMulTransposeA,
        MatMulTransposeB,
        MatMulNhcw,
        MatMulNhcwTransposeA,
        MatMulNhcwTransposeB,
        ReduceSum,
        Transpose,
        Total,
    };

    RecognizedOperatorType GetRecognizedOperatorType() const noexcept { return m_recognizedOperatorType; }

    bool IsMatMulOperatorType() const noexcept;

protected:
    void ParseEquationComponents();
    RecognizedOperatorType DetermineRecognizedOperatorType();

protected:
    struct Component
    {
        uint32_t labelIndexBegin;
        uint32_t labelIndexEnd;

        uint32_t GetDimensionCount() const noexcept
        {
            return labelIndexEnd - labelIndexBegin;
        }
        gsl::span<const uint32_t> GetLabels(gsl::span<const uint32_t> labels) const
        {
            return labels.subspan(labelIndexBegin, labelIndexEnd - labelIndexBegin);
        };
    };

    std::string m_equation;
    std::vector<uint32_t> m_labelIndices; // Concatenation of all labels as rebased indices ("ij,ai" -> 0,1,2,0).
    std::vector<Component> m_components; // All components in order, including inputs and output.
    std::vector<uint32_t> m_outputDimensions;
    RecognizedOperatorType m_recognizedOperatorType = RecognizedOperatorType::None;
};

class MatMulHelperBase
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    MatMulHelperBase(const Info_t& info, const Shape_t& shape, uint32_t aTensorIndex, uint32_t bTensorIndex) :
        m_aTensorIndex(aTensorIndex),
        m_bTensorIndex(bTensorIndex)
    {}

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
protected:
    uint32_t m_aTensorIndex = 0;
    uint32_t m_bTensorIndex = 1;
};

class MatMulHelper : public MatMulHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    MatMulHelper(const Info_t& info, const Shape_t& shape) : MatMulHelperBase(info, shape, 0, 1) {}
};

class FusedMatMulHelper
{
public:
    template<typename Info_t, typename Shape_t>
    FusedMatMulHelper(const Info_t& info, const Shape_t& shape) {}

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class QLinearMatMulHelper : public MatMulHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    QLinearMatMulHelper(const Info_t& info, const Shape_t& shape) : MatMulHelperBase(info, shape, 0, 3) {}
};

class MatMulIntegerToFloatHelper : public MatMulHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    MatMulIntegerToFloatHelper(const Info_t& info, const Shape_t& shape) : MatMulHelperBase(info, shape, 0, 1) {}
};


class TopKHelper
{
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        uint32_t opsetVersion
    );

public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    TopKHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shape), opsetVersion);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    uint32_t m_k;
    uint32_t m_axis;
};

class RecurrentHelper
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    RecurrentHelper(const Info_t& info, const Shape_t& shape)
    {
        m_hiddenSize = info.template GetOptionalAttribute<int>(AttrName::HiddenSize, 1);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    int m_hiddenSize = 0;
};

class ConcatHelperBase
{
public:
    void Initialize(
        const MLOperatorAttributes& operatorAttributes,
        gsl::span<const DimensionType> inputDimensions
    );

    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    ConcatHelperBase(const Info_t& info, const Shape_t& shape, uint32_t firstInputIndex)
    {
        Initialize(info, shape.GetInputTensorShape(firstInputIndex));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo, uint32_t firstInputIndex, uint32_t step) const;

protected:
    int m_axis;
};

class ConcatHelper: public ConcatHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    ConcatHelper(const Info_t& info, const Shape_t& shape) : ConcatHelperBase(info, shape, 0) {}
    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class QLinearConcatHelper: public ConcatHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    QLinearConcatHelper(const Info_t& info, const Shape_t& shape) : ConcatHelperBase(info, shape, 2) {}
    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class CropHelper
{
public:
    enum BorderDim
    {
        Left,
        Top,
        Right,
        Bottom
    };
    enum ScaleDim
    {
        Height,
        Width
    };

    void Initialize(
        const MLOperatorAttributes& operatorAttributes,
        gsl::span<const DimensionType> inputDimensions
    );

    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    CropHelper(const Info_t& info, const Shape_t& shape)
    {
        Initialize(info, shape.GetInputTensorShape(0));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    uint32_t m_offsets[NchwDimensionCount];
    uint32_t m_sizes[NchwSpatialDimensionCount];
};

class DepthToSpaceHelper
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    DepthToSpaceHelper(const Info_t& info, const Shape_t& shape)
    {
        m_blockSize = info.template GetOptionalAttribute<int32_t>(AttrName::BlockSize, -1);
        ML_CHECK_VALID_ARGUMENT(m_blockSize > 0, "Attribute blocksize is missing or equal to zero.");
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    int32_t m_blockSize;
};

class SpaceToDepthHelper
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    SpaceToDepthHelper(const Info_t& info, const Shape_t& shape)
    {
        m_blockSize = info.template GetOptionalAttribute<int32_t>(AttrName::BlockSize, -1);
        ML_CHECK_VALID_ARGUMENT(m_blockSize > 0, "Attribute blocksize is missing or equal to zero.");
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    int32_t m_blockSize;
};

class FlattenHelper
{
public:
    void Initialize(
        const MLOperatorAttributes& operatorAttributes,
        gsl::span<const DimensionType> inputDimensions
    );

    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    FlattenHelper(const Info_t& info, const Shape_t& shape)
    {
        Initialize(info, shape.GetInputTensorShape(0));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    int m_axis = 1;
};

class MultinomialHelper
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    MultinomialHelper(const Info_t& info, const Shape_t& shape) {}

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class GatherHelper
{
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation
    );

public:

    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    GatherHelper(const Info_t& info, const Shape_t& shape)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shape));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    int m_axis = 0;
};

class GatherNdHelper
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    GatherNdHelper(const Info_t& info, const Shape_t& shape)
    {
        m_batchCount = info.template GetOptionalAttribute<int32_t>(AttrName::BatchDimensions, 0);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    int32_t m_batchCount;
};

class PoolingHelperBase
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    PoolingHelperBase(
        const Info_t& info,
        const Shape_t& shape,
        bool useGlobalPooling
    )
    :   m_kernel(useGlobalPooling
            ? InitializeGlobalKernel(info, shape.GetInputTensorShape(0))
            : InitializeKernel(info, static_cast<uint32_t>(shape.GetInputTensorShape(0).size()), gsl::span<uint32_t>()))
    {
        if (!useGlobalPooling)
        {
            ResolveAutoPadding(m_kernel, shape.GetInputTensorShape(0));
        }
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    KernelArgs m_kernel;
};

class UnpoolingHelper
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template<typename Info_t, typename Shape_t>
    UnpoolingHelper(
        const Info_t& info,
        const Shape_t& shape
    )
    :   m_inputShape(shape.GetInputTensorShape(0)),
        m_kernel(InitializeKernel(info, static_cast<uint32_t>(m_inputShape.size()), gsl::span<uint32_t>()))
    {
        Initialize();
    }

    void Initialize();

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    std::vector<DimensionType> m_inputShape;
    std::vector<DimensionType> m_inferredOutputDimensions;
    KernelArgs m_kernel;
};

class GlobalPoolingHelper : public PoolingHelperBase
{
public:
    template <typename Info_t, typename Shape_t>
    GlobalPoolingHelper(const Info_t& info, const Shape_t& shape) : PoolingHelperBase(info, shape, true) {}
};

class PoolingHelper : public PoolingHelperBase
{
public:
    template <typename Info_t, typename Shape_t>
    PoolingHelper(const Info_t& info, const Shape_t& shape) : PoolingHelperBase(info, shape, false) {}
};

class RoiPoolingHelperBase
{
public:
    enum InputTensors { INPUT, ROIS, BATCH_INDICES };

    RoiPoolingHelperBase()
    {}

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    uint32_t m_outputSizeW = 1;
    uint32_t m_outputSizeH = 1;
};

class RoiPoolingHelper : public RoiPoolingHelperBase
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    RoiPoolingHelper(const Info_t& info, const Shape_t& shape)
    {
        std::vector<int> pooledShape = info.GetOptionalAttributeVectorInt32(AttrName::PooledShape);
        ML_CHECK_VALID_ARGUMENT(pooledShape.size() == 2, "Pooled shape must be 2.");
        m_outputSizeH = pooledShape[0];
        m_outputSizeW = pooledShape[1];
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class RoiAlignHelper : public RoiPoolingHelperBase
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    RoiAlignHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion)
    {
        m_outputSizeW = info.template GetOptionalAttribute<uint32_t>(AttrName::OutputWidth, 1);
        m_outputSizeH = info.template GetOptionalAttribute<uint32_t>(AttrName::OutputHeight, 1);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class QLinearAveragePoolingHelper : public PoolingHelperBase
{
public:
    template <typename Info_t, typename Shape_t>
    QLinearAveragePoolingHelper(const Info_t& info, const Shape_t& shape) : PoolingHelperBase(info, shape, false) {}
    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

};

class QLinearGlobalAveragePoolingHelper : public PoolingHelperBase
{
public:
    template <typename Info_t, typename Shape_t>
    QLinearGlobalAveragePoolingHelper(const Info_t& info, const Shape_t& shape) : PoolingHelperBase(info, shape, true) {}
    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

};

class SqueezeHelper
{
public:
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        uint32_t opsetVersion
    );

    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    SqueezeHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shape), opsetVersion);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    std::vector<int> m_axes;
};

class UnsqueezeHelper
{
public:
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        uint32_t opsetVersion
    );

    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    UnsqueezeHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shape), opsetVersion);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    std::vector<int32_t> m_axes;
};

template <typename T>
void CALLBACK ShapeInferenceFunction(IMLOperatorShapeInferenceContext* inference_context)
{
    MLShapeInferenceContext helperContext(inference_context);
    T opHelper(helperContext, helperContext);

    // EdgeInfo to contain whether tensor, whether unused, and what shape is
    std::vector<EdgeShapes> outputShapes = opHelper.GetOutputShapes(helperContext);

    for (uint32_t i = 0; i < outputShapes.size(); ++i)
    {
        if (outputShapes[i].IsTensor() && !outputShapes[i].IsUnused())
        {
            helperContext.SetOutputTensorShape(i, outputShapes[i].GetShape());
        }
    }
}

class ReshapeHelper
{
    void Initialize(const IKernelInformationAdapter& kernelInformation);

public:
    template <typename Info_t, typename Shape_t>
    ReshapeHelper(const Info_t& info, const Shape_t& shape)
    {
        Initialize(KernelInformationAdapter(info));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    std::vector<int> m_shapeDims;
};

class ExpandHelper
{
public:
    template <typename Info_t, typename Shape_t>
    ExpandHelper(const Info_t& info, const Shape_t& shape)
    {
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
};

class ConstantOfShapeHelper
{
public:
    template <typename Info_t, typename Shape_t>
    ConstantOfShapeHelper(const Info_t& info, const Shape_t& shape)
    {
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
};

class TileHelper
{
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation
    );

public:
    template <typename Info_t, typename Shape_t>
    TileHelper(const Info_t& info, const Shape_t& shapeInfo)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shapeInfo));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    std::vector<uint32_t> m_repeatsData;
    std::vector<uint32_t> m_inputDimensions;
    std::vector<uint32_t> m_outputDimensions;
};

class ResizeHelper
{
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation,
        uint32_t opsetVersion
    );

public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    ResizeHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shape), opsetVersion);
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    std::vector<DimensionType> m_inputDimensions;
    std::vector<DimensionType> m_outputDimensions;
    std::vector<float> m_scales;
    std::vector<float> m_regionOfInterest; // Stored as [start1, ..., startN, end1, ..., endN], where N is the input rank.
};

class RangeHelper
{
public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later.
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
    template <typename Info_t, typename Shape_t>
    RangeHelper(const Info_t& info, const Shape_t& shape)
    {
        auto startTensor = info.GetConstantInputTensor(0);
        auto limitTensor = info.GetConstantInputTensor(1);
        auto deltaTensor = info.GetConstantInputTensor(2);
        Initialize(startTensor, limitTensor, deltaTensor);
    }

    void Initialize(
        const MLOperatorTensor& startTensor,
        const MLOperatorTensor& limitTensor,
        const MLOperatorTensor& deltaTensor
    );

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    std::vector<DimensionType> m_outputDimensions;

    MLOperatorTensorDataType m_tensorDataType = MLOperatorTensorDataType::Undefined;
    using TensorScalarData = typename std::aligned_storage_t<sizeof(double), alignof(double)>;
    TensorScalarData m_valueStart;
    TensorScalarData m_valueLimit;
    TensorScalarData m_valueDelta;
};

class OneHotHelper
{
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation
    );

public:
    template <typename Info_t, typename Shape_t>
    OneHotHelper(const Info_t& info, const Shape_t& shapeInfo)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shapeInfo));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    int32_t m_onnxAxis = 0;       // Original ONNX attribute value, including negative value.
    uint32_t m_absoluteAxis = 0;  // Absolute index value.
    std::vector<uint32_t> m_indicesDimensions;
    std::vector<uint32_t> m_outputDimensions;
};

class BatchNormalizationHelper
{
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation
    );

public:
    template <typename Info_t, typename Shape_t>
    BatchNormalizationHelper(const Info_t& info, const Shape_t& shapeInfo)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shapeInfo));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class ShapeHelper {
public:
    template <typename Info_t, typename Shape_t>
    ShapeHelper(const Info_t& info, const Shape_t& shapeInfo)
    {
        Initialize(KernelInformationAdapter(info), ShapeInformationAdapter(shapeInfo));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

protected:
    uint32_t m_sliceStart = 0;
    uint32_t m_sliceEnd = 0;

private:
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation
    );
};

class SizeHelper {
public:
    template <typename Info_t, typename Shape_t>
    SizeHelper(const Info_t& info, const Shape_t& shapeInfo) { }
    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class EmbedLayerNormalizationHelper
{
    void Initialize(
        const IKernelInformationAdapter& kernelInformation,
        const IShapeInformationAdapter& shapeInformation
    );

public:
    template <typename Info_t, typename Shape_t>
    EmbedLayerNormalizationHelper(const Info_t& info, const Shape_t& shapeInfo) { }
    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class MultiHeadAttentionHelper
{
public:
    template <typename Info_t, typename Shape_t>
    MultiHeadAttentionHelper(const Info_t& info, const Shape_t& shapeInfo)
    {
        Initialize(KernelInformationAdapter(info));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

private:
    void Initialize(const IKernelInformationAdapter& kernelInformation);
    uint32_t m_numHeads;
};

class AttentionHelper
{
public:
    template <typename Info_t, typename Shape_t>
    AttentionHelper(const Info_t& info, const Shape_t& shapeInfo)
    {
        Initialize(KernelInformationAdapter(info));
    }

    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

private:
    void Initialize(const IKernelInformationAdapter& kernelInformation);
    std::vector<int32_t> m_qkvHiddenSizes;
};

class SkipLayerNormHelper
{
public:
    template <typename Info_t, typename Shape_t>
    SkipLayerNormHelper(const Info_t& info, const Shape_t& shapeInfo) {}
    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class BiasSplitGeluHelper {
public:
    template <typename Info_t, typename Shape_t>
    BiasSplitGeluHelper(const Info_t& info, const Shape_t& shapeInfo) { }
    std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

using ShapeInferenceHelper_Conv = ConvHelper;
using ShapeInferenceHelper_NhwcConv = NhwcConvHelper;
using ShapeInferenceHelper_ConvTranspose = ConvTransposeHelper;
using ShapeInferenceHelper_ConvTransposeWithDynamicPads = ConvTransposeWithDynamicPadsHelper;
using ShapeInferenceHelper_ConvInteger = ConvHelper;
using ShapeInferenceHelper_QLinearConv = QLinearConvHelper;
using ShapeInferenceHelper_AveragePool = PoolingHelper;
using ShapeInferenceHelper_GlobalAveragePool = GlobalPoolingHelper;
using ShapeInferenceHelper_MaxPool = PoolingHelper;
using ShapeInferenceHelper_GlobalMaxPool = GlobalPoolingHelper;
using ShapeInferenceHelper_MaxUnpool = UnpoolingHelper;
using ShapeInferenceHelper_LpPool = PoolingHelper;
using ShapeInferenceHelper_GlobalLpPool = GlobalPoolingHelper;
using ShapeInferenceHelper_MaxRoiPool = RoiPoolingHelper;
using ShapeInferenceHelper_QLinearAveragePool = QLinearAveragePoolingHelper;
using ShapeInferenceHelper_QLinearGlobalAveragePool = QLinearGlobalAveragePoolingHelper;
using ShapeInferenceHelper_RoiAlign10 = VersionedOpsetHelper<RoiAlignHelper, 10>;
using ShapeInferenceHelper_RoiAlign16 = VersionedOpsetHelper<RoiAlignHelper, 16>;
using ShapeInferenceHelper_InstanceNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_BatchNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_BatchNormalization15 = BatchNormalizationHelper;

using ShapeInferenceHelper_LRN = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_MeanVarianceNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_GroupNorm = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_LayerNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_LayerNormalization17 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_SkipLayerNormalization = SkipLayerNormHelper;
using ShapeInferenceHelper_EmbedLayerNormalization = EmbedLayerNormalizationHelper;
using ShapeInferenceHelper_LpNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_RNN = RecurrentHelper;
using ShapeInferenceHelper_GRU = RecurrentHelper;
using ShapeInferenceHelper_LSTM = RecurrentHelper;
using ShapeInferenceHelper_BiasAdd = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_Gather = GatherHelper;
using ShapeInferenceHelper_GatherElements = GetOutputShapeAsSpecificInputShapeHelper<1>;
using ShapeInferenceHelper_ScatterElements = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Scatter9 = ShapeInferenceHelper_ScatterElements; // Old deprecated alias for ScatterElements.
using ShapeInferenceHelper_Scatter11 = ShapeInferenceHelper_ScatterElements; // Old deprecated alias for ScatterElements.
using ShapeInferenceHelper_Scatter13 = ShapeInferenceHelper_ScatterElements; // Old deprecated alias for ScatterElements.
using ShapeInferenceHelper_GatherND = GatherNdHelper;
using ShapeInferenceHelper_ScatterND = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_Flatten7 = FlattenHelper;
using ShapeInferenceHelper_Flatten9 = FlattenHelper;
using ShapeInferenceHelper_Flatten11 = FlattenHelper;
using ShapeInferenceHelper_Flatten13 = FlattenHelper;
using ShapeInferenceHelper_Split7 = VersionedOpsetHelper<SplitHelper, 7>;
using ShapeInferenceHelper_Split11 = VersionedOpsetHelper<SplitHelper, 11>;
using ShapeInferenceHelper_Split13 = VersionedOpsetHelper<SplitHelper, 13>;
using ShapeInferenceHelper_Split18 = VersionedOpsetHelper<SplitHelper, 18>;
using ShapeInferenceHelper_Transpose = TransposeHelper;
using ShapeInferenceHelper_Concat = ConcatHelper;
using ShapeInferenceHelper_QLinearConcat = QLinearConcatHelper;
using ShapeInferenceHelper_Slice7 = VersionedOpsetHelper<SliceHelper, 7>;
using ShapeInferenceHelper_Slice10 = VersionedOpsetHelper<SliceHelper, 10>;
using ShapeInferenceHelper_Slice11 = VersionedOpsetHelper<SliceHelper, 11>; // Note 11 and 10 are identical - no functional change.
using ShapeInferenceHelper_Slice13 = VersionedOpsetHelper<SliceHelper, 13>; // Note 13 and 10 are identical - no functional change, just new types.
using ShapeInferenceHelper_Pad7 = VersionedOpsetHelper<PaddingHelper, 7>;
using ShapeInferenceHelper_Pad11 = VersionedOpsetHelper<PaddingHelper, 11>;
using ShapeInferenceHelper_Pad13 = VersionedOpsetHelper<PaddingHelper, 13>;
using ShapeInferenceHelper_Pad18 = VersionedOpsetHelper<PaddingHelper, 18>;

using ShapeInferenceHelper_SpaceToDepth = SpaceToDepthHelper;
using ShapeInferenceHelper_DepthToSpace = DepthToSpaceHelper;
using ShapeInferenceHelper_Squeeze7 = VersionedOpsetHelper<SqueezeHelper, 7>;
using ShapeInferenceHelper_Squeeze11 = VersionedOpsetHelper<SqueezeHelper, 11>;
using ShapeInferenceHelper_Squeeze13 = VersionedOpsetHelper<SqueezeHelper, 13>;
using ShapeInferenceHelper_Unsqueeze7 = VersionedOpsetHelper<UnsqueezeHelper, 7>;
using ShapeInferenceHelper_Unsqueeze11 = VersionedOpsetHelper<UnsqueezeHelper, 11>;
using ShapeInferenceHelper_Unsqueeze13 = VersionedOpsetHelper<UnsqueezeHelper, 13>;
using ShapeInferenceHelper_EyeLike = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Trilu = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_Expand = ExpandHelper;
using ShapeInferenceHelper_Reshape7 = ReshapeHelper;
using ShapeInferenceHelper_Reshape13 = ReshapeHelper;
using ShapeInferenceHelper_Reshape14 = ReshapeHelper;
using ShapeInferenceHelper_ConstantOfShape = ConstantOfShapeHelper;
using ShapeInferenceHelper_Tile = TileHelper;
using ShapeInferenceHelper_Resize10 = VersionedOpsetHelper<ResizeHelper, 10>;
using ShapeInferenceHelper_Resize11 = VersionedOpsetHelper<ResizeHelper, 11>;
using ShapeInferenceHelper_Resize13 = VersionedOpsetHelper<ResizeHelper, 13>;
using ShapeInferenceHelper_OneHot = OneHotHelper;

using ShapeInferenceHelper_Sqrt = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Reciprocal = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Pow = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Exp = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Log = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Abs = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Ceil = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Floor = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Clip7 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Clip11 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Clip12 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Clip13 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Greater = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Less = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_GreaterOrEqual = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_LessOrEqual = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Equal = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Not = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_And = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Or = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Xor = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Add = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Sub = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Mul = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Div = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Sum = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Mean = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Max = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Min = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Cos = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Sin = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Tan = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Acos = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Asin = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Atan = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Affine = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_QuantizeLinear = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_DequantizeLinear = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_QLinearSigmoid = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Attention = AttentionHelper;
using ShapeInferenceHelper_MultiHeadAttention = MultiHeadAttentionHelper;
using ShapeInferenceHelper_Sign = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_IsNaN = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Erf = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Sinh = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Cosh = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Asinh = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Acosh = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Atanh = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Where = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_IsInf = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Mod = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_BitShift= GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Round = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_QuickGelu = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_BitwiseAnd = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_BitwiseOr = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_BitwiseXor = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_BitwiseNot = GetBroadcastedOutputShapeHelper;

using ShapeInferenceHelper_ReduceSum = ReduceHelper;
using ShapeInferenceHelper_ReduceMean = ReduceHelper;
using ShapeInferenceHelper_ReduceProd = ReduceHelper;
using ShapeInferenceHelper_ReduceLogSum = ReduceHelper;
using ShapeInferenceHelper_ReduceLogSumExp = ReduceHelper;
using ShapeInferenceHelper_ReduceSumSquare = ReduceHelper;
using ShapeInferenceHelper_ReduceL1 = ReduceHelper;
using ShapeInferenceHelper_ReduceL2 = ReduceHelper;
using ShapeInferenceHelper_ReduceMax = ReduceHelper;
using ShapeInferenceHelper_ReduceMin = ReduceHelper;
using ShapeInferenceHelper_Einsum12 = VersionedOpsetHelper<EinSumHelper, 12>;
using ShapeInferenceHelper_ArgMax = ArgMinArgMaxHelper;
using ShapeInferenceHelper_ArgMin = ArgMinArgMaxHelper;
using ShapeInferenceHelper_Gemm = GemmHelper;
using ShapeInferenceHelper_Neg = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_Crop = CropHelper;
using ShapeInferenceHelper_ImageScaler = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Upsample7 = VersionedOpsetHelper<ResizeHelper, 7>;
using ShapeInferenceHelper_Upsample9 = VersionedOpsetHelper<ResizeHelper, 9>;
using ShapeInferenceHelper_Upsample10 = VersionedOpsetHelper<ResizeHelper, 10>;

using ShapeInferenceHelper_Sigmoid = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_HardSigmoid = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Tanh = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_ScaledTanh = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Relu = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_LeakyRelu = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_PRelu = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_ThresholdedRelu = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Elu = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Celu = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Selu = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Softmax = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Softmax13 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_LogSoftmax = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_LogSoftmax13 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Hardmax = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Hardmax13 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Softsign = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Softplus = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_ParametricSoftplus = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Dropout = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Shrink = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Gelu = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_BiasGelu = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_Identity7 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Identity13 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Identity14 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Identity16 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_MatMul = MatMulHelper;
using ShapeInferenceHelper_MatMulInteger = MatMulHelper;
using ShapeInferenceHelper_DynamicQuantizeMatMul = MatMulHelper;
using ShapeInferenceHelper_MatMulIntegerToFloat = MatMulIntegerToFloatHelper;
using ShapeInferenceHelper_QLinearMatMul = QLinearMatMulHelper;
using ShapeInferenceHelper_QLinearAdd = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_DynamicQuantizeLinear = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_Cast = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_MemcpyFromHost = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_MemcpyToHost = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_TopK7 = VersionedOpsetHelper<TopKHelper, 7>;
using ShapeInferenceHelper_TopK10 = VersionedOpsetHelper<TopKHelper, 10>;
using ShapeInferenceHelper_TopK11 = VersionedOpsetHelper<TopKHelper, 11>;

using ShapeInferenceHelper_RandomUniform = RandomUniformHelper;
using ShapeInferenceHelper_RandomUniformLike = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_RandomNormal = RandomNormalHelper;
using ShapeInferenceHelper_RandomNormalLike = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Multinomial = MultinomialHelper;

using ShapeInferenceHelper_ReverseSequence = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_CumSum11 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_CumSum14 = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Range = RangeHelper;

using ShapeInferenceHelper_CastLike15 = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_DmlFusedConv = ConvHelper;
using ShapeInferenceHelper_DmlFusedConvTranspose = ConvTransposeHelper;
using ShapeInferenceHelper_DmlFusedInstanceNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_DmlFusedBatchNormalization = BatchNormalizationHelper;
using ShapeInferenceHelper_DmlFusedMeanVarianceNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_DmlFusedGemm = GemmHelper;
using ShapeInferenceHelper_DmlFusedMatMul = MatMulHelper;
using ShapeInferenceHelper_FusedMatMul = FusedMatMulHelper;
using ShapeInferenceHelper_FusedMatMulActivation = FusedMatMulHelper;
using ShapeInferenceHelper_DmlFusedAdd = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_DmlFusedSum = GetBroadcastedOutputShapeHelper;

using ShapeInferenceHelper_Shape = ShapeHelper;
using ShapeInferenceHelper_Size = SizeHelper;
using ShapeInferenceHelper_BiasSplitGelu = BiasSplitGeluHelper;

}  // namespace OperatorHelper

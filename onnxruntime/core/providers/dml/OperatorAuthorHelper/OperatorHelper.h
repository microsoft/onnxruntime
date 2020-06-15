// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "Common.h"
#include "Attributes.h"
#include "MLOperatorAuthorHelper.h"
#include "core/common/common.h"

namespace OperatorHelper {
bool ContainsEmptyDimensions(gsl::span<const DimensionType> dimensions);

std::vector<DimensionType> BroadcastTensorShape(
    gsl::span<const DimensionType> inputShape0,
    gsl::span<const DimensionType> inputShape1);

// Find all the occurrences of a value, and return the array indices (in ascending order).
//
// e.g. input values = {2,1,3,1,1,5}
//      value = 1
//      output indices = {1,3,4}
#pragma optimize("", off)
template <typename T>
void FindValueIndices(gsl::span<const T> values, T value, /*out*/ std::vector<uint32_t>& indices) {
  indices.clear();
  for (size_t i = 0, valuesCount = values.size(); i < valuesCount; ++i) {
    // Work around compiler bug on x86 release by using data() rather than operator [] directly.
    // cl.exe 19.20.27412.4 for x86
    if (values.data()[i] == value) {
      indices.push_back(gsl::narrow_cast<uint32_t>(i));
    }
  }
}
#pragma optimize("", on)

// Convert any negative axis into an absolute axis relative to the back end.
// So given 3 dimensions, a -1 refers to axis 2, and -3 to axis 0.
uint32_t HandleNegativeAxis(int32_t signedOnnxAxis, uint32_t dimCount);

void HandleNegativeAxes(gsl::span<int32_t> onnxAxes, uint32_t dimCount);

// Remove array entries of the given indices (in ascending order), shifting them toward the front.
// There is a special check to avoid removing all the values, since returning a completely
// empty array would frequently causes errors later in many uses (such as with dimensions).
//
// e.g. input values = {2,1,3,1,1,5}
//      ellidable input indices = {1,3,4}
//      output values = {2,3,5}
template <typename T>
void RemoveValuesByIndex(gsl::span<const uint32_t> indices, bool keepOneValue, /*inout*/ std::vector<T>& values) {
  assert(std::is_sorted(indices.begin(), indices.end()));

  // Keep the last value at least, if all values would otherwise be removed.
  if (keepOneValue && !indices.empty() && indices.size() == values.size()) {
    indices = indices.first(indices.size() - 1);
  }

  auto indicesIterator = indices.begin();
  auto indicesEnd = indices.end();
  size_t oldValuesCount = values.size();
  size_t newValuesCount = 0;
  size_t nextIndex = (indicesIterator == indicesEnd) ? SIZE_MAX : *(indicesIterator++);

  // For every value, either skip the entry, or copy it to the output.
  for (size_t i = 0; i < oldValuesCount; ++i) {
    if (i == nextIndex)  // Skip and remove entry.
    {
      nextIndex = (indicesIterator == indicesEnd) ? SIZE_MAX : *(indicesIterator++);
    } else  // Keep and copy entry.
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

class EdgeShapes {
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

struct KernelArgs {
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
  uint32_t spatialDimensionCount = 0;

  KernelArgs(uint32_t spatialDimensionCount) : spatialDimensionCount(spatialDimensionCount)
  {
    ML_CHECK_VALID_ARGUMENT(spatialDimensionCount <= NcdhwSpatialDimensionCount);
  }

  void FillWithLeadingValues(gsl::span<const uint32_t> input, gsl::span<uint32_t> output, uint32_t fillCount, uint32_t value) {
    // e.g.
    // input = [5,6,7,8]
    // fillcount = 2
    // value = 1
    // output = [1,1,5,6,7,8]

    const size_t inputCount = input.size();
    const size_t outputCount = output.size();
    const size_t clampedFillCount = std::min(size_t(fillCount), outputCount);
    const size_t copyCount = std::min(outputCount - fillCount, inputCount);

    std::fill_n(output.data(), fillCount, value);
    std::copy_n(input.data(), copyCount, output.data() + fillCount);
  }

  // Create a copy of an existing kernel args with a minimum dimension count,
  // filling the leading attribute values with 1's or 0's respectively.
  KernelArgs(KernelArgs const& kernelArgs, uint32_t minimumDimensionCount) : autoPad(kernelArgs.autoPad),
                                                                             autoPadSameUpper(kernelArgs.autoPadSameUpper),
                                                                             spatialDimensionCount(std::max(kernelArgs.spatialDimensionCount, minimumDimensionCount)) {
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
    const KernelArgs& args);

std::vector<DimensionType> InitializeKernelOutputDimsTranspose(
    gsl::span<const DimensionType> inputDimensions,
    const KernelArgs& args);

KernelArgs InitializeGlobalKernel(gsl::span<const DimensionType> inputDimensions);

KernelArgs InitializeKernel(
    const MLOperatorAttributes& kernelInfo,
    uint32_t inputDimensionCount,
    gsl::span<const uint32_t> filterTensorShape);

void ResolveAutoPadding(
    KernelArgs& args,
    gsl::span<const DimensionType> inputDimensions);

void MatMulShapeMapping(
  std::vector<DimensionType>& inputShape0,
  std::vector<DimensionType>& inputShape1,
  std::vector<DimensionType>& outputShape);

class GetOutputShapeAsInputShapeHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  // Default to first input tensor.
  template <typename Info_t, typename Shape_t>
  GetOutputShapeAsInputShapeHelper(const Info_t& info, const Shape_t& shape){
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
class GetOutputShapeAsSpecificInputShapeHelper : public GetOutputShapeAsInputShapeHelper {
public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  GetOutputShapeAsSpecificInputShapeHelper(const Info_t& info, const Shape_t& shape)
  : GetOutputShapeAsInputShapeHelper(info, shape, InputTensorIndex)
  {}
};

class GetBroadcastedOutputShapeHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  GetBroadcastedOutputShapeHelper(const Info_t& info, const Shape_t& shape){};

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class RandomUniformHelperBase {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  template <typename Info_t>
  RandomUniformHelperBase(const Info_t& info) {
    m_high = info.GetOptionalAttribute<float>(AttrName::High, 1.0f);
    m_low = info.GetOptionalAttribute<float>(AttrName::Low, 0.0f);

    if (info.HasAttribute(AttrName::Seed, MLOperatorAttributeType::Float)) {
      m_seed = info.GetAttribute<float>(AttrName::Seed);
    } else {
      m_seed = static_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }
  }

 protected:
  float m_high;
  float m_low;
  float m_seed;
};

class RandomUniformHelper : public RandomUniformHelperBase {
 public:
  template <typename Info_t, typename Shape_t>
  RandomUniformHelper(const Info_t& info, const Shape_t& shape) : RandomUniformHelperBase(info) {
    auto shapeAttribute = info.GetOptionalAttributeVectorInt32(AttrName::Shape);
    ML_CHECK_VALID_ARGUMENT(!shapeAttribute.empty(), "Attribute shape is missing.");
    m_tensorShape.assign(shapeAttribute.begin(), shapeAttribute.end());
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 private:
  // Returns an empty vector if the optional attribute is missing.
  std::vector<uint32_t> m_tensorShape;
};

class RandomNormalHelperBase {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  template <typename Info_t>
  RandomNormalHelperBase(const Info_t& info) {
    m_mean = info.GetOptionalAttribute<float>(AttrName::Mean, 0.0f);
    m_scale = info.GetOptionalAttribute<float>(AttrName::Scale, 1.0f);

    if (info.HasAttribute(AttrName::Seed, MLOperatorAttributeType::Float)) {
      m_seed = info.GetAttribute<float>(AttrName::Seed);
    } else {
      m_seed = static_cast<float>(std::chrono::high_resolution_clock::now().time_since_epoch().count());
    }
  }

 protected:
  float m_mean;
  float m_scale;
  float m_seed;
};

class RandomNormalHelper : public RandomNormalHelperBase {
 public:
  template <typename Info_t, typename Shape_t>
  RandomNormalHelper(const Info_t& info, const Shape_t& shape) : RandomNormalHelperBase(info) {
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

public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later. 
    template<typename Info_t, typename Shape_t>
    ConvolutionHelperBase(const Info_t& info, const Shape_t& shape, bool transpose, bool hasDynamicPads, uint32_t inputTensorIndex, uint32_t filterTensorIndex) :
        m_kernel(InitializeKernel(info, shape.GetInputTensorDimensionCount(inputTensorIndex), shape.GetInputTensorShape(filterTensorIndex))),
        m_inputTensorIndex(inputTensorIndex),
        m_filterTensorIndex(filterTensorIndex)
    {
        m_groupCount = info.GetOptionalAttribute<uint32_t>(AttrName::Group, 1);
        
        if (!transpose)
        {
            InitializeKernelAndShapes(shape);
        }
        else
        {
            InitializeKernelAndShapesTransposed(info, shape, hasDynamicPads);
        }
    }

  void ResolvingPadding(gsl::span<const DimensionType> inputDimensions);

  const std::vector<EdgeShapes>& GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const {
    ORT_UNUSED_PARAMETER(shapeInfo);
    return m_outputShapes;
  }

  template <typename Shape_t>
  void InitializeKernelAndShapes(const Shape_t& shapeInfo) {
    const std::vector<DimensionType> inputDimensions = shapeInfo.GetInputTensorShape(m_inputTensorIndex);
    const std::vector<DimensionType> filterDims = shapeInfo.GetInputTensorShape(m_filterTensorIndex);

        ML_CHECK_VALID_ARGUMENT(
            inputDimensions.size() >= 3 && inputDimensions.size() <= 5,
            "Input dimensions must be: 3, 4, 5."
        );
        
        ResolvingPadding(inputDimensions);
        
        m_outputShapes.resize(1);
        m_outputShapes[0] = InitializeKernelOutputDimensions(inputDimensions, m_kernel);
        m_outputShapes[0].GetShape()[C] = filterDims[K];
    }
    
    
    template<typename Info_t, typename Shape_t>
    void InitializeKernelAndShapesTransposed(const Info_t& info, const Shape_t& shapeInfo, bool hasDynamicPads)
    {
        std::vector<int> outputShape = info.GetOptionalAttributeVectorInt32(AttrName::OutputShape);
        if (!outputShape.empty())
        {
            ML_CHECK_VALID_ARGUMENT(
                outputShape.size() >= m_kernel.spatialDimensionCount,
                "The output shape must equal the number of spatial dimensions"
            );
        }

        const std::vector<DimensionType> inputDimensions = shapeInfo.GetInputTensorShape(m_inputTensorIndex);
        const std::vector<DimensionType> filterDims = shapeInfo.GetInputTensorShape(m_filterTensorIndex);

        ML_CHECK_VALID_ARGUMENT(inputDimensions.size() > NonspatialDimensionCount, "Input dimensions must be >= 3");

        if (hasDynamicPads)
        {
            MLOperatorTensor padsTensor = info.GetConstantInputTensor(2);
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

        if (!outputShape.empty()) {
            // Start padding, end padding, and output padding are all ignored if output shape is set.
            std::fill(m_kernel.outputPadding, m_kernel.outputPadding + m_kernel.spatialDimensionCount, 0);

            if (outputShape.size() > 2) {
                ML_CHECK_VALID_ARGUMENT(outputShape[outputShape.size() - 3] == gsl::narrow_cast<int>(m_outputShapes[0].GetShape()[C]), "Output channel must be equivalent to filter channel.");
            }

            for (size_t i = 0; i < m_kernel.spatialDimensionCount; ++i) {
                size_t outputIndex = outputShape.size() - m_kernel.spatialDimensionCount + i;
                ML_CHECK_VALID_ARGUMENT(outputShape[outputIndex] >= gsl::narrow_cast<int>(inputDimensions[H + i]), "Output dimension cannot be smaller than input dimension.");
                m_outputShapes[0].GetShape()[H + i] = outputShape[outputIndex];
            }

            const int dimOffset = gsl::narrow_cast<int>(inputDimensions.size() - m_kernel.spatialDimensionCount);

            for (size_t i = 0; i < m_kernel.spatialDimensionCount; ++i) {
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

                m_kernel.startPadding[i] = m_kernel.autoPadSameUpper ? (paddings + 1) / 2 : paddings / 2;
                m_kernel.endPadding[i] = paddings - m_kernel.startPadding[i];
            }
        }
  }

 protected:
  uint32_t m_groupCount;
  uint32_t m_inputTensorIndex;
  uint32_t m_filterTensorIndex;
  KernelArgs m_kernel;
  std::vector<EdgeShapes> m_outputShapes;
};

class ConvHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    ConvHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, false, false, 0, 1) {}
};

class ConvTransposeHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    ConvTransposeHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, true, false, 0, 1) {}
};

class ConvTransposeWithDynamicPadsHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    ConvTransposeWithDynamicPadsHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, true, true, 0, 1) {}
};

class QLinearConvHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    QLinearConvHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, false, false, 0, 3) {}
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
        m_transA = info.GetOptionalAttribute<int>(AttrName::TransA, 0);
        m_transB = info.GetOptionalAttribute<int>(AttrName::TransB, 0);
        m_broadcast = info.GetOptionalAttribute<int>(AttrName::Broadcast, 0);
        m_alpha = info.GetOptionalAttribute<float>(AttrName::Alpha, 1.0f);
        m_beta = info.GetOptionalAttribute<float>(AttrName::Beta, 0.0f);
    }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

  enum InputTensors { IN_A,
                      IN_B,
                      IN_C };

 protected:
  bool m_transA = false;
  bool m_transB = false;
  bool m_broadcast = false;
  float m_alpha = 0.0f;
  float m_beta = 0.0f;
};

class TransposeHelper {
 public:
  void Initialize(
      const MLOperatorAttributes& operatorAttributes,
      gsl::span<const DimensionType> inputDimensions);

  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  TransposeHelper(const Info_t& info, const Shape_t& shape) {
    Initialize(info, shape.GetInputTensorShape(0));
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  std::vector<int> m_permutations;
};

class SplitHelper {
 public:
  void Initialize(
      const MLOperatorAttributes& operatorAttributes,
      gsl::span<const DimensionType> inputDimensions);

  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  SplitHelper(const Info_t& info, const Shape_t& shape) {
    Initialize(info, shape.GetInputTensorShape(0));
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  int m_axis = 0;
  std::vector<int> m_split;
};

class SliceHelper
{
public:
    template<typename Info_t>
    void ReadIndexTensors(
        const Info_t& operatorInfo,
        /*out*/ std::vector<int32_t>& starts,
        /*out*/ std::vector<int32_t>& ends,
        /*out*/ std::vector<int32_t>& axes,
        /*out*/ std::vector<int32_t>& steps
        )
    {
        // Get starts, ends, optional axes, and optional steps from constant inputs.
        ReadCpuLocalTensorIntoInt32(operatorInfo.GetConstantInputTensor(1), /*out*/ starts);
        ReadCpuLocalTensorIntoInt32(operatorInfo.GetConstantInputTensor(2), /*out*/ ends);
        if (operatorInfo.IsInputValid(3))
        {
            ReadCpuLocalTensorIntoInt32(operatorInfo.GetConstantInputTensor(3), /*out*/ axes);
        }
        if (operatorInfo.IsInputValid(4))
        {
            ReadCpuLocalTensorIntoInt32(operatorInfo.GetConstantInputTensor(4), /*out*/ steps);
        }
    }

    template<typename Info_t>
    void Initialize(
        const Info_t& operatorInfo,
        gsl::span<const DimensionType> inputDimensions,
        uint32_t opsetVersion
        )
    {
        std::vector<int32_t> starts;
        std::vector<int32_t> ends;
        std::vector<int32_t> axes;
        std::vector<int32_t> steps;

        if (opsetVersion == 7)
        {
            // Read starts, ends, and axes from attributes.
            starts = operatorInfo.GetOptionalAttributeVectorInt32(AttrName::Starts);
            ends = operatorInfo.GetOptionalAttributeVectorInt32(AttrName::Ends);
            axes = operatorInfo.GetOptionalAttributeVectorInt32(AttrName::Axes);
        }
        else if (opsetVersion == 10 || opsetVersion == 11)
        {
            // Read starts, ends, and axes from tensors.
            ReadIndexTensors(operatorInfo, /*out*/ starts, /*out*/ ends, /*out*/ axes, /*out*/ steps);
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
            ML_CHECK_VALID_ARGUMENT(dimIndex < inputDimensions.size(), "'axes' must be valid with within actual input dimensions.");
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

    // Info_t is used to obtain attributes which will be used for calculating the output shape later. 
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value. 
    template<typename Info_t, typename Shape_t>
    SliceHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion)
    {
        Initialize(info, shape.GetInputTensorShape(0), opsetVersion);
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
public:
    void Initialize(const MLOperatorAttributes& operatorAttributes, gsl::span<int32_t> padding, uint32_t opsetVersion);

  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  PaddingHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion) {
    std::vector<int32_t> padding;
    if (opsetVersion >= 11)
    {
        MLOperatorTensor padsTensor = info.GetConstantInputTensor(1);
        ReadCpuLocalTensorIntoInt32(padsTensor, /*out*/ padding);
    }
    else
    {
        padding = info.GetOptionalAttributeVectorInt32(AttrName::Pads);
    }

    Initialize(info, padding, opsetVersion);
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

class ReduceHelperBase {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  ReduceHelperBase(const Info_t& info, const Shape_t& shape, bool usingAxes) {
    m_keepDims = info.GetOptionalAttribute<int>(AttrName::KeepDims, 1);
    if (usingAxes) {
      m_axes = info.GetOptionalAttributeVectorInt32(AttrName::Axes);
    } else {
      int axis = info.GetOptionalAttribute<int>(AttrName::Axis, 0);
      m_axes.push_back(axis);
    }
    std::vector<uint32_t> inputShape = shape.GetInputTensorShape(0);
    HandleNegativeAxes(/*inout*/ m_axes, gsl::narrow_cast<uint32_t>(inputShape.size()));
    AdjustAxesAndOutputShape(inputShape);
  }
  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 private:
  void AdjustAxesAndOutputShape(const std::vector<uint32_t>& inputShape);

 protected:
  std::vector<int> m_axes;
  int m_keepDims = 0;
};

class ArgMinArgMaxHelper : public ReduceHelperBase {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  ArgMinArgMaxHelper(const Info_t& info, const Shape_t& shape) : ReduceHelperBase(info, shape, false) {}
};

class ReduceHelper : public ReduceHelperBase {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  ReduceHelper(const Info_t& info, const Shape_t& shape) : ReduceHelperBase(info, shape, true) {}
};

class MatMulHelperBase {
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

class QLinearMatMulHelper : public MatMulHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    QLinearMatMulHelper(const Info_t& info, const Shape_t& shape) : MatMulHelperBase(info, shape, 0, 3) {}
};


class TopKHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  TopKHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion) {
    int32_t k;
    if (opsetVersion >= 10) {
      MLOperatorTensor kTensor = info.GetConstantInputTensor(1);
      k = gsl::narrow_cast<int32_t>(ReadScalarTensorCastToInt64(kTensor));
    } else {
      k = info.GetOptionalAttribute<int32_t>(AttrName::K, -1);
    }
    ML_CHECK_VALID_ARGUMENT(k >= 0, "Attribute k is missing or negative.");
    m_k = k;

    auto inputShape = shape.GetInputTensorShape(0);
    int32_t axis = info.GetOptionalAttribute<int32_t>(AttrName::Axis, -1);
    m_axis = HandleNegativeAxis(axis, gsl::narrow_cast<uint32_t>(inputShape.size()));
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  uint32_t m_k;
  uint32_t m_axis;
};

class RecurrentHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  RecurrentHelper(const Info_t& info, const Shape_t& shape) {
    m_hiddenSize = info.GetOptionalAttribute<int>(AttrName::HiddenSize, 1);
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  int m_hiddenSize = 0;
};

class ConcatHelper {
 public:
  void Initialize(
      const MLOperatorAttributes& operatorAttributes,
      gsl::span<const DimensionType> inputDimensions);

  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  ConcatHelper(const Info_t& info, const Shape_t& shape) {
    Initialize(info, shape.GetInputTensorShape(0));
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  int m_axis;
};

class CropHelper {
 public:
  enum BorderDim { Left,
                   Top,
                   Right,
                   Bottom };
  enum ScaleDim { Height,
                  Width };

  void Initialize(
      const MLOperatorAttributes& operatorAttributes,
      gsl::span<const DimensionType> inputDimensions);

  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  CropHelper(const Info_t& info, const Shape_t& shape) {
    Initialize(info, shape.GetInputTensorShape(0));
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  uint32_t m_offsets[NchwDimensionCount];
  uint32_t m_sizes[NchwSpatialDimensionCount];
};

class DepthToSpaceHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  DepthToSpaceHelper(const Info_t& info, const Shape_t& shape) {
    m_blockSize = info.GetOptionalAttribute<int32_t>(AttrName::BlockSize, -1);
    ML_CHECK_VALID_ARGUMENT(m_blockSize > 0, "Attribute blocksize is missing or equal to zero.");
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  int32_t m_blockSize;
};

class SpaceToDepthHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  SpaceToDepthHelper(const Info_t& info, const Shape_t& shape) {
    m_blockSize = info.GetOptionalAttribute<int32_t>(AttrName::BlockSize, -1);
    ML_CHECK_VALID_ARGUMENT(m_blockSize > 0, "Attribute blocksize is missing or equal to zero.");
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  int32_t m_blockSize;
};

class FlattenHelper {
 public:
  void Initialize(
      const MLOperatorAttributes& operatorAttributes,
      gsl::span<const DimensionType> inputDimensions);

  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  FlattenHelper(const Info_t& info, const Shape_t& shape) {
    Initialize(info, shape.GetInputTensorShape(0));
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  int m_axis = 1;
};

class MultinomialHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  MultinomialHelper(const Info_t& info, const Shape_t& shape) {}

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class GatherHelper {
 public:
  void Initialize(
      const MLOperatorAttributes& operatorAttributes,
      gsl::span<const DimensionType> dataDimensions);

  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  GatherHelper(const Info_t& info, const Shape_t& shape) {
    Initialize(info, shape.GetInputTensorShape(0));
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  int m_axis = 0;
};

class GatherNdHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  GatherNdHelper(const Info_t& info, const Shape_t& shape) {
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class PoolingHelperBase {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  PoolingHelperBase(
      const Info_t& info,
      const Shape_t& shape,
      bool useGlobalPooling) : m_kernel(useGlobalPooling
                                            ? InitializeGlobalKernel(shape.GetInputTensorShape(0))
                                            : InitializeKernel(info, static_cast<uint32_t>(shape.GetInputTensorShape(0).size()), gsl::span<uint32_t>())) {
    if (!useGlobalPooling) {
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
  : m_inputShape(shape.GetInputTensorShape(0)),
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

class GlobalPoolingHelper : public PoolingHelperBase {
 public:
  template <typename Info_t, typename Shape_t>
  GlobalPoolingHelper(const Info_t& info, const Shape_t& shape) : PoolingHelperBase(info, shape, true) {}
};

class PoolingHelper : public PoolingHelperBase {
 public:
  template <typename Info_t, typename Shape_t>
  PoolingHelper(const Info_t& info, const Shape_t& shape) : PoolingHelperBase(info, shape, false) {}
};

class RoiPoolingHelper {
 public:
  enum InputTensors { INPUT,
                      ROIS };

  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  RoiPoolingHelper(const Info_t& info, const Shape_t& shape) {
    std::vector<int> pooledShape = info.GetOptionalAttributeVectorInt32(AttrName::PooledShape);
    ML_CHECK_VALID_ARGUMENT(pooledShape.size() == 2, "Pooled shape must be 2.");
    m_pooledSizeH = pooledShape[0];
    m_pooledSizeW = pooledShape[1];
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  uint32_t m_pooledSizeW;
  uint32_t m_pooledSizeH;
};

class SqueezeHelper {
 public:
  void Initialize(
    gsl::span<const int32_t> axes,
    gsl::span<const DimensionType> inputDimensions);

  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  SqueezeHelper(const Info_t& info, const Shape_t& shape) {
    Initialize(
      info.GetOptionalAttributeVectorInt32(AttrName::Axes),
      shape.GetInputTensorShape(0));
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  std::vector<int> m_axes;
};

class UnsqueezeHelper {
 public:
  void Initialize(
    gsl::span<const int32_t> axes,
    gsl::span<const DimensionType> inputDimensions);

  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  UnsqueezeHelper(const Info_t& info, const Shape_t& shape) {
    Initialize(
      info.GetOptionalAttributeVectorInt32(AttrName::Axes),
      shape.GetInputTensorShape(0));
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  std::vector<int32_t> m_axes;
};

template <typename T>
void CALLBACK ShapeInferenceFunction(IMLOperatorShapeInferenceContext* inference_context) {
  MLShapeInferenceContext helperContext(inference_context);
  T opHelper(helperContext, helperContext);

  // EdgeInfo to contain whether tensor, whether unused, and what shape is
  std::vector<EdgeShapes> outputShapes = opHelper.GetOutputShapes(helperContext);

  for (uint32_t i = 0; i < outputShapes.size(); ++i) {
    if (outputShapes[i].IsTensor() && !outputShapes[i].IsUnused()) {
      helperContext.SetOutputTensorShape(i, outputShapes[i].GetShape());
    }
  }
}

class ReshapeHelper {
 public:
  template <typename Info_t, typename Shape_t>
  ReshapeHelper(const Info_t& info, const Shape_t& shape) {
    ML_CHECK_VALID_ARGUMENT(info.GetInputCount() >= 2);
    ML_CHECK_VALID_ARGUMENT(info.GetOutputCount() >= 1);

    // The 'shape' tensor is a 1D tensor holding the new shape to reshape to,
    // and the first element of its own shape holds how many dimensions there
    // will be for the output.
    MLOperatorTensor shapeTensor = info.GetConstantInputTensor(1);
    ReadCpuLocalTensorIntoInt32(shapeTensor, /*out*/ m_shapeDims);
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  std::vector<int> m_shapeDims;
};

class ExpandHelper {
 public:
  template <typename Info_t, typename Shape_t>
  ExpandHelper(const Info_t& info, const Shape_t& shape) {
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
};

class ConstantOfShapeHelper {
 public:
  template <typename Info_t, typename Shape_t>
  ConstantOfShapeHelper(const Info_t& info, const Shape_t& shape) {
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
};

class TileHelper {
 public:
  template <typename Info_t, typename Shape_t>
  TileHelper(const Info_t& info, const Shape_t& shapeInfo) {
    m_inputDimensions = shapeInfo.GetInputTensorShape(0);

    // Read the repeats tensor.
    const std::vector<uint32_t> repeatsTensorDimensions = shapeInfo.GetInputTensorShape(1);
    ML_CHECK_VALID_ARGUMENT(repeatsTensorDimensions.size() == 1, "Tile's repeats tensor must be 1D.");
    const size_t dimCount = repeatsTensorDimensions[0];

    MLOperatorTensor repeatsTensor = info.GetConstantInputTensor(1);
    const int64_t* repeatsData = repeatsTensor.GetData<int64_t>();
    ML_CHECK_VALID_ARGUMENT(m_inputDimensions.size() == dimCount, "Tile's repeats tensor must be the same dimension count as the input tensor.");
    ML_CHECK_VALID_ARGUMENT(repeatsTensor.IsCpuData(), "Tile's repeats tensor must be CPU Tensor.");

    for (size_t i = 0; i < dimCount; ++i) {
      ML_CHECK_VALID_ARGUMENT(repeatsData[i] > 0, "Repeat values should be > 0.");
      m_repeatsData.push_back(gsl::narrow_cast<uint32_t>(repeatsData[i]));
    }

    // Update the computed output shape accordingly, repeat every axis's length by the repeat count.
    m_outputDimensions.assign(m_inputDimensions.begin(), m_inputDimensions.end());

    for (size_t dimIndex = 0; dimIndex < dimCount; ++dimIndex) {
      m_outputDimensions[dimIndex] *= m_repeatsData[dimIndex];
    }
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  std::vector<uint32_t> m_repeatsData;
  std::vector<uint32_t> m_inputDimensions;
  std::vector<uint32_t> m_outputDimensions;
};

class ResizeHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  ResizeHelper(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion) {

    m_inputDimensions = shape.GetInputTensorShape(0);
    std::vector<int32_t> outputSizes;

    if (opsetVersion >= 11) {
      if (info.IsInputValid(1))
      {
          MLOperatorTensor regionOfInterestTensor = info.GetConstantInputTensor(1);
          ReadCpuLocalTensorIntoFloat32(regionOfInterestTensor, /*out*/ m_regionOfInterest);
      }
      if (info.IsInputValid(2))
      {
          MLOperatorTensor scalesTensor = info.GetConstantInputTensor(2);
          ReadCpuLocalTensorIntoFloat32(scalesTensor, /*out*/ m_scales);
      }
      if (info.IsInputValid(3))
      {
          MLOperatorTensor outputSizesTensor = info.GetConstantInputTensor(3);
          ReadCpuLocalTensorIntoInt32(outputSizesTensor, /*out*/ outputSizes);
      }
    }
    else if (opsetVersion >= 9) {
      // Read the scales from the 2nd tensor.
      // Compatible with Upsample-9/Upsample-10 and Resize-10.
      MLOperatorTensor scalesTensor = info.GetConstantInputTensor(1);
      ReadCpuLocalTensorIntoFloat32(scalesTensor, /*out*/ m_scales);
    } else
    {
      // From attribute, compatible with Upsample-7.
      m_scales = info.GetOptionalAttribute<std::vector<float>>(AttrName::Scales, std::vector<float>());
    }

    Initialize(outputSizes);
  }

  void Initialize(gsl::span<const int32_t> outputSizes);

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  std::vector<DimensionType> m_inputDimensions;
  std::vector<DimensionType> m_outputDimensions;
  std::vector<float> m_scales;
  std::vector<float> m_regionOfInterest; // Stored as [start1, ..., startN, end1, ..., endN], where N is the input rank.
};

class RangeHelper {
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

class OneHotHelper {
 public:
  template <typename Info_t, typename Shape_t>
  OneHotHelper(const Info_t& info, const Shape_t& shapeInfo) {
    ML_CHECK_VALID_ARGUMENT(info.GetInputCount() == 3);
    ML_CHECK_VALID_ARGUMENT(info.GetOutputCount() == 1);

    const std::vector<DimensionType> inputDimensions = shapeInfo.GetInputTensorShape(0);
    std::vector<uint32_t> outputDimensions;

    m_onnxAxis = info.GetOptionalAttribute<int32_t>(AttrName::Axis, -1);

    // Get 'depth' tensor, which is really a scalar for the output size along the given axis.
    MLOperatorTensor shapeTensor = info.GetConstantInputTensor(1);

    auto indicesShape = shapeInfo.GetInputTensorShape(0);
    m_absoluteAxis = HandleNegativeAxis(m_onnxAxis, gsl::narrow_cast<uint32_t>(indicesShape.size() + 1));

    // The shape tensor ('depth') is a 0D tensor holding the size for the output tensor along the specified axis.
    // It must be registered as OrtMemType::OrtMemTypeCPUInput for CPU read access.
    const int64_t depth64 = ReadScalarTensorCastToInt64(shapeTensor);
    ML_CHECK_VALID_ARGUMENT(depth64 > 0, "Negative or zero 'depth' values for OneHot are illegal.");
    const uint32_t depth = gsl::narrow_cast<uint32_t>(depth64);
    m_outputDimensions.assign(indicesShape.begin(), indicesShape.end());
    m_outputDimensions.insert(m_outputDimensions.begin() + m_absoluteAxis, depth);
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  int32_t m_onnxAxis = 0;       // Original ONNX attribute value, including negative value.
  uint32_t m_absoluteAxis = 0;  // Absolute index value.
  std::vector<uint32_t> m_indicesDimensions;
  std::vector<uint32_t> m_outputDimensions;
};

using ShapeInferenceHelper_Conv = ConvHelper;
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
using ShapeInferenceHelper_InstanceNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_BatchNormalization = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_LRN = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_MeanVarianceNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_LpNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_RNN = RecurrentHelper;
using ShapeInferenceHelper_GRU = RecurrentHelper;
using ShapeInferenceHelper_LSTM = RecurrentHelper;

using ShapeInferenceHelper_Gather = GatherHelper;
using ShapeInferenceHelper_GatherElements = GetOutputShapeAsSpecificInputShapeHelper<1>;
using ShapeInferenceHelper_ScatterElements = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Scatter9 = ShapeInferenceHelper_ScatterElements; // Old deprecated alias for ScatterElements.
using ShapeInferenceHelper_Scatter11 = ShapeInferenceHelper_ScatterElements; // Old deprecated alias for ScatterElements.
using ShapeInferenceHelper_GatherND = GatherNdHelper;
using ShapeInferenceHelper_ScatterND = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_Flatten = FlattenHelper;
using ShapeInferenceHelper_Split = SplitHelper;
using ShapeInferenceHelper_Transpose = TransposeHelper;
using ShapeInferenceHelper_Concat = ConcatHelper;
using ShapeInferenceHelper_Slice7 = VersionedOpsetHelper<SliceHelper, 7>;
using ShapeInferenceHelper_Slice10 = VersionedOpsetHelper<SliceHelper, 10>;
using ShapeInferenceHelper_Slice11 = VersionedOpsetHelper<SliceHelper, 11>; // Note 11 and 10 are identical - no functional change.
using ShapeInferenceHelper_Pad7 = VersionedOpsetHelper<PaddingHelper, 7>;
using ShapeInferenceHelper_Pad11 = VersionedOpsetHelper<PaddingHelper, 11>;

using ShapeInferenceHelper_SpaceToDepth = SpaceToDepthHelper;
using ShapeInferenceHelper_DepthToSpace = DepthToSpaceHelper;
using ShapeInferenceHelper_Squeeze = SqueezeHelper;
using ShapeInferenceHelper_Unsqueeze = UnsqueezeHelper;
using ShapeInferenceHelper_EyeLike = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_Expand = ExpandHelper;
using ShapeInferenceHelper_Reshape = ReshapeHelper;
using ShapeInferenceHelper_ConstantOfShape = ConstantOfShapeHelper;
using ShapeInferenceHelper_Tile = TileHelper;
using ShapeInferenceHelper_Resize10 = VersionedOpsetHelper<ResizeHelper, 10>;
using ShapeInferenceHelper_Resize11 = VersionedOpsetHelper<ResizeHelper, 11>;
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
using ShapeInferenceHelper_Greater = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Less = GetBroadcastedOutputShapeHelper;
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
using ShapeInferenceHelper_Sign = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_IsNan = GetBroadcastedOutputShapeHelper;
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
using ShapeInferenceHelper_Selu = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Softmax = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_LogSoftmax = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Hardmax = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Softsign = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Softplus = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_ParametricSoftplus = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Dropout = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Shrink = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_Identity = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_MatMul = MatMulHelper;
using ShapeInferenceHelper_MatMulInteger = MatMulHelper;
using ShapeInferenceHelper_QLinearMatMul = QLinearMatMulHelper;

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
using ShapeInferenceHelper_CumSum = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Range = RangeHelper;

using ShapeInferenceHelper_FusedConv = ConvHelper;
using ShapeInferenceHelper_FusedConvTranspose = ConvTransposeHelper;
using ShapeInferenceHelper_FusedInstanceNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_FusedBatchNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_FusedMeanVarianceNormalization = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_FusedGemm = GemmHelper;
using ShapeInferenceHelper_FusedMatMul = MatMulHelper;
using ShapeInferenceHelper_FusedAdd = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_FusedSum = GetBroadcastedOutputShapeHelper;

}  // namespace OperatorHelper

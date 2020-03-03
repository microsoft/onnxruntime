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

int64_t ReadAsInt64(MLOperatorTensorDataType tensorDataType, const void* p);

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
  uint32_t windowSize[NcdhwSpatialDimensionCount];
  uint32_t startPadding[NcdhwSpatialDimensionCount];
  uint32_t endPadding[NcdhwSpatialDimensionCount];
  uint32_t outputPadding[NcdhwSpatialDimensionCount];

  KernelArgs(uint32_t spatialDimensionCount) : autoPad(false),
                                               autoPadSameUpper(false),
                                               spatialDimensionCount(spatialDimensionCount) {
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

  // This is true if padding must be automatically computed based on input sizes.
  // ResolveAutoPadding must happen during Compute rather than initialization.
  // This is temporary until kernel initialization routine once Lotus can provide
  // sizes at operator initialization.
  bool autoPad;
  bool autoPadSameUpper;
  uint32_t spatialDimensionCount;
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

class GetOutputShapeAsInputShapeHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  GetOutputShapeAsInputShapeHelper(const Info_t& info, const Shape_t& shape){
    ORT_UNUSED_PARAMETER(info);
    ORT_UNUSED_PARAMETER(shape);
  };

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
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
    enum InputTensor { X, Filter};
    enum InputDims { N, C, H, W };

public:
    // Info_t is used to obtain attributes which will be used for calculating the output shape later. 
    template<typename Info_t, typename Shape_t>
    ConvolutionHelperBase(const Info_t& info, const Shape_t& shape, bool transpose, bool hasDynamicPads) :
        m_kernel(InitializeKernel(info, shape.GetInputTensorDimensionCount(0), shape.GetInputTensorShape(1)))
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
    const std::vector<DimensionType> inputDimensions = shapeInfo.GetInputTensorShape(0);
    const std::vector<DimensionType> filterDims = shapeInfo.GetInputTensorShape(1);

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

    const std::vector<DimensionType> inputDimensions = shapeInfo.GetInputTensorShape(0);
    const std::vector<DimensionType> filterDims = shapeInfo.GetInputTensorShape(1);

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
  KernelArgs m_kernel;
  std::vector<EdgeShapes> m_outputShapes;
};

class ConvHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    ConvHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, false, false) {}
};

class ConvTransposeHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    ConvTransposeHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, true, false) {}
};

class ConvTransposeWithDynamicPadsHelper : public ConvolutionHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    ConvTransposeWithDynamicPadsHelper(const Info_t& info, const Shape_t& shape) : ConvolutionHelperBase(info, shape, true, true) {}
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

class SliceHelperBase
{
public:
    template<typename Info_t, typename Index_t>
    void ReadIndexTensors(
        const Info_t& operatorInfo,
        std::vector<int32_t>& starts,
        std::vector<int32_t>& ends,
        std::vector<int32_t>& axes,
        std::vector<int32_t>& steps
    )
    {
        // Get starts, ends, optional axes and optional steps from constant inputs.
        MLOperatorTensor startsTensor = operatorInfo.GetConstantInputTensor(1);
        const std::vector<uint32_t>& startsTensorDimensions = startsTensor.GetShape();
        size_t dimCount = startsTensorDimensions[0];
        const Index_t* startsData = startsTensor.GetData<Index_t>();
        for (size_t i = 0; i < dimCount; ++i)
        {
            starts.push_back(gsl::narrow_cast<int32_t>(startsData[i]));
        }

        MLOperatorTensor endsTensor = operatorInfo.GetConstantInputTensor(2);
        const std::vector<uint32_t>& endsTensorDimensions = endsTensor.GetShape();
        dimCount = endsTensorDimensions[0];
        const Index_t* endsData = endsTensor.GetData<Index_t>();
        for (size_t i = 0; i < dimCount; ++i)
        {
            ends.push_back(gsl::narrow_cast<int32_t>(endsData[i]));
        }
        uint32_t inputCount = operatorInfo.GetInputCount();
        if (inputCount > 3)
        {
            MLOperatorTensor axesTensor = operatorInfo.GetConstantInputTensor(3);
            const std::vector<uint32_t>& axesTensorDimensions = axesTensor.GetShape();
            dimCount = axesTensorDimensions[0];
            const Index_t* axesData = axesTensor.GetData<Index_t>();
            for (size_t i = 0; i < dimCount; ++i)
            {
                axes.push_back(gsl::narrow_cast<int32_t>(axesData[i]));
            }
        }

        if (inputCount > 4)
        {
            MLOperatorTensor stepsTensor = operatorInfo.GetConstantInputTensor(4);
            const std::vector<uint32_t>& stepsTensorDimensions = stepsTensor.GetShape();
            dimCount = stepsTensorDimensions[0];
            const Index_t* stepsData = stepsTensor.GetData<Index_t>();
            for (size_t i = 0; i < dimCount; ++i)
            {
                steps.push_back(gsl::narrow_cast<int32_t>(stepsData[i]));
            }
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
            // Get starts, ends and axes from attributes
            starts = operatorInfo.GetOptionalAttributeVectorInt32(AttrName::Starts);
            ends = operatorInfo.GetOptionalAttributeVectorInt32(AttrName::Ends);
            axes = operatorInfo.GetOptionalAttributeVectorInt32(AttrName::Axes);
        }
        else if (opsetVersion == 10)
        {
            if (operatorInfo.GetConstantInputTensor(1).GetTensorDataType() == MLOperatorTensorDataType::Int32)
            {
                ReadIndexTensors<Info_t, int32_t>(operatorInfo, starts, ends, axes, steps);
            }
            else
            {
                THROW_HR_IF(E_INVALIDARG, operatorInfo.GetConstantInputTensor(1).GetTensorDataType() != MLOperatorTensorDataType::Int64);
                ReadIndexTensors<Info_t, int64_t>(operatorInfo, starts, ends, axes, steps);
            }
        }
        
        const uint32_t dimCount = gsl::narrow_cast<int32_t>(inputDimensions.size());
        HandleNegativeAxes(/*inout*/ axes, dimCount); 
         
        ML_CHECK_VALID_ARGUMENT(starts.size() == ends.size(), "'starts' must equal 'ends' in size.");
        ML_CHECK_VALID_ARGUMENT(axes.empty() || starts.size() == axes.size(), "'axes' must equal 'starts' in size, or 'axes' must be empty.");

        m_outputDimensions.assign(inputDimensions.begin(), inputDimensions.end());
        m_offsets.resize(m_outputDimensions.size());
        m_sizes.resize(m_outputDimensions.size());
        m_strides.resize(m_outputDimensions.size(), 1); // Only a stride of 1 element is supported by ONNX 1.2.

        // Set initial defaults lest 'starts' and 'ends' arrays are shorter than the dimension count.
        std::copy(inputDimensions.begin(), inputDimensions.begin() + m_outputDimensions.size(), m_sizes.begin());

        // Clamp selected dimensions to given 'starts' and 'ends'.
        for (int i = 0, ci = gsl::narrow_cast<int>(starts.size()); i < ci; ++i)
        {
            int dimIndex = i;
            if (!axes.empty())
            {
                dimIndex = axes[i];
            }
            ML_CHECK_VALID_ARGUMENT(dimIndex < static_cast<int>(inputDimensions.size()), "'axes' must be valid with within actual input dimensions.");

            // Positive values are offsets from 0.
            // Negative values are offsets from the dimension's size.
            int dim = gsl::narrow_cast<int>(inputDimensions[dimIndex]);
            int start = (starts[i] < 0) ? (starts[i] + dim) : starts[i];
            int end = (ends[i] < 0) ? (ends[i] + dim) : ends[i];

            // Clamp the dimensions to the slice extents.
            // Clamp negative numbers to 0, per case test_slice_start_out_of_bounds.
            start = std::max(start, 0);
            end = std::min(end, dim);
            int size = std::max(end - start, 0);

            m_outputDimensions[dimIndex] = size;
            m_offsets[dimIndex] = start;
            m_sizes[dimIndex] = gsl::narrow_cast<uint32_t>(size);
        }
    }

    // Info_t is used to obtain attributes which will be used for calculating the output shape later. 
    // Shape_t is used to obtain input shape which will be used for adjusting attribute value. 
    template<typename Info_t, typename Shape_t>
    SliceHelperBase(const Info_t& info, const Shape_t& shape, uint32_t opsetVersion)
    {
        Initialize(info, shape.GetInputTensorShape(0), opsetVersion);
    }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  std::vector<DimensionType> m_outputDimensions;
  std::vector<uint32_t> m_offsets;
  std::vector<uint32_t> m_sizes;
  std::vector<uint32_t> m_strides;
};

class SliceHelper : public SliceHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    SliceHelper(const Info_t& info, const Shape_t& shape) : SliceHelperBase(info, shape, 7) {}
};

class Slice10Helper : public SliceHelperBase
{
public:
    template<typename Info_t, typename Shape_t>
    Slice10Helper(const Info_t& info, const Shape_t& shape) : SliceHelperBase(info, shape, 10) {}
};


class PaddingHelper
{
public:
    void Initialize(const MLOperatorAttributes& operatorAttributes);

  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  PaddingHelper(const Info_t& info, const Shape_t& shape) {
    Initialize(info);
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  std::vector<uint32_t> m_startPadding;
  std::vector<uint32_t> m_endPadding;
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

class MatMulHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  MatMulHelper(const Info_t& info, const Shape_t& shape) {}

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;
};

class TopKHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  TopKHelper(const Info_t& info, const Shape_t& shape) {
    m_k = info.GetOptionalAttribute<int32_t>(AttrName::K, -1);
    ML_CHECK_VALID_ARGUMENT(m_k >= 0, "Attribute k is missing.");

    m_axis = info.GetOptionalAttribute<int32_t>(AttrName::Axis, -1);
    auto inputShape = shape.GetInputTensorShape(0);

    if (m_axis < 0) {
      m_axis = m_axis + gsl::narrow_cast<uint32_t>(inputShape.size());
    }
    ML_CHECK_VALID_ARGUMENT(m_axis >= 0 && m_axis < gsl::narrow_cast<int32_t>(inputShape.size()));
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  int32_t m_k;
  int32_t m_axis;
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
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  SqueezeHelper(const Info_t& info, const Shape_t& shape) {
    m_axes = info.GetOptionalAttributeVectorInt32(AttrName::Axes);
    std::sort(m_axes.begin(), m_axes.end());
  }

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  std::vector<int> m_axes;
};

class UnsqueezeHelper {
 public:
  // Info_t is used to obtain attributes which will be used for calculating the output shape later.
  // Shape_t is used to obtain input shape which will be used for adjusting attribute value.
  template <typename Info_t, typename Shape_t>
  UnsqueezeHelper(const Info_t& info, const Shape_t& shape) {
    m_axes = info.GetOptionalAttributeVectorInt32(AttrName::Axes);
    std::sort(m_axes.begin(), m_axes.end());
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

    MLOperatorTensor shapeTensor = info.GetConstantInputTensor(1);

    // The 'shape' tensor is a 1D tensor holding the new shape to reshape to,
    // and the first element of its own shape holds how many dimensions there
    // will be for the output.
    std::vector<uint32_t> shapeTensorDimensions = shapeTensor.GetShape();
    ML_CHECK_VALID_ARGUMENT(shapeTensorDimensions.size() == 1, "Reshape's shape tensor must be 1D.");
    size_t dimCount = shapeTensorDimensions[0];

    ML_CHECK_VALID_ARGUMENT(shapeTensor.IsCpuData(), "Reshape's shape tensor must be CPU Tensor.");
    const int64_t* shapeData = shapeTensor.GetData<int64_t>();

    // Shape of shape tensor is how many dims to reshape to.
    for (size_t i = 0; i < dimCount; ++i) {
      m_shapeDims.push_back(gsl::narrow_cast<int>(shapeData[i]));
    }
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
  ResizeHelper(const Info_t& info, const Shape_t& shape) {
    // Read the scales from the 2nd tensor.
    if (info.GetInputCount() > 1) {
      MLOperatorTensor scalesTensor = info.GetConstantInputTensor(1);
      Initialize(scalesTensor, shape.GetInputTensorShape(0));
    } else  // From attribute.
    {
      Initialize(info, shape.GetInputTensorShape(0));
    }
  }

  void Initialize(
      const MLOperatorAttributes& operatorAttributes,
      gsl::span<const DimensionType> inputDimensions);

  void Initialize(
      const MLOperatorTensor& scalesTensor,
      gsl::span<const DimensionType> inputDimensions);

  void InitializeOutputDimensions(
      gsl::span<const float> scales,
      gsl::span<const DimensionType> inputDimensions);

  std::vector<EdgeShapes> GetOutputShapes(const MLShapeInferenceContext& shapeInfo) const;

 protected:
  std::vector<DimensionType> m_inputDimensions;
  std::vector<DimensionType> m_outputDimensions;
  std::vector<float> m_scales;  // Cached scales to check for updates/invalidate operator.
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
    const uint32_t depthElementCount = ComputeElementCountFromDimensions(shapeTensor.GetShape());
    ML_CHECK_VALID_ARGUMENT(shapeTensor.IsCpuData(), "OneHots's 'depth' tensor must be a CPU Tensor.");
    ML_CHECK_VALID_ARGUMENT(depthElementCount == 1, "OneHots's 'depth' tensor must have one element.");
    const void* tensorData = shapeTensor.GetByteData();
    const int64_t depth64 = ReadAsInt64(shapeTensor.GetTensorDataType(), tensorData);
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
using ShapeInferenceHelper_AveragePool = PoolingHelper;
using ShapeInferenceHelper_GlobalAveragePool = GlobalPoolingHelper;
using ShapeInferenceHelper_MaxPool = PoolingHelper;
using ShapeInferenceHelper_GlobalMaxPool = GlobalPoolingHelper;
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

using ShapeInferenceHelper_Flatten = FlattenHelper;
using ShapeInferenceHelper_Split = SplitHelper;
using ShapeInferenceHelper_Transpose = TransposeHelper;
using ShapeInferenceHelper_Concat = ConcatHelper;
using ShapeInferenceHelper_Slice7 = SliceHelper;
using ShapeInferenceHelper_Slice10 = Slice10Helper;
using ShapeInferenceHelper_Pad = PaddingHelper;
using ShapeInferenceHelper_SpaceToDepth = SpaceToDepthHelper;
using ShapeInferenceHelper_DepthToSpace = DepthToSpaceHelper;
using ShapeInferenceHelper_Squeeze = SqueezeHelper;
using ShapeInferenceHelper_Unsqueeze = UnsqueezeHelper;
using ShapeInferenceHelper_EyeLike = GetOutputShapeAsInputShapeHelper;

using ShapeInferenceHelper_Expand = ExpandHelper;
using ShapeInferenceHelper_Reshape = ReshapeHelper;
using ShapeInferenceHelper_ConstantOfShape = ConstantOfShapeHelper;
using ShapeInferenceHelper_Tile = TileHelper;
using ShapeInferenceHelper_Resize = ResizeHelper;
using ShapeInferenceHelper_OneHot = OneHotHelper;

using ShapeInferenceHelper_Sqrt = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Reciprocal = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Pow = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Exp = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Log = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Abs = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Ceil = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Floor = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Clip = GetOutputShapeAsInputShapeHelper;
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
using ShapeInferenceHelper_Scatter = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Sign = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_IsNan = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Erf = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Sinh = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Cosh = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Asinh = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Acosh = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Atanh = GetBroadcastedOutputShapeHelper;
using ShapeInferenceHelper_Where = GetBroadcastedOutputShapeHelper;

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
using ShapeInferenceHelper_Upsample = ResizeHelper;

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
using ShapeInferenceHelper_Cast = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_MemcpyFromHost = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_MemcpyToHost = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_TopK = TopKHelper;

using ShapeInferenceHelper_RandomUniform = RandomUniformHelper;
using ShapeInferenceHelper_RandomUniformLike = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_RandomNormal = RandomNormalHelper;
using ShapeInferenceHelper_RandomNormalLike = GetOutputShapeAsInputShapeHelper;
using ShapeInferenceHelper_Multinomial = MultinomialHelper;

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

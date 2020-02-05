// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ImageFeatureDescriptor.h"
#include "ImageFeatureValue.h"
#include "IMapFeatureValue.h"
#include "ISequenceFeatureValue.h"
#include "TensorFeatureDescriptor.h"

namespace Windows::AI::MachineLearning {

namespace error_strings {
using namespace winrt::Windows::AI::MachineLearning;

// This must be kept in sync with the TensorKind enum in Windows.AI.MachineLearning.idl
const char* SzTensorKind[] =
    {
        "Undefined",
        "Float",
        "UInt8",
        "Int8",
        "UInt16",
        "Int16",
        "Int32",
        "Int64",
        "String",
        "Boolean",
        "Float16",
        "Double",
        "UInt32",
        "UInt64",
        "Complex64",
        "Complex128",
};

static std::string ToString(ILearningModelFeatureDescriptor descriptor);

static std::string ToString(const std::vector<int64_t>& shape) {
  std::ostringstream stream;
  stream << "[";
  std::copy(shape.begin(), shape.end(), std::ostream_iterator<int64_t>(stream, ","));
  stream << "]";

  return stream.str();
}

static std::string ToString(winrt::Windows::Foundation::Collections::IVectorView<int64_t> shape) {
  auto shapeVec = std::vector<int64_t>(begin(shape), end(shape));
  return ToString(shapeVec);
}

static std::string ToString(
    TensorKind kind,
    winrt::Windows::Foundation::Collections::IVectorView<int64_t> shape) {
  FAIL_FAST_IF_MSG(kind == TensorKind::Complex128, "Unexpected TensorKind Complex128.");
  FAIL_FAST_IF_MSG(kind == TensorKind::Complex64, "Unexpected TensorKind Complex64.");
  FAIL_FAST_IF_MSG(kind == TensorKind::Undefined, "Unexpected TensorKind Undefined.");

  std::ostringstream stream;
  stream << SzTensorKind[static_cast<uint32_t>(kind)] << ToString(shape);
  return stream.str();
}

static std::string ToString(ITensorFeatureDescriptor descriptor) {
  return ToString(descriptor.TensorKind(), descriptor.Shape());
}

static std::string ToString(ITensor value) {
  return ToString(value.TensorKind(), value.Shape());
}

static std::string ToString(IMapFeatureDescriptor descriptor) {
  auto keyKind = descriptor.KeyKind();
  FAIL_FAST_IF_MSG(keyKind == TensorKind::Complex128, "Unexpected TensorKind Complex128.");
  FAIL_FAST_IF_MSG(keyKind == TensorKind::Complex64, "Unexpected TensorKind Complex64.");
  FAIL_FAST_IF_MSG(keyKind == TensorKind::Undefined, "Unexpected TensorKind Undefined.");

  auto valueDescriptor = descriptor.ValueDescriptor();
  std::ostringstream stream;
  stream << "Map<" << SzTensorKind[static_cast<uint32_t>(keyKind)] << "," << ToString(valueDescriptor) << ">";
  return stream.str();
}

static std::string ToString(winrt::com_ptr<IMapFeatureValue> value) {
  TensorKind keyKind;
  FAIL_FAST_IF_FAILED(value->get_KeyKind(&keyKind));
  FAIL_FAST_IF_MSG(keyKind == TensorKind::Complex128, "Unexpected TensorKind Complex128.");
  FAIL_FAST_IF_MSG(keyKind == TensorKind::Complex64, "Unexpected TensorKind Complex64.");
  FAIL_FAST_IF_MSG(keyKind == TensorKind::Undefined, "Unexpected TensorKind Undefined.");

  ILearningModelFeatureDescriptor valueDescriptor;
  FAIL_FAST_IF_FAILED(value->get_ValueDescriptor(&valueDescriptor));
  std::ostringstream stream;
  stream << "Map<" << SzTensorKind[static_cast<uint32_t>(keyKind)] << "," << ToString(valueDescriptor) << ">";
  return stream.str();
}

static std::string ToString(ISequenceFeatureDescriptor descriptor) {
  auto elementDescriptor = descriptor.ElementDescriptor();
  std::ostringstream stream;
  stream << "Sequence<" << ToString(elementDescriptor) << ">";
  return stream.str();
}

static std::string ToString(winrt::com_ptr<ISequenceFeatureValue> value) {
  ILearningModelFeatureDescriptor elementDescriptor;
  FAIL_FAST_IF_FAILED(value->get_ElementDescriptor(&elementDescriptor));

  std::ostringstream stream;
  stream << "Sequence<" << ToString(elementDescriptor) << ">";
  return stream.str().c_str();
}

static std::string ToString(IImageFeatureDescriptor descriptor) {
  std::ostringstream stream;
  stream << "Image[" << descriptor.Width() << "x" << descriptor.Height() << "]";
  return stream.str();
}

static std::string ToString(winrt::com_ptr<implementation::ImageFeatureValue> value) {
  std::ostringstream stream;
  stream << "Image[" << value->Widths()[0] << "x" << value->Heights()[0] << "]";
  return stream.str();
}

static std::string ToString(ILearningModelFeatureDescriptor descriptor) {
  switch (descriptor.Kind()) {
    case LearningModelFeatureKind::Image:
      return ToString(descriptor.as<IImageFeatureDescriptor>());
      break;
    case LearningModelFeatureKind::Map:
      return ToString(descriptor.as<IMapFeatureDescriptor>());
      break;
    case LearningModelFeatureKind::Sequence:
      return ToString(descriptor.as<ISequenceFeatureDescriptor>());
      break;
    case LearningModelFeatureKind::Tensor:
      return ToString(descriptor.as<ITensorFeatureDescriptor>());
    default:
      FAIL_FAST_MSG("Unexpected descriptor LearningModelFeatureKind.");
  }
}

static std::string ToString(ILearningModelFeatureValue value) {
  switch (value.Kind()) {
    case LearningModelFeatureKind::Image:
      return ToString(value.as<implementation::ImageFeatureValue>());
      break;
    case LearningModelFeatureKind::Map:
      return ToString(value.as<IMapFeatureValue>());
      break;
    case LearningModelFeatureKind::Sequence:
      return ToString(value.as<ISequenceFeatureValue>());
      break;
    case LearningModelFeatureKind::Tensor:
      return ToString(value.as<ITensor>());
    default:
      FAIL_FAST_MSG("Unexpected descriptor LearningModelFeatureKind.");
  }
}
}  // namespace error_strings

// This file produces the IsFeatureValueCompatibleWithDescriptor helper method.
// It is used in the Bind() call to determine whether a feature value aggrees
// with the input or output descriptor present on the model.
//
// These checks are accomplished by indexing into the FeatureKindCompatibilityMatrix.
// This matrix is indexed by Kind, and is a group of function pointers which accept
// a feature value and descriptr, and return whether they are compatible.
namespace compatibility_details {
using namespace winrt::Windows::AI::MachineLearning;

using K = LearningModelFeatureKind;

static void not_compatible_hr(HRESULT hr, ILearningModelFeatureValue value, ILearningModelFeatureDescriptor descriptor) {
  auto name = WinML::Strings::UTF8FromHString(descriptor.Name());

  WINML_THROW_IF_FAILED_MSG(
      hr,
      "Model variable %s, expects %s, but binding was attempted with an incompatible type %s.",
      name.c_str(),
      error_strings::ToString(descriptor).c_str(),
      error_strings::ToString(value).c_str());
}

static void not_compatible(ILearningModelFeatureValue value, ILearningModelFeatureDescriptor descriptor) {
  not_compatible_hr(WINML_ERR_INVALID_BINDING, value, descriptor);
}

static HRESULT verify(ILearningModelFeatureDescriptor first, ILearningModelFeatureDescriptor second) {
  RETURN_HR_IF(WINML_ERR_INVALID_BINDING, first.Kind() != second.Kind());

  if (auto mapFirst = first.try_as<MapFeatureDescriptor>()) {
    auto mapSecond = second.try_as<MapFeatureDescriptor>();
    RETURN_HR_IF_NULL(WINML_ERR_INVALID_BINDING, mapSecond);
    RETURN_HR_IF(WINML_ERR_INVALID_BINDING, mapFirst.KeyKind() != mapSecond.KeyKind());
    return verify(mapFirst.ValueDescriptor(), mapSecond.ValueDescriptor());
  }

  if (auto sequenceFirst = first.try_as<SequenceFeatureDescriptor>()) {
    auto sequenceSecond = second.try_as<SequenceFeatureDescriptor>();
    RETURN_HR_IF_NULL(WINML_ERR_INVALID_BINDING, sequenceSecond);
    return verify(sequenceFirst.ElementDescriptor(), sequenceSecond.ElementDescriptor());
  }

  if (auto tensorFirst = first.try_as<TensorFeatureDescriptor>()) {
    auto tensorSecond = second.try_as<TensorFeatureDescriptor>();
    RETURN_HR_IF_NULL(WINML_ERR_INVALID_BINDING, tensorSecond);
    RETURN_HR_IF(WINML_ERR_INVALID_BINDING, tensorFirst.TensorKind() != tensorSecond.TensorKind());

    // since we only really support scalars inside maps and sequences,
    // make sure that each dimension is either -1 or 1.
    // Note that they don't have be the same since they're still compatible.
    for (auto&& dim : tensorFirst.Shape()) {
      RETURN_HR_IF(WINML_ERR_INVALID_BINDING, (dim != -1 && dim != 1));
    }
    for (auto&& dim : tensorSecond.Shape()) {
      RETURN_HR_IF(WINML_ERR_INVALID_BINDING, (dim != -1 && dim != 1));
    }
    return S_OK;
  }

  return WINML_ERR_INVALID_BINDING;
}

/*
        Checks if FeatureValue matches the feature description of a model
        TValue: feature value from binding
        TFeatureDescriptor: feature description from model
    */
template <K TValue, K TFeatureDescriptor>
void verify(ILearningModelFeatureValue value, ILearningModelFeatureDescriptor descriptor) {
  not_compatible(value, descriptor);
}

template <>
void verify<K::Tensor, K::Tensor>(
    ILearningModelFeatureValue value,
    ILearningModelFeatureDescriptor descriptor) {
  thrower fail = std::bind(not_compatible_hr, std::placeholders::_1, value, descriptor);
  enforce check = std::bind(enforce_not_false, std::placeholders::_1, std::placeholders::_2, fail);

  auto tensorValue = value.as<ITensor>();
  auto tensorDescriptor = descriptor.as<ITensorFeatureDescriptor>();
  check(WINML_ERR_INVALID_BINDING, tensorValue.TensorKind() == tensorDescriptor.TensorKind());

  auto spValueProvider = tensorValue.as<ILotusValueProviderPrivate>();

  bool isPlaceHolder;
  if (SUCCEEDED(spValueProvider->IsPlaceholder(&isPlaceHolder)) && !isPlaceHolder) {
    // Placeholders dont have shapes set, so do the shape check for non-Placeholders
    auto tensorValueShape = tensorValue.Shape();
    auto tensorDescriptorShape = tensorDescriptor.Shape();
    check(WINML_ERR_SIZE_MISMATCH, tensorValueShape.Size() == tensorDescriptorShape.Size());

    for (unsigned i = 0; i < tensorValueShape.Size(); i++) {
      if (tensorDescriptorShape.GetAt(i) == -1) {
        // For free dimensions, the dimension will be set to -1.
        // In that case skip validation.
        continue;
      }
      check(WINML_ERR_SIZE_MISMATCH, tensorValueShape.GetAt(i) == tensorDescriptorShape.GetAt(i));
    }
  }
}

template <>
void verify<K::Map, K::Map>(
    ILearningModelFeatureValue value,
    ILearningModelFeatureDescriptor descriptor) {
  thrower fail = std::bind(not_compatible_hr, std::placeholders::_1, value, descriptor);
  enforce check = std::bind(enforce_not_false, std::placeholders::_1, std::placeholders::_2, fail);
  enforce_succeeded check_succeeded = std::bind(enforce_not_failed, std::placeholders::_1, fail);

  auto spMapFeatureValue = value.as<WinML::IMapFeatureValue>();
  auto mapDescriptor = descriptor.as<IMapFeatureDescriptor>();

  TensorKind valueKeyKind;
  check_succeeded(spMapFeatureValue->get_KeyKind(&valueKeyKind));
  check(WINML_ERR_INVALID_BINDING, valueKeyKind == mapDescriptor.KeyKind());

  ILearningModelFeatureDescriptor valueValueDescriptor;
  check_succeeded(spMapFeatureValue->get_ValueDescriptor(&valueValueDescriptor));

  check_succeeded(verify(valueValueDescriptor, mapDescriptor.ValueDescriptor()));
}

template <>
void verify<K::Sequence, K::Sequence>(
    ILearningModelFeatureValue value,
    ILearningModelFeatureDescriptor descriptor) {
  thrower fail = std::bind(not_compatible_hr, std::placeholders::_1, value, descriptor);
  enforce_succeeded check_succeeded = std::bind(enforce_not_failed, std::placeholders::_1, fail);

  auto spSequenceFeatureValue = value.as<WinML::ISequenceFeatureValue>();
  auto sequenceDescriptor = descriptor.as<ISequenceFeatureDescriptor>();

  ILearningModelFeatureDescriptor valueElementDescriptor;
  check_succeeded(spSequenceFeatureValue->get_ElementDescriptor(&valueElementDescriptor));

  check_succeeded(verify(valueElementDescriptor, sequenceDescriptor.ElementDescriptor()));
}

template <>
void verify<K::Image, K::Image>(
    ILearningModelFeatureValue value,
    ILearningModelFeatureDescriptor descriptor) {
  // No check is needed here. Because:
  // For batchSize==1, no matter what shape the input has (smaller or larger), we support to bind it.
  // For batchSize > 1,
  // 1. for non-free dimension, we support to bind a batch of inputs with different shapes
  //    because we would reshape the inputs to same size as descriptor specified.
  // 2. for free dimension, we have check in ImageFeatureValue that all inputs must have the same shape.
  //    And the check will be triggered at GetOrtValue step before binding.
  return;
}

template <>
void verify<K::Tensor, K::Image>(
    ILearningModelFeatureValue value,
    ILearningModelFeatureDescriptor descriptor) {
  thrower fail = std::bind(not_compatible_hr, std::placeholders::_1, value, descriptor);
  enforce check = std::bind(enforce_not_false, std::placeholders::_1, std::placeholders::_2, fail);
  enforce_succeeded check_succeeded = std::bind(enforce_not_failed, std::placeholders::_1, fail);

  auto tensorValue = value.as<ITensor>();
  auto imageDescriptor = descriptor.as<implementation::ImageFeatureDescriptor>();

  check(WINML_ERR_INVALID_BINDING, tensorValue.TensorKind() == TensorKind::Float);

  auto spValueProvider = tensorValue.as<ILotusValueProviderPrivate>();

  bool isPlaceHolder;
  if (SUCCEEDED(spValueProvider->IsPlaceholder(&isPlaceHolder)) && !isPlaceHolder) {
    auto tensorValueShape = tensorValue.Shape();
    auto imageDescriptorShape = imageDescriptor->Shape();

    check(WINML_ERR_SIZE_MISMATCH, tensorValueShape.Size() == imageDescriptorShape.Size());

    for (unsigned i = 0; i < tensorValueShape.Size(); i++) {
      // Free dimensions on images are indicated by setting the shape size -1/MAXUINT
      // In that case, ignore the tensor size check
      if (imageDescriptorShape.GetAt(i) != -1) {
        check(WINML_ERR_SIZE_MISMATCH, tensorValueShape.GetAt(i) == imageDescriptorShape.GetAt(i));
      }
    }
  }
}

/*
        This is the case when a model expects a tensor, but image is passed in for binding.
        There are two main scenarios for this:
        1. Image metadata does not exist: We should be tolerant to the models that does not have Image Metadata.
            In this case, user can still pass in ImageFeatureValue as long as it meets the requirement for image tensorization
        2. Model may have Image metadata that values that we do not support. In this case we should reject binding ImageFeatureValue
           https://github.com/onnx/onnx/blob/master/docs/MetadataProps.md
           Supported metadata values in RS5
               - Image.BitmapPixelFormat: Gray8, RGB8, BGR8
               - Image.ColorSpaceGamma: SRGB
               - Image.NominalPixelRagne: NominalRange_0_255
    */
template <>
void verify<K::Image, K::Tensor>(
    ILearningModelFeatureValue value,
    ILearningModelFeatureDescriptor descriptor) {
  thrower fail = std::bind(not_compatible_hr, std::placeholders::_1, value, descriptor);
  enforce check = std::bind(enforce_not_false, std::placeholders::_1, std::placeholders::_2, fail);

  auto imageValue = value.as<implementation::ImageFeatureValue>();
  auto tensorDescriptor = descriptor.as<implementation::TensorFeatureDescriptor>();

  check(WINML_ERR_INVALID_BINDING, !tensorDescriptor->IsUnsupportedMetaData());
  // NCHW: images must be 4 dimensions
  auto tensorDescriptorShape = tensorDescriptor->Shape();
  check(WINML_ERR_SIZE_MISMATCH, 4 == tensorDescriptorShape.Size());
}

static void (*FeatureKindCompatibilityMatrix[4][4])(ILearningModelFeatureValue, ILearningModelFeatureDescriptor) =
    {
        //                 Tensor,                          Sequence,                           Map,                    Image
        /* Tensor */ {verify<K::Tensor, K::Tensor>, not_compatible, not_compatible, verify<K::Tensor, K::Image>},
        /* Sequence */ {not_compatible, verify<K::Sequence, K::Sequence>, not_compatible, not_compatible},
        /* Map */ {not_compatible, not_compatible, verify<K::Map, K::Map>, not_compatible},
        /* Image */ {verify<K::Image, K::Tensor>, not_compatible, not_compatible, verify<K::Image, K::Image>}};
}  // namespace compatibility_details

inline void VerifyFeatureValueCompatibleWithDescriptor(
    winrt::Windows::AI::MachineLearning::ILearningModelFeatureValue value,
    winrt::Windows::AI::MachineLearning::ILearningModelFeatureDescriptor descriptor) {
  using namespace compatibility_details;

  auto pfnAreKindsCompatible =
      FeatureKindCompatibilityMatrix
          [static_cast<unsigned>(value.Kind())][static_cast<unsigned>(descriptor.Kind())];

  pfnAreKindsCompatible(value, descriptor);
}

}  // namespace Windows::AI::MachineLearning

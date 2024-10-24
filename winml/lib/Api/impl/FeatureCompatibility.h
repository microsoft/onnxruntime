// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "ImageFeatureDescriptor.h"
#include "ImageFeatureValue.h"
#include "IMapFeatureValue.h"
#include "ISequenceFeatureValue.h"
#include "TensorFeatureDescriptor.h"
#include "NamespaceAliases.h"

namespace _winml {

namespace error_strings {

// This must be kept in sync with the TensorKind enum in Windows.AI.MachineLearning.idl
__declspec(selectany) const char* SzTensorKind[] = {
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

static std::string ToString(winml::ILearningModelFeatureDescriptor descriptor);

static std::string ToString(const std::vector<int64_t>& shape) {
  std::ostringstream stream;
  stream << "[";
  std::copy(shape.begin(), shape.end(), std::ostream_iterator<int64_t>(stream, ","));
  stream << "]";

  return stream.str();
}

static std::string ToString(wfc::IVectorView<int64_t> shape) {
  auto shapeVec = std::vector<int64_t>(begin(shape), end(shape));
  return ToString(shapeVec);
}

static std::string ToString(winml::TensorKind kind, wfc::IVectorView<int64_t> shape) {
  // Any unrecognized data type is considered "Undefined".
  if (static_cast<uint32_t>(kind) >= std::size(SzTensorKind)) {
    kind = winml::TensorKind::Undefined;
  }

  std::ostringstream stream;
  stream << SzTensorKind[static_cast<uint32_t>(kind)] << ToString(shape);
  return stream.str();
}

static std::string ToString(winml::ITensorFeatureDescriptor descriptor) {
  return ToString(descriptor.TensorKind(), descriptor.Shape());
}

static std::string ToString(winml::ITensor value) {
  return ToString(value.TensorKind(), value.Shape());
}

static std::string ToString(winml::IMapFeatureDescriptor descriptor) {
  auto keyKind = descriptor.KeyKind();
  // Any unrecognized data type is considered "Undefined".
  if (static_cast<uint32_t>(keyKind) >= std::size(SzTensorKind)) {
    keyKind = winml::TensorKind::Undefined;
  }

  auto valueDescriptor = descriptor.ValueDescriptor();
  std::ostringstream stream;
  stream << "Map<" << SzTensorKind[static_cast<uint32_t>(keyKind)] << "," << ToString(valueDescriptor) << ">";
  return stream.str();
}

static std::string ToString(winrt::com_ptr<_winml::IMapFeatureValue> value) {
  winml::TensorKind keyKind;
  FAIL_FAST_IF_FAILED(value->get_KeyKind(&keyKind));
  // Any unrecognized data type is considered "Undefined".
  if (static_cast<uint32_t>(keyKind) >= std::size(SzTensorKind)) {
    keyKind = winml::TensorKind::Undefined;
  }

  winml::ILearningModelFeatureDescriptor valueDescriptor;
  FAIL_FAST_IF_FAILED(value->get_ValueDescriptor(&valueDescriptor));
  std::ostringstream stream;
  stream << "Map<" << SzTensorKind[static_cast<uint32_t>(keyKind)] << "," << ToString(valueDescriptor) << ">";
  return stream.str();
}

static std::string ToString(winml::ISequenceFeatureDescriptor descriptor) {
  auto elementDescriptor = descriptor.ElementDescriptor();
  std::ostringstream stream;
  stream << "Sequence<" << ToString(elementDescriptor) << ">";
  return stream.str();
}

static std::string ToString(winrt::com_ptr<_winml::ISequenceFeatureValue> value) {
  winml::ILearningModelFeatureDescriptor elementDescriptor;
  FAIL_FAST_IF_FAILED(value->get_ElementDescriptor(&elementDescriptor));

  std::ostringstream stream;
  stream << "Sequence<" << ToString(elementDescriptor) << ">";
  return stream.str().c_str();
}

static std::string ToString(winml::IImageFeatureDescriptor descriptor) {
  std::ostringstream stream;
  stream << "Image[" << descriptor.Width() << "x" << descriptor.Height() << "]";
  return stream.str();
}

static std::string ToString(winrt::com_ptr<winmlp::ImageFeatureValue> value) {
  std::ostringstream stream;
  stream << "Image[" << value->Widths()[0] << "x" << value->Heights()[0] << "]";
  return stream.str();
}

static std::string ToString(winml::ILearningModelFeatureDescriptor descriptor) {
  switch (descriptor.Kind()) {
    case winml::LearningModelFeatureKind::Image:
      return ToString(descriptor.as<winml::IImageFeatureDescriptor>());
      break;
    case winml::LearningModelFeatureKind::Map:
      return ToString(descriptor.as<winml::IMapFeatureDescriptor>());
      break;
    case winml::LearningModelFeatureKind::Sequence:
      return ToString(descriptor.as<winml::ISequenceFeatureDescriptor>());
      break;
    case winml::LearningModelFeatureKind::Tensor:
      return ToString(descriptor.as<winml::ITensorFeatureDescriptor>());
    default:
      FAIL_FAST_MSG("Unexpected descriptor LearningModelFeatureKind.");
  }
}

static std::string ToString(winml::ILearningModelFeatureValue value) {
  switch (value.Kind()) {
    case winml::LearningModelFeatureKind::Image:
      return ToString(value.as<winmlp::ImageFeatureValue>());
      break;
    case winml::LearningModelFeatureKind::Map:
      return ToString(value.as<_winml::IMapFeatureValue>());
      break;
    case winml::LearningModelFeatureKind::Sequence:
      return ToString(value.as<_winml::ISequenceFeatureValue>());
      break;
    case winml::LearningModelFeatureKind::Tensor:
      return ToString(value.as<winml::ITensor>());
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

using K = winml::LearningModelFeatureKind;

static void not_compatible_hr(
  HRESULT hr, winml::ILearningModelFeatureValue value, winml::ILearningModelFeatureDescriptor descriptor
) {
  auto name = _winml::Strings::UTF8FromHString(descriptor.Name());

  WINML_THROW_IF_FAILED_MSG(
    hr,
    "Model variable %s, expects %s, but binding was attempted with an incompatible type %s.",
    name.c_str(),
    error_strings::ToString(descriptor).c_str(),
    error_strings::ToString(value).c_str()
  );
}

static void not_compatible(winml::ILearningModelFeatureValue value, winml::ILearningModelFeatureDescriptor descriptor) {
  not_compatible_hr(WINML_ERR_INVALID_BINDING, value, descriptor);
}

// This method is used in validating sequeance and map type's internal element, key and value types.
static HRESULT verify(winml::ILearningModelFeatureDescriptor first, winml::ILearningModelFeatureDescriptor second) {
  RETURN_HR_IF(WINML_ERR_INVALID_BINDING, first.Kind() != second.Kind());

  if (auto mapFirst = first.try_as<winml::MapFeatureDescriptor>()) {
    auto mapSecond = second.try_as<winml::MapFeatureDescriptor>();
    RETURN_HR_IF_NULL(WINML_ERR_INVALID_BINDING, mapSecond);
    RETURN_HR_IF(WINML_ERR_INVALID_BINDING, mapFirst.KeyKind() != mapSecond.KeyKind());
    return verify(mapFirst.ValueDescriptor(), mapSecond.ValueDescriptor());
  }

  if (auto sequenceFirst = first.try_as<winml::SequenceFeatureDescriptor>()) {
    auto sequenceSecond = second.try_as<winml::SequenceFeatureDescriptor>();
    RETURN_HR_IF_NULL(WINML_ERR_INVALID_BINDING, sequenceSecond);
    return verify(sequenceFirst.ElementDescriptor(), sequenceSecond.ElementDescriptor());
  }

  if (auto tensorFirst = first.try_as<winml::TensorFeatureDescriptor>()) {
    auto tensorSecond = second.try_as<winml::TensorFeatureDescriptor>();
    RETURN_HR_IF_NULL(WINML_ERR_INVALID_BINDING, tensorSecond);
    RETURN_HR_IF(WINML_ERR_INVALID_BINDING, tensorFirst.TensorKind() != tensorSecond.TensorKind());
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
void verify(winml::ILearningModelFeatureValue value, winml::ILearningModelFeatureDescriptor descriptor) {
  not_compatible(value, descriptor);
}

template <>
void verify<K::Tensor, K::Tensor>(
  winml::ILearningModelFeatureValue value, winml::ILearningModelFeatureDescriptor descriptor
) {
  thrower fail = std::bind(not_compatible_hr, std::placeholders::_1, value, descriptor);
  enforce check = std::bind(enforce_not_false, std::placeholders::_1, std::placeholders::_2, fail);

  auto tensorValue = value.as<winml::ITensor>();
  auto tensorDescriptor = descriptor.as<winml::ITensorFeatureDescriptor>();
  check(WINML_ERR_INVALID_BINDING, tensorValue.TensorKind() == tensorDescriptor.TensorKind());

  auto spValueProvider = tensorValue.as<_winml::ILotusValueProviderPrivate>();

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
  winml::ILearningModelFeatureValue value, winml::ILearningModelFeatureDescriptor descriptor
) {
  thrower fail = std::bind(not_compatible_hr, std::placeholders::_1, value, descriptor);
  enforce check = std::bind(enforce_not_false, std::placeholders::_1, std::placeholders::_2, fail);
  enforce_succeeded check_succeeded = std::bind(enforce_not_failed, std::placeholders::_1, fail);

  auto spMapFeatureValue = value.as<_winml::IMapFeatureValue>();
  auto mapDescriptor = descriptor.as<winml::IMapFeatureDescriptor>();

  winml::TensorKind valueKeyKind;
  check_succeeded(spMapFeatureValue->get_KeyKind(&valueKeyKind));
  check(WINML_ERR_INVALID_BINDING, valueKeyKind == mapDescriptor.KeyKind());

  winml::ILearningModelFeatureDescriptor valueValueDescriptor;
  check_succeeded(spMapFeatureValue->get_ValueDescriptor(&valueValueDescriptor));

  check_succeeded(verify(valueValueDescriptor, mapDescriptor.ValueDescriptor()));
}

template <>
void verify<K::Sequence, K::Sequence>(
  winml::ILearningModelFeatureValue value, winml::ILearningModelFeatureDescriptor descriptor
) {
  thrower fail = std::bind(not_compatible_hr, std::placeholders::_1, value, descriptor);
  enforce_succeeded check_succeeded = std::bind(enforce_not_failed, std::placeholders::_1, fail);

  auto spSequenceFeatureValue = value.as<_winml::ISequenceFeatureValue>();
  auto sequenceDescriptor = descriptor.as<winml::ISequenceFeatureDescriptor>();

  winml::ILearningModelFeatureDescriptor valueElementDescriptor;
  check_succeeded(spSequenceFeatureValue->get_ElementDescriptor(&valueElementDescriptor));

  check_succeeded(verify(valueElementDescriptor, sequenceDescriptor.ElementDescriptor()));
}

template <>
void verify<K::Image, K::Image>(
  winml::ILearningModelFeatureValue value, winml::ILearningModelFeatureDescriptor descriptor
) {
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
  winml::ILearningModelFeatureValue value, winml::ILearningModelFeatureDescriptor descriptor
) {
  thrower fail = std::bind(not_compatible_hr, std::placeholders::_1, value, descriptor);
  enforce check = std::bind(enforce_not_false, std::placeholders::_1, std::placeholders::_2, fail);
  enforce_succeeded check_succeeded = std::bind(enforce_not_failed, std::placeholders::_1, fail);

  auto tensorValue = value.as<winml::ITensor>();
  auto imageDescriptor = descriptor.as<winmlp::ImageFeatureDescriptor>();

  check(WINML_ERR_INVALID_BINDING, tensorValue.TensorKind() == winml::TensorKind::Float);

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
           https://github.com/onnx/onnx/blob/main/docs/MetadataProps.md
           Supported metadata values in RS5
               - Image.BitmapPixelFormat: Gray8, RGB8, BGR8
               - Image.ColorSpaceGamma: SRGB
               - Image.NominalPixelRagne: NominalRange_0_255
    */
template <>
void verify<K::Image, K::Tensor>(
  winml::ILearningModelFeatureValue value, winml::ILearningModelFeatureDescriptor descriptor
) {
  thrower fail = std::bind(not_compatible_hr, std::placeholders::_1, value, descriptor);
  enforce check = std::bind(enforce_not_false, std::placeholders::_1, std::placeholders::_2, fail);

  auto imageValue = value.as<winmlp::ImageFeatureValue>();
  auto tensorDescriptor = descriptor.as<winmlp::TensorFeatureDescriptor>();

  check(WINML_ERR_INVALID_BINDING, !tensorDescriptor->IsUnsupportedMetaData());
  // NCHW: images must be 4 dimensions
  auto tensorDescriptorShape = tensorDescriptor->Shape();
  check(WINML_ERR_SIZE_MISMATCH, 4 == tensorDescriptorShape.Size());
}

static void (*FeatureKindCompatibilityMatrix[4][4])(
  winml::ILearningModelFeatureValue, winml::ILearningModelFeatureDescriptor
) = {
  //                 Tensor,                          Sequence,                           Map,                    Image
  /* Tensor */ {verify<K::Tensor, K::Tensor>, not_compatible, not_compatible, verify<K::Tensor, K::Image>},
 /* Sequence */
  {not_compatible, verify<K::Sequence, K::Sequence>, not_compatible, not_compatible},
 /* Map */
  {not_compatible, not_compatible, verify<K::Map, K::Map>, not_compatible},
 /* Image */
  {verify<K::Image, K::Tensor>, not_compatible, not_compatible, verify<K::Image, K::Image>}
};
}  // namespace compatibility_details

inline void VerifyFeatureValueCompatibleWithDescriptor(
  winml::ILearningModelFeatureValue value, winml::ILearningModelFeatureDescriptor descriptor
) {
  using namespace compatibility_details;

  auto pfnAreKindsCompatible =
    FeatureKindCompatibilityMatrix[static_cast<unsigned>(value.Kind())][static_cast<unsigned>(descriptor.Kind())];

  pfnAreKindsCompatible(value, descriptor);
}

}  // namespace _winml

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"

#include <evntrace.h>

#include "OnnxruntimeDescriptorConverter.h"
#include "ImageFeatureDescriptor.h"
#include "MapFeatureDescriptor.h"
#include "SequenceFeatureDescriptor.h"
#include "TensorFeatureDescriptor.h"

#include "winrt/windows.foundation.collections.h"
#include "winrt/windows.graphics.imaging.h"

#include "OnnxruntimeEngine.h"

#include "OnnxruntimeErrors.h"

// BitmapPixelFormat constants
static const char* c_bitmap_pixel_format_key = "Image.BitmapPixelFormat";
static const char* c_supported_pixel_formats[] =
    {
        "Gray8",
        "Rgb8",
        "Bgr8"};

// ColorSpaceGamma constants
// Unlike the other supported value arrays, this is an UNSUPPORTED list.
// Unfortunately, the original RS5 implementation blocked unsupported
// color_space_gamma values (Linear), and did not allow the actual supported
// values (SRGB).
static const char* c_color_space_key = "Image.ColorSpaceGamma";
static const char* c_unsupported_color_spaces[] =
    {
        "Linear"};

// NominalPixelRange constants
static const char* c_nominal_range_key = "Image.NominalPixelRange";
static const char* c_supported_nominal_ranges[] =
    {
        "NominalRange_0_255",
        "Normalized_0_1",
        "Normalized_1_1"};

namespace _winml {

// Forward declare CreateFeatureDescriptor
static winml::ILearningModelFeatureDescriptor
CreateFeatureDescriptor(
    OnnxruntimeEngineFactory* engine_factory,
    const OnnxruntimeValueInfoWrapper* feature_descriptor,
    const std::unordered_map<std::string, std::string>& metadata);

static winml::TensorKind
TensorKindFromONNXTensorElementDataType(ONNXTensorElementDataType dataType) {
  switch (dataType) {
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL: {
      return winml::TensorKind::Boolean;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
      return winml::TensorKind::String;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: {
      return winml::TensorKind::Float16;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      return winml::TensorKind::Float;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
      return winml::TensorKind::Double;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8: {
      return winml::TensorKind::Int8;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16: {
      return winml::TensorKind::Int16;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32: {
      return winml::TensorKind::Int32;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      return winml::TensorKind::Int64;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8: {
      return winml::TensorKind::UInt8;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16: {
      return winml::TensorKind::UInt16;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32: {
      return winml::TensorKind::UInt32;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64: {
      return winml::TensorKind::UInt64;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64: {
      return winml::TensorKind::Complex64;
    }
    case ONNXTensorElementDataType::ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128: {
      return winml::TensorKind::Complex128;
    }
    default: {
      return winml::TensorKind::Undefined;
    }
  }
}

static std::string
TensorKindToString(winml::TensorKind tensorKind) {
  switch (tensorKind) {
    case winml::TensorKind::Float: {
      return "float";
    }
    case winml::TensorKind::UInt8: {
      return "uint8";
    }
    case winml::TensorKind::Int8: {
      return "int8";
    }
    case winml::TensorKind::UInt16: {
      return "uint16";
    }
    case winml::TensorKind::Int16: {
      return "int16";
    }
    case winml::TensorKind::Int32: {
      return "int32";
    }
    case winml::TensorKind::Int64: {
      return "int64";
    }
    case winml::TensorKind::String: {
      return "string";
    }
    case winml::TensorKind::Boolean: {
      return "boolean";
    }
    case winml::TensorKind::Float16: {
      return "float16";
    }
    case winml::TensorKind::Double: {
      return "double";
    }
    case winml::TensorKind::UInt32: {
      return "uint32";
    }
    case winml::TensorKind::UInt64: {
      return "uint64";
    }
    case winml::TensorKind::Complex64: {
      return "complex64";
    }
    case winml::TensorKind::Complex128: {
      return "complex128";
    }
    case winml::TensorKind::Undefined:
    default: {
      return "undefined";
    }
  }
}

static const char*
FetchMetadataValueOrNull(
    const std::unordered_map<std::string, std::string>& metadata,
    const char* metadata_key) {
  auto metadata_pair = metadata.find(metadata_key);
  auto metadata_exists = metadata_pair != metadata.end();
  return metadata_exists
             ? metadata_pair->second.c_str()
             : nullptr;
}

template <unsigned TNumSupportedValues>
static bool
IsValueInRange(
    const char* value,
    const char* (&range)[TNumSupportedValues]) {
  if (value) {
    auto range_end = range + TNumSupportedValues;
    auto found = std::find_if(
                     range,
                     range_end,
                     [&](auto& supported_value) {
                       return std::strcmp(supported_value, value) == 0;
                     }) != range_end;
    return found;
  }
  return false;
}

enum class RangeType { AllowedList,
                       BlockedList };

template <unsigned TNumSupportedValues>
static bool
CheckImageMetadataIsUnsupported(
    const std::unordered_map<std::string, std::string>& metadata,
    const char* metadata_key,
    const char* (&range)[TNumSupportedValues],
    std::ostringstream& log_stream,
    RangeType range_type = RangeType::AllowedList) {
  // Check is the model has pixel format metadata.
  // This is retrieved from the metadata (which is global to the model).
  // We only consider formats that are supported in the image converter.
  // If not supported you MUST bind as a tensor.
  auto value = FetchMetadataValueOrNull(metadata, metadata_key);
  auto metadata_exists = value != nullptr;
  if (metadata_exists) {
    auto found = IsValueInRange(value, range);

    // if list of allowed values
    auto is_allowed_list = range_type == RangeType::AllowedList;
    auto is_not_in_allowed_list = is_allowed_list && !found;

    // if list of blocked values
    auto is_blocked_list = range_type == RangeType::BlockedList;
    auto is_in_blocked_list = is_blocked_list && found;

    auto is_unsupported = is_not_in_allowed_list || is_in_blocked_list;

    // log
    if (is_unsupported) {
      log_stream << "Unsupported "
                 << metadata_key
                 << ": "
                 << value
                 << " found."
                 << std::endl;
    }

    return is_unsupported;
  }

  // No metadata, so it cannot be unsupported
  return false;
}

static std::pair<wgi::BitmapPixelFormat, wgi::BitmapAlphaMode>
CreateBitmapPixelFormatAndAlphaModeInfo(
    const char* pixel_format) {
  if (pixel_format) {
    auto comparator =
        std::bind(std::strcmp, pixel_format, std::placeholders::_1);

    if (0 == comparator("Gray8")) {
      return {wgi::BitmapPixelFormat::Gray8, wgi::BitmapAlphaMode::Premultiplied};
    } else if (0 == comparator("Rgb8")) {
      return {wgi::BitmapPixelFormat::Rgba8, wgi::BitmapAlphaMode::Premultiplied};
    } else if (0 == comparator("Bgr8")) {
      return {wgi::BitmapPixelFormat::Bgra8, wgi::BitmapAlphaMode::Premultiplied};
    } else if (0 == comparator("Rgba8")) {
      return {wgi::BitmapPixelFormat::Rgba8, wgi::BitmapAlphaMode::Straight};
    } else if (0 == comparator("Bgra8")) {
      return {wgi::BitmapPixelFormat::Bgra8, wgi::BitmapAlphaMode::Straight};
    }
  }

  // default value, non conforming values are overridden to Bgra8, Premultiplied
  return {wgi::BitmapPixelFormat::Bgra8, wgi::BitmapAlphaMode::Premultiplied};
}

static winmlp::ImageColorSpaceGamma
CreateImageColorSpaceGamma(const char* color_space_gamma) {
  if (color_space_gamma) {
    auto comparator =
        std::bind(std::strcmp, color_space_gamma, std::placeholders::_1);

    if (0 == comparator("Linear")) {
      return winmlp::ImageColorSpaceGamma::ImageColorSpaceGamma_Linear;
    } else if (0 == comparator("SRGB")) {
      return winmlp::ImageColorSpaceGamma::ImageColorSpaceGamma_SRGB;
    }
  }

  // default value, non conforming values are overridden to SRGB
  return winmlp::ImageColorSpaceGamma::ImageColorSpaceGamma_SRGB;
}

static winml::LearningModelPixelRange
CreateImageNominalPixelRange(const char* nominal_range) {
  if (nominal_range) {
    auto comparator =
        std::bind(std::strcmp, nominal_range, std::placeholders::_1);

    if (0 == comparator("NominalRange_0_255")) {
      return winml::LearningModelPixelRange::ZeroTo255;
    } else if (0 == comparator("Normalized_0_1")) {
      return winml::LearningModelPixelRange::ZeroToOne;
    } else if (0 == comparator("Normalized_1_1")) {
      return winml::LearningModelPixelRange::MinusOneToOne;
    }
  }

  // default value, non conforming values are overridden to NominalRange_0_255
  return winml::LearningModelPixelRange::ZeroTo255;
}

enum class TensorType { Tensor_Data,
                        Tensor_Image,
                        Tensor_Data_UnsupportedImageMetadata };

static TensorType
GetTensorType(
    OnnxruntimeEngineFactory* engine_factory,
    OrtTypeInfo* type_info,
    const std::unordered_map<std::string, std::string>& metadata) {
  const char* denotation;
  size_t len;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetDenotationFromTypeInfo(type_info, &denotation, &len),
                      engine_factory->UseOrtApi());

  constexpr char c_image[] = "IMAGE";
  auto has_image_denotation = strncmp(denotation, c_image, _countof(c_image)) == 0;
  if (!has_image_denotation) {
    return TensorType::Tensor_Data;
  }

  // Create log_stream to capture any warning messages
  // for improperly annotated image tensor
  std::ostringstream log_stream;

  // Check if the tensor value_info_proto is of type float.
  // IMAGE tensors MUST be of type float
  const OrtTensorTypeAndShapeInfo* tensor_info;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->CastTypeInfoToTensorInfo(type_info, &tensor_info),
                      engine_factory->UseOrtApi());

  ONNXTensorElementDataType tensor_element_data_type;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetTensorElementType(tensor_info, &tensor_element_data_type),
                      engine_factory->UseOrtApi());

  auto tensor_kind = _winml::TensorKindFromONNXTensorElementDataType(tensor_element_data_type);
  auto is_float_tensor = tensor_kind == winml::TensorKind::Float;
  if (!is_float_tensor) {
    log_stream << "Unsupported image with " << TensorKindToString(tensor_kind)
               << " found." << std::endl;
  }

  // Check if the model has pixel format and color space metadata.
  // This is retrieved from the metadata (which is global to the model).
  // We only consider formats that are supported in the image converter.
  // If not supported you MUST bind as a tensor.
  auto has_unsupported_pixel_format =
      CheckImageMetadataIsUnsupported(metadata, c_bitmap_pixel_format_key,
                                      c_supported_pixel_formats, log_stream);
  auto has_unsupported_nominal_range =
      CheckImageMetadataIsUnsupported(metadata, c_nominal_range_key,
                                      c_supported_nominal_ranges, log_stream);

  // Unfortunately, the original RS5 implementation blocked unsupported
  // color_space_gamma values (Linear), and did not allow the actual supported
  // values (SRGB) like the other image metadata.
  //
  // So to keep parity with RS5, we continue to check against a list of
  // unsupported color spaces.
  auto has_unsupported_color_space_gamma =
      CheckImageMetadataIsUnsupported(metadata, c_color_space_key,
                                      c_unsupported_color_spaces, log_stream, RangeType::BlockedList);

  bool has_unsupported_image_metadata =
      has_unsupported_pixel_format ||
      has_unsupported_color_space_gamma ||
      has_unsupported_nominal_range;

  auto is_tensor_improperly_annotated_as_image =
      has_image_denotation &&
      (!is_float_tensor ||
       has_unsupported_image_metadata);

  if (is_tensor_improperly_annotated_as_image) {
    TraceLoggingWrite(winml_trace_logging_provider,
                      "WinMLInputValidation",
                      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
                      TraceLoggingLevel(WINEVENT_LEVEL_WARNING),
                      TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
                      TraceLoggingString(log_stream.str().c_str()));
  }

  auto is_valid_image_tensor =
      has_image_denotation && is_float_tensor && !has_unsupported_image_metadata;

  return is_valid_image_tensor
             ? TensorType::Tensor_Image
             : has_unsupported_image_metadata
                   ? TensorType::Tensor_Data_UnsupportedImageMetadata
                   : TensorType::Tensor_Data;
}

static winml::ILearningModelFeatureDescriptor
CreateTensorFeatureDescriptor(
    OnnxruntimeEngineFactory* engine_factory,
    const OnnxruntimeValueInfoWrapper* feature_descriptor,
    const std::unordered_map<std::string, std::string>& metadata,
    bool has_unsupported_image_metadata) {
  auto type_info = feature_descriptor->type_info_.get();

  const OrtTensorTypeAndShapeInfo* tensor_info;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->CastTypeInfoToTensorInfo(type_info, &tensor_info),
                      engine_factory->UseOrtApi());
  size_t num_dims;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetDimensionsCount(tensor_info, &num_dims),
                      engine_factory->UseOrtApi());

  auto shape = std::vector<int64_t>(num_dims);
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetDimensions(tensor_info, shape.data(), shape.size()),
                      engine_factory->UseOrtApi());

  ONNXTensorElementDataType tensor_element_data_type;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetTensorElementType(tensor_info, &tensor_element_data_type),
                      engine_factory->UseOrtApi());

  auto kind = _winml::TensorKindFromONNXTensorElementDataType(tensor_element_data_type);

  auto descriptor = winrt::make<winmlp::TensorFeatureDescriptor>(
      feature_descriptor->name_,
      feature_descriptor->description_,  // description
      kind,
      shape,
      feature_descriptor->name_length_ > 0,  // is_required
      has_unsupported_image_metadata);

  return descriptor.as<winml::ILearningModelFeatureDescriptor>();
}

static winml::ILearningModelFeatureDescriptor
CreateImageFeatureDescriptor(
    OnnxruntimeEngineFactory* engine_factory,
    const OnnxruntimeValueInfoWrapper* feature_descriptor,
    const std::unordered_map<std::string, std::string>& metadata) {
  auto type_info = feature_descriptor->type_info_.get();

  const OrtTensorTypeAndShapeInfo* tensor_info;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->CastTypeInfoToTensorInfo(type_info, &tensor_info),
                      engine_factory->UseOrtApi());

  size_t num_dims;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetDimensionsCount(tensor_info, &num_dims),
                      engine_factory->UseOrtApi());

  auto shape = std::vector<int64_t>(num_dims);
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetDimensions(tensor_info, shape.data(), shape.size()),
                      engine_factory->UseOrtApi());

  ONNXTensorElementDataType tensor_element_data_type;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetTensorElementType(tensor_info, &tensor_element_data_type),
                      engine_factory->UseOrtApi());
  auto kind = _winml::TensorKindFromONNXTensorElementDataType(tensor_element_data_type);

  // pixel format and alpha
  auto pixel_format_value = FetchMetadataValueOrNull(metadata, c_bitmap_pixel_format_key);
  auto format_info = CreateBitmapPixelFormatAndAlphaModeInfo(pixel_format_value);
  auto pixel_format = format_info.first;
  auto alpha_mode = format_info.second;

  // color space gamma value
  auto color_space_gamma_value = FetchMetadataValueOrNull(metadata, c_color_space_key);
  auto color_space_gamma = CreateImageColorSpaceGamma(color_space_gamma_value);

  // nominal range
  auto nominal_range_value = FetchMetadataValueOrNull(metadata, c_nominal_range_key);
  auto nominal_range = CreateImageNominalPixelRange(nominal_range_value);

  // The current code assumes that the shape will be in NCHW.
  // Should the model metadata be read instead???
  const int c_height_dimension = 2;
  const int c_width_dimension = 3;
  auto height = static_cast<uint32_t>(shape[c_height_dimension]);
  auto width = static_cast<uint32_t>(shape[c_width_dimension]);
  auto descriptor = winrt::make<winmlp::ImageFeatureDescriptor>(
      feature_descriptor->name_,
      feature_descriptor->description_,
      kind,
      shape,
      feature_descriptor->name_length_ > 0,  // is_required
      pixel_format,
      alpha_mode,
      width,
      height,
      nominal_range,
      color_space_gamma);

  return descriptor.as<winml::ILearningModelFeatureDescriptor>();
}

static winml::ILearningModelFeatureDescriptor
CreateMapFeatureDescriptor(
    OnnxruntimeEngineFactory* engine_factory,
    const OnnxruntimeValueInfoWrapper* feature_descriptor,
    const std::unordered_map<std::string, std::string>& metadata) {
  auto type_info = feature_descriptor->type_info_.get();

  const OrtMapTypeInfo* map_info;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->CastTypeInfoToMapTypeInfo(type_info, &map_info),
                      engine_factory->UseOrtApi());

  ONNXTensorElementDataType map_key_data_type;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetMapKeyType(map_info, &map_key_data_type),
                      engine_factory->UseOrtApi());

  auto key_kind = _winml::TensorKindFromONNXTensorElementDataType(map_key_data_type);

  OrtTypeInfo* map_value_type_info;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetMapValueType(map_info, &map_value_type_info),
                      engine_factory->UseOrtApi());

  UniqueOrtTypeInfo unique_map_value_type_info(map_value_type_info, engine_factory->UseOrtApi()->ReleaseTypeInfo);

  OnnxruntimeValueInfoWrapper dummy_ort_value_info_wrapper;
  dummy_ort_value_info_wrapper.description_ = feature_descriptor->description_;
  dummy_ort_value_info_wrapper.description_length_ = feature_descriptor->description_length_;
  dummy_ort_value_info_wrapper.name_ = feature_descriptor->name_;
  dummy_ort_value_info_wrapper.name_length_ = feature_descriptor->name_length_;
  dummy_ort_value_info_wrapper.type_info_ = std::move(unique_map_value_type_info);

  auto value_descriptor =
      CreateFeatureDescriptor(engine_factory, &dummy_ort_value_info_wrapper, metadata);

  auto descriptor = winrt::make<winmlp::MapFeatureDescriptor>(
      feature_descriptor->name_,
      feature_descriptor->description_,
      feature_descriptor->name_length_ > 0,  // is_required
      key_kind,
      value_descriptor);
  return descriptor.as<winml::ILearningModelFeatureDescriptor>();
}

static winml::ILearningModelFeatureDescriptor
CreateSequenceFeatureDescriptor(
    OnnxruntimeEngineFactory* engine_factory,
    const OnnxruntimeValueInfoWrapper* feature_descriptor,
    const std::unordered_map<std::string, std::string>& metadata) {
  auto type_info = feature_descriptor->type_info_.get();

  const OrtSequenceTypeInfo* sequence_info;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->CastTypeInfoToSequenceTypeInfo(type_info, &sequence_info),
                      engine_factory->UseOrtApi());

  OrtTypeInfo* sequence_element_type_info;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetSequenceElementType(sequence_info, &sequence_element_type_info),
                      engine_factory->UseOrtApi());

  UniqueOrtTypeInfo unique_sequence_element_type_info(sequence_element_type_info, engine_factory->UseOrtApi()->ReleaseTypeInfo);

  OnnxruntimeValueInfoWrapper dummy_ort_value_info_wrapper;
  dummy_ort_value_info_wrapper.description_ = feature_descriptor->description_;
  dummy_ort_value_info_wrapper.description_length_ = feature_descriptor->description_length_;
  dummy_ort_value_info_wrapper.name_ = feature_descriptor->name_;
  dummy_ort_value_info_wrapper.name_length_ = feature_descriptor->name_length_;
  dummy_ort_value_info_wrapper.type_info_ = std::move(unique_sequence_element_type_info);

  auto element_descriptor =
      CreateFeatureDescriptor(engine_factory, &dummy_ort_value_info_wrapper, metadata);

  auto descriptor = winrt::make<winmlp::SequenceFeatureDescriptor>(
      feature_descriptor->name_,
      feature_descriptor->description_,
      feature_descriptor->name_length_ > 0,  // is_required
      element_descriptor);

  return descriptor.as<winml::ILearningModelFeatureDescriptor>();
}

static winml::ILearningModelFeatureDescriptor
CreateFeatureDescriptor(
    OnnxruntimeEngineFactory* engine_factory,
    const OnnxruntimeValueInfoWrapper* feature_descriptor,
    const std::unordered_map<std::string, std::string>& metadata) {
  auto type_info = feature_descriptor->type_info_.get();

  ONNXType onnx_type;
  THROW_IF_NOT_OK_MSG(engine_factory->UseOrtApi()->GetOnnxTypeFromTypeInfo(type_info, &onnx_type),
                      engine_factory->UseOrtApi());

  switch (onnx_type) {
    case ONNXType::ONNX_TYPE_TENSOR: {
      auto tensor_type = GetTensorType(engine_factory, type_info, metadata);
      if (tensor_type == TensorType::Tensor_Image) {
        return CreateImageFeatureDescriptor(
            engine_factory,
            feature_descriptor,
            metadata);
      } else {
        auto has_unsupported_image_metadata =
            tensor_type == TensorType::Tensor_Data_UnsupportedImageMetadata;
        return CreateTensorFeatureDescriptor(
            engine_factory,
            feature_descriptor,
            metadata,
            has_unsupported_image_metadata);
      }
    }
    case ONNXType::ONNX_TYPE_MAP: {
      return CreateMapFeatureDescriptor(
          engine_factory,
          feature_descriptor,
          metadata);
    }
    case ONNXType::ONNX_TYPE_SEQUENCE: {
      return CreateSequenceFeatureDescriptor(
          engine_factory,
          feature_descriptor,
          metadata);
    }
    default:
      throw winrt::hresult_not_implemented();
  }
}

OnnxruntimeDescriptorConverter::OnnxruntimeDescriptorConverter(
    OnnxruntimeEngineFactory* engine_factory,
    const std::unordered_map<std::string, std::string>& metadata) : engine_factory_(engine_factory), metadata_(metadata) {}

wfc::IVector<winml::ILearningModelFeatureDescriptor>
OnnxruntimeDescriptorConverter::ConvertToLearningModelDescriptors(const OnnxruntimeValueInfoWrapper* descriptors, size_t num_descriptors) {
  auto features = winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>();

  for (size_t i = 0; i < num_descriptors;  i++) {
    const auto& descriptor = descriptors[i];
    auto learning_model_descriptor = _winml::CreateFeatureDescriptor(engine_factory_.Get(), &descriptor, metadata_);
    features.Append(learning_model_descriptor);
  }

  return features;
}
}  // namespace _winml

// Copyright (c) Microsoft Corporation.
// Licensed under the MIT License.

#include "pch.h"

#include <evntrace.h>

#include "FeatureDescriptorFactory.h"
#include "ImageFeatureDescriptor.h"
#include "MapFeatureDescriptor.h"
#include "SequenceFeatureDescriptor.h"
#include "TensorFeatureDescriptor.h"

#include "winrt/windows.foundation.collections.h"
#include "winrt/windows.graphics.imaging.h"

using namespace winrt::Windows::AI::MachineLearning;

// BitmapPixelFormat constants
static const char* c_bitmap_pixel_format_key = "Image.BitmapPixelFormat";
static const char* c_supported_pixel_formats[] =
{
  "Gray8",
  "Rgb8",
  "Bgr8"
};

// ColorSpaceGamma constants
// Unlike the other supported value arrays, this is an UNSUPPORTED list.
// Unfortunately, the original RS5 implementation blocked unsupported
// color_space_gamma values (Linear), and did not allow the actual supported
// values (SRGB).
static const char* c_color_space_key = "Image.ColorSpaceGamma";
static const char* c_unsupported_color_spaces[] =
{
  "Linear"
};

// NominalPixelRange constants
static const char* c_nominal_range_key = "Image.NominalPixelRange";
static const char* c_supported_nominal_ranges[] =
{
  "NominalRange_0_255"
};

namespace Windows::AI::MachineLearning {

//
//// Forward declare CreateFeatureDescriptor
//static winml::ILearningModelFeatureDescriptor
//CreateFeatureDescriptor(
//    const onnx::ValueInfoProto* value_info_proto,
//    const std::unordered_map<std::string, std::string>& metadata);
//
//static TensorKind
//TensorKindFromOnnxDataType(
//    ONNX_NAMESPACE::TensorProto_DataType dataType) {
//  using TensorType = ONNX_NAMESPACE::TensorProto_DataType;
//  switch (dataType) {
//    case TensorType::TensorProto_DataType_BOOL: {
//      return TensorKind::Boolean;
//    }
//    case TensorType::TensorProto_DataType_STRING: {
//      return TensorKind::String;
//    }
//    case TensorType::TensorProto_DataType_FLOAT16: {
//      return TensorKind::Float16;
//    }
//    case TensorType::TensorProto_DataType_FLOAT: {
//      return TensorKind::Float;
//    }
//    case TensorType::TensorProto_DataType_DOUBLE: {
//      return TensorKind::Double;
//    }
//    case TensorType::TensorProto_DataType_INT8: {
//      return TensorKind::Int8;
//    }
//    case TensorType::TensorProto_DataType_INT16: {
//      return TensorKind::Int16;
//    }
//    case TensorType::TensorProto_DataType_INT32: {
//      return TensorKind::Int32;
//    }
//    case TensorType::TensorProto_DataType_INT64: {
//      return TensorKind::Int64;
//    }
//    case TensorType::TensorProto_DataType_UINT8: {
//      return TensorKind::UInt8;
//    }
//    case TensorType::TensorProto_DataType_UINT16: {
//      return TensorKind::UInt16;
//    }
//    case TensorType::TensorProto_DataType_UINT32: {
//      return TensorKind::UInt32;
//    }
//    case TensorType::TensorProto_DataType_UINT64: {
//      return TensorKind::UInt64;
//    }
//    case TensorType::TensorProto_DataType_COMPLEX64: {
//      return TensorKind::Complex64;
//    }
//    case TensorType::TensorProto_DataType_COMPLEX128: {
//      return TensorKind::Complex128;
//    }
//    default: { return TensorKind::Undefined; }
//  }
//}
//
//static std::string
//TensorKindToString(TensorKind tensorKind) {
//  switch (tensorKind) {
//    case TensorKind::Float: {
//      return "float";
//    }
//    case TensorKind::UInt8: {
//      return "uint8";
//    }
//    case TensorKind::Int8: {
//      return "int8";
//    }
//    case TensorKind::UInt16: {
//      return "uint16";
//    }
//    case TensorKind::Int16: {
//      return "int16";
//    }
//    case TensorKind::Int32: {
//      return "int32";
//    }
//    case TensorKind::Int64: {
//      return "int64";
//    }
//    case TensorKind::String: {
//      return "string";
//    }
//    case TensorKind::Boolean: {
//      return "boolean";
//    }
//    case TensorKind::Float16: {
//      return "float16";
//    }
//    case TensorKind::Double: {
//      return "double";
//    }
//    case TensorKind::UInt32: {
//      return "uint32";
//    }
//    case TensorKind::UInt64: {
//      return "uint64";
//    }
//    case TensorKind::Complex64: {
//      return "complex64";
//    }
//    case TensorKind::Complex128: {
//      return "complex128";
//    }
//    case TensorKind::Undefined:
//    default: { return "undefined"; }
//  }
//}
//
//static std::vector<int64_t>
//ConvertShapeProtoToVector(
//    const ::onnx::TensorShapeProto& shape_proto) {
//  std::vector<int64_t> shape;
//  for (int i = 0; i < shape_proto.dim_size(); i++) {
//    auto& dim = shape_proto.dim(i);
//    if (dim.has_dim_param()) {
//      shape.push_back(-1);
//    } else if (dim.has_dim_value()) {
//      shape.push_back(dim.dim_value());
//    } else {
//      winrt::throw_hresult(E_INVALIDARG);
//    }
//  }
//
//  return shape;
//}
//
//static const char*
//FetchMetadataValueOrNull(
//    const std::unordered_map<std::string, std::string>& metadata,
//    const char* metadata_key) {
//  auto metadata_pair = metadata.find(metadata_key);
//  auto metadata_exists = metadata_pair != metadata.end();
//  return metadata_exists
//             ? metadata_pair->second.c_str()
//             : nullptr;
//}
//
//template <unsigned TNumSupportedValues>
//static bool
//IsValueInRange(
//    const char* value,
//    const char* (&range)[TNumSupportedValues]) {
//  if (value) {
//    auto range_end = range + TNumSupportedValues;
//    auto found = std::find_if(
//                     range,
//                     range_end,
//                     [&](auto& supported_value) {
//                       return std::strcmp(supported_value, value) == 0;
//                     }) != range_end;
//    return found;
//  }
//  return false;
//}
//
//enum class RangeType { AllowedList,
//                       BlockedList };
//
//template <unsigned TNumSupportedValues>
//static bool
//CheckImageMetadataIsUnsupported(
//    const std::unordered_map<std::string, std::string>& metadata,
//    const char* metadata_key,
//    const char* (&range)[TNumSupportedValues],
//    std::ostringstream& log_stream,
//    RangeType range_type = RangeType::AllowedList) {
//  // Check is the model has pixel format metadata.
//  // This is retrieved from the metadata (which is global to the model).
//  // We only consider formats that are supported in the image converter.
//  // If not supported you MUST bind as a tensor.
//  auto value = FetchMetadataValueOrNull(metadata, metadata_key);
//  auto metadata_exists = value != nullptr;
//  if (metadata_exists) {
//    auto found = IsValueInRange(value, range);
//
//    // if list of allowed values
//    auto is_allowed_list = range_type == RangeType::AllowedList;
//    auto is_not_in_allowed_list = is_allowed_list && !found;
//
//    // if list of blocked values
//    auto is_blocked_list = range_type == RangeType::BlockedList;
//    auto is_in_blocked_list = is_blocked_list && found;
//
//    auto is_unsupported = is_not_in_allowed_list || is_in_blocked_list;
//
//    // log
//    if (is_unsupported) {
//      log_stream << "Unsupported "
//                 << metadata_key
//                 << ": "
//                 << value
//                 << " found."
//                 << std::endl;
//    }
//
//    return is_unsupported;
//  }
//
//  // No metadata, so it cannot be unsupported
//  return false;
//}
//
//static std::pair<wgi::BitmapPixelFormat, wgi::BitmapAlphaMode>
//CreateBitmapPixelFormatAndAlphaModeInfo(
//    const char* pixel_format) {
//  if (pixel_format) {
//    auto comparator =
//        std::bind(std::strcmp, pixel_format, std::placeholders::_1);
//
//    if (0 == comparator("Gray8")) {
//      return {wgi::BitmapPixelFormat::Gray8, wgi::BitmapAlphaMode::Premultiplied};
//    } else if (0 == comparator("Rgb8")) {
//      return {wgi::BitmapPixelFormat::Rgba8, wgi::BitmapAlphaMode::Premultiplied};
//    } else if (0 == comparator("Bgr8")) {
//      return {wgi::BitmapPixelFormat::Bgra8, wgi::BitmapAlphaMode::Premultiplied};
//    } else if (0 == comparator("Rgba8")) {
//      return {wgi::BitmapPixelFormat::Rgba8, wgi::BitmapAlphaMode::Straight};
//    } else if (0 == comparator("Bgra8")) {
//      return {wgi::BitmapPixelFormat::Bgra8, wgi::BitmapAlphaMode::Straight};
//    }
//  }
//
//  // default value, non conforming values are overridden to Bgra8, Premultiplied
//  return {wgi::BitmapPixelFormat::Bgra8, wgi::BitmapAlphaMode::Premultiplied};
//}
//
//static winmlp::ImageColorSpaceGamma
//CreateImageColorSpaceGamma(const char* color_space_gamma) {
//  using namespace winmlp;
//
//  if (color_space_gamma) {
//    auto comparator =
//        std::bind(std::strcmp, color_space_gamma, std::placeholders::_1);
//
//    if (0 == comparator("Linear")) {
//      return ImageColorSpaceGamma::ImageColorSpaceGamma_Linear;
//    } else if (0 == comparator("SRGB")) {
//      return ImageColorSpaceGamma::ImageColorSpaceGamma_SRGB;
//    }
//  }
//
//  // default value, non conforming values are overridden to SRGB
//  return ImageColorSpaceGamma::ImageColorSpaceGamma_SRGB;
//}
//
//static winmlp::ImageNominalPixelRange
//CreateImageNominalPixelRange(const char* nominal_range) {
//  using namespace winmlp;
//
//  if (nominal_range) {
//    auto comparator =
//        std::bind(std::strcmp, nominal_range, std::placeholders::_1);
//
//    if (0 == comparator("NominalRange_0_255")) {
//      return ImageNominalPixelRange::ImageNominalPixelRange_NominalRange_0_255;
//    } else if (0 == comparator("Normalized_0_1")) {
//      return ImageNominalPixelRange::ImageNominalPixelRange_Normalized_0_1;
//    } else if (0 == comparator("Normalized_1_1")) {
//      return ImageNominalPixelRange::ImageNominalPixelRange_Normalized_1_1;
//    } else if (0 == comparator("NominalRange_16_235")) {
//      return ImageNominalPixelRange::ImageNominalPixelRange_NominalRange_16_235;
//    }
//  }
//
//  // default value, non conforming values are overridden to NominalRange_0_255
//  return ImageNominalPixelRange::ImageNominalPixelRange_NominalRange_0_255;
//}
//
//enum class TensorType { Tensor_Data,
//                        Tensor_Image,
//                        Tensor_Data_UnsupportedImageMetadata };
//
//static TensorType
//GetTensorType(
//    const ::onnx::ValueInfoProto* value_info_proto,
//    const std::unordered_map<std::string, std::string>& metadata) {
//  const auto& type_proto = value_info_proto->type();
//
//  THROW_HR_IF_MSG(
//      E_FAIL,
//      type_proto.has_tensor_type() == false,
//      "Malformed onnx file.");
//
//  auto has_image_denotation = type_proto.denotation() == "IMAGE";
//  if (!has_image_denotation) {
//    return TensorType::Tensor_Data;
//  }
//
//  // Create log_stream to capture any warning messages
//  // for improperly annotated image tensor
//  std::ostringstream log_stream;
//
//  // Check if the tensor value_info_proto is of type float.
//  // IMAGE tensors MUST be of type float
//  const auto& tensor_type = type_proto.tensor_type();
//  auto tensor_kind = WinML::TensorKindFromOnnxDataType(
//      onnx::TensorProto_DataType(tensor_type.elem_type()));
//  auto is_float_tensor = tensor_kind == TensorKind::Float;
//  if (!is_float_tensor) {
//    log_stream << "Unsupported image with " << TensorKindToString(tensor_kind)
//               << " found." << std::endl;
//  }
//
//  // Check if the model has pixel format and color space metadata.
//  // This is retrieved from the metadata (which is global to the model).
//  // We only consider formats that are supported in the image converter.
//  // If not supported you MUST bind as a tensor.
//  auto has_unsupported_pixel_format =
//      CheckImageMetadataIsUnsupported(metadata, c_bitmap_pixel_format_key,
//                                      c_supported_pixel_formats, log_stream);
//  auto has_unsupported_nominal_range =
//      CheckImageMetadataIsUnsupported(metadata, c_nominal_range_key,
//                                      c_supported_nominal_ranges, log_stream);
//
//  // Unfortunately, the original RS5 implementation blocked unsupported
//  // color_space_gamma values (Linear), and did not allow the actual supported
//  // values (SRGB) like the other image metadata.
//  //
//  // So to keep parity with RS5, we continue to check against a list of
//  // unsupported color spaces.
//  auto has_unsupported_color_space_gamma =
//      CheckImageMetadataIsUnsupported(metadata, c_color_space_key,
//                                      c_unsupported_color_spaces, log_stream, RangeType::BlockedList);
//
//  bool has_unsupported_image_metadata =
//      has_unsupported_pixel_format ||
//      has_unsupported_color_space_gamma ||
//      has_unsupported_nominal_range;
//
//  auto is_tensor_improperly_annotated_as_image =
//      has_image_denotation &&
//      (!is_float_tensor ||
//       has_unsupported_image_metadata);
//
//  if (is_tensor_improperly_annotated_as_image) {
//    TraceLoggingWrite(winmla::winml_trace_logging_provider,
//                      "WinMLInputValidation",
//                      TraceLoggingKeyword(WINML_PROVIDER_KEYWORD_DEFAULT),
//                      TraceLoggingLevel(WINEVENT_LEVEL_WARNING),
//                      TraceLoggingOpcode(EVENT_TRACE_TYPE_INFO),
//                      TraceLoggingString(log_stream.str().c_str()));
//  }
//
//  auto is_valid_image_tensor =
//      has_image_denotation && is_float_tensor && !has_unsupported_image_metadata;
//
//  return is_valid_image_tensor
//             ? TensorType::Tensor_Image
//             : has_unsupported_image_metadata
//                   ? TensorType::Tensor_Data_UnsupportedImageMetadata
//                   : TensorType::Tensor_Data;
//}
//
//static winml::ILearningModelFeatureDescriptor
//CreateTensorFeatureDescriptor(
//    const onnx::ValueInfoProto* value_info_proto,
//    const std::unordered_map<std::string, std::string>& metadata,
//    bool has_unsupported_image_metadata) {
//  const auto& type_proto = value_info_proto->type();
//  const auto& tensor_type = type_proto.tensor_type();
//  auto shape = WinML::ConvertShapeProtoToVector(tensor_type.shape());
//  auto kind = WinML::TensorKindFromOnnxDataType(
//      onnx::TensorProto_DataType(tensor_type.elem_type()));
//
//  TensorFeatureDescriptor descriptor(
//      WinML::Strings::HStringFromUTF8(value_info_proto->name()),
//      WinML::Strings::HStringFromUTF8(value_info_proto->doc_string()),  // description
//      value_info_proto->name().empty() == false,                        // is_required
//      kind,
//      shape,
//      has_unsupported_image_metadata);
//
//  return descriptor.as<winml::ILearningModelFeatureDescriptor>();
//}
//
//static winml::ILearningModelFeatureDescriptor
//CreateImageFeatureDescriptor(
//    const onnx::ValueInfoProto* value_info_proto,
//    const std::unordered_map<std::string, std::string>& metadata) {
//  const auto& type_proto = value_info_proto->type();
//  const auto& tensor_type = type_proto.tensor_type();
//  auto shape = WinML::ConvertShapeProtoToVector(tensor_type.shape());
//  auto kind = WinML::TensorKindFromOnnxDataType(
//  onnx::TensorProto_DataType(tensor_type.elem_type()));
//
//  // pixel format and alpha
//  auto pixel_format_value = FetchMetadataValueOrNull(metadata, c_bitmap_pixel_format_key);
//  auto format_info = CreateBitmapPixelFormatAndAlphaModeInfo(pixel_format_value);
//  auto pixel_format = format_info.first;
//  auto alpha_mode = format_info.second;
//
//  // paulm:   commenting this out during layering.    gamma and nominal are never used
//  // since we only support one of them.  if a non support one is set, they all fall back 
//  // to TensorFeatureDescriptor (invalid image metadata)
//#ifdef DONE_LAYERING
//  // color space gamma value
//    auto color_space_gamma_value = FetchMetadataValueOrNull(metadata, c_color_space_key);
//    auto color_space_gamma = CreateImageColorSpaceGamma(color_space_gamma_value);
//
//  // nominal range
//    auto nominal_range_value = FetchMetadataValueOrNull(metadata, c_nominal_range_key);
//    auto nominal_range = CreateImageNominalPixelRange(nominal_range_value);
//#endif
//
//  // The current code assumes that the shape will be in NCHW.
//  // Should the model metadata be read instead???
//  const int c_height_dimension = 2;
//  const int c_width_dimension = 3;
//  auto height = static_cast<uint32_t>(shape[c_height_dimension]);
//  auto width = static_cast<uint32_t>(shape[c_width_dimension]);
//  ImageFeatureDescriptor descriptor(
//      WinML::Strings::HStringFromUTF8(value_info_proto->name()),
//      WinML::Strings::HStringFromUTF8(value_info_proto->doc_string()),
//      value_info_proto->name().empty() == false,  // is_required
//      kind,
//      shape,
//      pixel_format,
//      alpha_mode,
//      width,
//      height);
//
//  return descriptor.as<winml::ILearningModelFeatureDescriptor>();
//}
//
//static winml::ILearningModelFeatureDescriptor
//CreateMapFeatureDescriptor(
//    const onnx::ValueInfoProto* value_info_proto,
//    const std::unordered_map<std::string, std::string>& metadata) {
//  const auto& type_proto = value_info_proto->type();
//  auto type_proto_map = type_proto.map_type();
//
//  auto key_kind = WinML::TensorKindFromOnnxDataType(
//      onnx::TensorProto_DataType(type_proto_map.key_type()));
//
//  onnx::ValueInfoProto dummy_value_info_proto;
//  dummy_value_info_proto.set_name(value_info_proto->name().c_str());
//  dummy_value_info_proto.set_doc_string(value_info_proto->doc_string().c_str());
//  *dummy_value_info_proto.mutable_type() = type_proto_map.value_type();
//
//  auto value_descriptor =
//      CreateFeatureDescriptor(&dummy_value_info_proto, metadata);
//
//  MapFeatureDescriptor descriptor(
//      WinML::Strings::HStringFromUTF8(value_info_proto->name()),
//      WinML::Strings::HStringFromUTF8(value_info_proto->doc_string()),
//      value_info_proto->name().empty() == false,  // is_rRequired
//      key_kind,
//      value_descriptor);
//  return descriptor.as<winml::ILearningModelFeatureDescriptor>();
//}
//
//static winml::ILearningModelFeatureDescriptor
//CreateSequenceFeatureDescriptor(
//    const onnx::ValueInfoProto* value_info_proto,
//    const std::unordered_map<std::string, std::string>& metadata) {
//  const auto& type_proto = value_info_proto->type();
//  // assert(typeProto->has_sequence_type());
//  auto type_proto_sequence = type_proto.sequence_type();
//
//  onnx::ValueInfoProto dummy_value_info_proto;
//  dummy_value_info_proto.set_name(value_info_proto->name().c_str());
//  dummy_value_info_proto.set_doc_string(value_info_proto->doc_string().c_str());
//  *dummy_value_info_proto.mutable_type() = type_proto_sequence.elem_type();
//
//  auto element_descriptor =
//      CreateFeatureDescriptor(&dummy_value_info_proto, metadata);
//
//  SequenceFeatureDescriptor descriptor(
//      WinML::Strings::HStringFromUTF8(value_info_proto->name()),
//      WinML::Strings::HStringFromUTF8(value_info_proto->doc_string()),
//      value_info_proto->name().empty() == false,  // is_required
//      element_descriptor);
//
//  return descriptor.as<winml::ILearningModelFeatureDescriptor>();
//}
//
//static winml::ILearningModelFeatureDescriptor
//CreateFeatureDescriptor(
//    const onnx::ValueInfoProto* value_info_proto,
//    const std::unordered_map<std::string, std::string>& metadata) {
//  const auto& type_proto = value_info_proto->type();
//
//  using ValueCase = ::onnx::TypeProto::ValueCase;
//  switch (type_proto.value_case()) {
//    case ValueCase::kTensorType: {
//      auto tensor_type =
//          GetTensorType(value_info_proto, metadata);
//      if (tensor_type == TensorType::Tensor_Image) {
//        return CreateImageFeatureDescriptor(
//            value_info_proto,
//            metadata);
//      } else {
//        auto has_unsupported_image_metadata =
//            tensor_type == TensorType::Tensor_Data_UnsupportedImageMetadata;
//        return CreateTensorFeatureDescriptor(
//            value_info_proto,
//            metadata,
//            has_unsupported_image_metadata);
//      }
//    }
//    case ValueCase::kMapType: {
//      return CreateMapFeatureDescriptor(
//          value_info_proto,
//          metadata);
//    }
//    case ValueCase::kSequenceType: {
//      return CreateSequenceFeatureDescriptor(
//          value_info_proto,
//          metadata);
//    }
//    default:
//      throw winrt::hresult_not_implemented();
//  }
//}

FeatureDescriptorFactory::FeatureDescriptorFactory(
    const std::unordered_map<std::string, std::string>& metadata) : metadata_(metadata) {}

wfc::IVector<winml::ILearningModelFeatureDescriptor>
FeatureDescriptorFactory::CreateLearningModelFeatureDescriptors(const std::vector<FeatureDescriptor>& descriptors) {
  auto features =
      winrt::single_threaded_vector<winml::ILearningModelFeatureDescriptor>();
/*
  for (auto value_info_proto : value_info_protos) {
    auto descriptor = WinML::CreateFeatureDescriptor(value_info_proto, metadata_);
    features.Append(descriptor);
  }*/

  return features;
}
}  // namespace Windows::AI::MachineLearning

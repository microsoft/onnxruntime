// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>
#include <string>
#include <type_traits>

#include "core/session/onnxruntime_c_api.h"

namespace onnxruntime {
class DataTypeImpl;
}  // namespace onnxruntime

namespace ONNX_NAMESPACE {
class TypeProto;
}

// These types are only present in the winml adapter c api, so they are forward declared.
struct OrtMapTypeInfo;
struct OrtSequenceTypeInfo;
struct OrtOptionalTypeInfo;
struct OrtTensorTypeAndShapeInfo;

/**
 * the equivalent of ONNX_NAMESPACE::TypeProto
 * This class is mainly for the C API
 */
struct OrtTypeInfo {
  ONNXType type;
  std::string denotation;

  std::unique_ptr<OrtTensorTypeAndShapeInfo> data;
  std::unique_ptr<OrtMapTypeInfo> map_type_info;
  std::unique_ptr<OrtSequenceTypeInfo> sequence_type_info;
  std::unique_ptr<OrtOptionalTypeInfo> optional_type_info;

  std::unique_ptr<OrtTypeInfo> Clone() const;

  static std::unique_ptr<OrtTypeInfo> FromOrtValue(const OrtValue& value);
  static std::unique_ptr<OrtTypeInfo> FromTypeProto(const ONNX_NAMESPACE::TypeProto&);
  static const onnxruntime::DataTypeImpl* ElementTypeFromProto(int type);

  explicit OrtTypeInfo(ONNXType type) noexcept;

  explicit OrtTypeInfo(std::unique_ptr<OrtMapTypeInfo> map_type_info) noexcept;

  OrtTypeInfo(ONNXType type, std::unique_ptr<OrtTensorTypeAndShapeInfo> data) noexcept;

  explicit OrtTypeInfo(std::unique_ptr<OrtSequenceTypeInfo> sequence_type_info) noexcept;

  explicit OrtTypeInfo(std::unique_ptr<OrtOptionalTypeInfo> optional_type_info) noexcept;

  OrtTypeInfo(const OrtTypeInfo&) = delete;
  OrtTypeInfo& operator=(const OrtTypeInfo&) = delete;

  ~OrtTypeInfo();

  template <typename... Args>
  static std::unique_ptr<OrtTypeInfo> MakePtr(Args... args) {
    return std::make_unique<OrtTypeInfo>(std::forward<Args>(args)...);
  }
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/flatbuffers.h"
#include "core/framework/allocator.h"
#include "core/platform/env.h"

#include <gsl/gsl>
#include <filesystem>

#include "adapter_format/adapter_schema.fbs.h"

#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

struct OrtValue;

namespace onnxruntime {
namespace adapters {
namespace utils {

/// <summary>
/// Helper class to serialize Lora adapter
/// </summary>
class AdapterFormatBuilder {
 public:
  AdapterFormatBuilder() = default;

  /// <summary>
  /// Appends parameter tensor to the adapter builder
  /// </summary>
  /// <param name="name">parameter name</param>
  /// <param name="data_type"></param>
  /// <param name="shape"></param>
  /// <param name="data"></param>
  void AddParameter(const std::string& name, adapters::TensorDataType data_type,
                    gsl::span<const int64_t> shape, gsl::span<const uint8_t> data);

  /// <summary>
  /// Finishes serialization and returns a serialized byte vector
  /// </summary>
  /// <param name="adapter_version"></param>
  /// <param name="model_version"></param>
  /// <returns></returns>
  std::vector<uint8_t> Finish(int adapter_version, int model_version);

  /// <summary>
  /// Finishes serialization and returns a span to internal buffer.
  /// </summary>
  /// <param name="adapter_version"></param>
  /// <param name="model_version"></param>
  /// <returns></returns>
  gsl::span<uint8_t> FinishWithSpan(int adapter_version, int model_version);

 private:
  void FinishImpl(int adapter_version, int model_version);

  flatbuffers::FlatBufferBuilder builder_;
  std::vector<flatbuffers::Offset<adapters::Parameter>> params_;
};

/// <summary>
///
/// </summary>
/// <param name="bytes"></param>
/// <param name="num_bytes"></param>
/// <returns></returns>
bool IsAdapterFormatModelBytes(const void* bytes, size_t num_bytes);

void LoadStringFromLoraFormat(std::string& dst, const flatbuffers::String* fbs_string);

/// <summary>
/// The function loads the lora adapter bytes from the file system
/// </summary>
/// <param name="file_path">file path</param>
/// <returns>bytes in a vector</returns>
/// <throw>If the path can not be found</throw>
std::vector<uint8_t> LoadLoraAdapterBytes(const std::filesystem::path& file_path);

/// <summary>
/// This function memory maps the adapter file in memory
/// </summary>
/// <param name="file_path"></param>
/// <returns>memory handle and file size in a tuple</returns>
std::pair<Env::MappedMemoryPtr, size_t> MemoryMapAdapterFile(const std::filesystem::path& file_path);

/// <summary>
/// Validates underlying format and the format version
/// </summary>
/// <param name="bytes"></param>
/// <returns>Adapter ptr</returns>
const Adapter* ValidateAndGetAdapterFromBytes(gsl::span<const uint8_t> bytes);

/// <summary>
/// Serializes tensor data into flatbuffer
/// </summary>
/// <param name="flat_builder"></param>
/// <param name="name">parameter name</param>
/// <param name="doc">doc, optional</param>
/// <param name="data_type"></param>
/// <param name="shape"></param>
/// <param name="data"></param>
/// <param name="fbs_tensor">output offset</param>
void SaveLoraParameter(flatbuffers::FlatBufferBuilder& flat_builder, std::string_view name,
                       TensorDataType data_type,
                       gsl::span<const int64_t> shape, gsl::span<const uint8_t> data,
                       flatbuffers::Offset<Parameter>& fbs_tensor);

/// <summary>
/// Create an OrtValue on top of the flatbuffer tensor
/// No copying of data is done here. The caller is responsible for managing the lifetime of flatbuffer
/// structures.
///
/// In this scenario, one can memory map the entire flatbuffer tensor data into OrtValue without copying.
/// </summary>
/// <param name="tensor"></param>
/// <returns></returns>
std::pair<std::string, OrtValue> CreateOrtValueOverLoraParameter(const Parameter& param);
}  // namespace utils
}  // namespace adapters
}  // namespace onnxruntime

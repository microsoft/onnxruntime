// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"
#include "core/framework/ort_value.h"
#include "core/platform/env.h"

#include "lora/lora_format_utils.h"

#include <filesystem>
#include <string>
#include <variant>
#include <vector>

namespace onnxruntime {
namespace lora {

/// <summary>
/// Container to hold and access Lora Parameters
/// </summary>
class LoraAdapter {
 public:
  LoraAdapter() = default;
  explicit LoraAdapter(AllocatorPtr device_allocator)
      : device_allocator_(std::move(device_allocator)) {}
  ~LoraAdapter() = default;
  LoraAdapter(const LoraAdapter&) = delete;
  LoraAdapter& operator=(const LoraAdapter&) = delete;

  LoraAdapter(LoraAdapter&&) = default;
  LoraAdapter& operator=(LoraAdapter&&) = default;

  /// <summary>
  /// Load parameters into memory from an adapter file and validates its format.
  /// </summary>
  /// <param name="file_name">file name that can be opened</param>
  void Load(const std::filesystem::path& file_path);

  /// <summary>
  /// Load parameters from serialized bytes and validates its format.
  /// </summary>
  /// <param name="buffer"></param>
  void Load(std::vector<uint8_t> buffer);

  /// <summary>
  /// Memory maps adapter file into memory and validates its format.
  /// </summary>
  /// <param name="file_name"></param>
  void MemoryMap(const std::filesystem::path& file_path);

  /// <summary>
  /// Returns number of parameters in the adapter.
  /// The number is expected to be even as lora params come in pairs.
  /// </summary>
  /// <returns>size of params_values_ container</returns>
  size_t GetParamNum() const {
    return params_values_.size();
  }

  /// <summary>
  /// Gets lora format version
  /// </summary>
  /// <returns></returns>
  int LoraFormatVersion() const noexcept {
    return adapter_->format_version();
  }

  /// <summary>
  /// Gets adapter version
  /// </summary>
  /// <returns></returns>
  int AdapterVersion() const noexcept {
    return adapter_->adapter_version();
  }

  /// <summary>
  /// Gets model version for which the adapter was created
  /// </summary>
  /// <returns></returns>
  int ModelVersion() const noexcept {
    return adapter_->model_version();
  }

  /// <summary>
  /// Outputs Lora Parameters, their names and values
  /// into the supplied output iterators.
  /// </summary>
  /// <typeparam name="NamesOutputIter"></typeparam>
  /// <typeparam name="TensorOutputIter"></typeparam>
  /// <param name="names_out">output iterator that accepts const char*</param>
  /// <param name="tensor_out">output iterator that accepts OrtValue</param>
  template <class NamesOutputIter, class TensorOutputIter>
  void OutputAdaptersParameters(NamesOutputIter names_out,
                                TensorOutputIter tensor_out) const {
    for (const auto& [name, param] : params_values_) {
      *names_out = name.c_str();
      ++names_out;
      *tensor_out = param.ort_value_mapped_;
      ++tensor_out;
    }
  }

 private:
  void InitializeParamsValues();
  // Get the size of the buffer
  size_t GetBufferSize() const;

  struct BufferHolder {
    explicit BufferHolder(std::vector<uint8_t> buffer) : buffer_(std::move(buffer)) {}
    std::vector<uint8_t> buffer_;
  };

  struct MemMapHolder {
    MemMapHolder(Env::MappedMemoryPtr mapped_memory, size_t file_size)
        : mapped_memory_(std::move(mapped_memory)), file_size_(file_size) {}
    Env::MappedMemoryPtr mapped_memory_;
    size_t file_size_;
  };

  std::variant<std::monostate, MemMapHolder, BufferHolder> buffer_;

  /// <summary>
  /// Represents a named lora parameter (tensor)
  /// </summary>
  struct LoraParam {
    LoraParam() = default;
    explicit LoraParam(OrtValue ort_value_mapped) noexcept;
    LoraParam(OrtValue ort_value_mapped, OrtValue ort_value_device) noexcept;

    OrtValue ort_value_mapped_;
    OrtValue ort_value_device_;
  };

  AllocatorPtr device_allocator_;
  const Adapter* adapter_{nullptr};
  InlinedHashMap<std::string, LoraParam> params_values_;
};

}  // namespace lora
}  // namespace onnxruntime

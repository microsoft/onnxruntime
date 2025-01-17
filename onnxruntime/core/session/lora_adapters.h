// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/common/inlined_containers.h"
#include "core/framework/allocator.h"
#include "core/framework/ort_value.h"
#include "core/platform/env.h"

#include "lora/adapter_format_utils.h"

#include <filesystem>
#include <string>
#include <variant>
#include <vector>
#include <unordered_map>

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
  /// Represents a named lora parameter (tensor)
  /// </summary>
  class Param {
   public:
    Param() = default;
    explicit Param(OrtValue ort_value_mapped) noexcept;
    Param(OrtValue ort_value_mapped, OrtValue ort_value_device) noexcept;

    const OrtValue& GetMapped() const noexcept {
      return ort_value_mapped_;
    }

    // For python interface
    OrtValue& GetMapped() noexcept {
      return ort_value_mapped_;
    }

    const OrtValue& GetDeviceOrMapped() const noexcept {
      if (ort_value_device_.IsAllocated()) {
        return ort_value_device_;
      }
      return ort_value_mapped_;
    }

   private:
    OrtValue ort_value_mapped_;
    OrtValue ort_value_device_;
  };

  using param_const_iterator = std::unordered_map<std::string, Param>::const_iterator;
  using param_iterator = std::unordered_map<std::string, Param>::iterator;

  /// <summary>
  /// Obtain a range of the iterators
  /// </summary>
  /// <returns></returns>
  std::pair<param_const_iterator, param_const_iterator> GetParamIterators() const {
    return std::make_pair(params_values_.cbegin(), params_values_.cend());
  }

  std::pair<param_iterator, param_iterator> GetParamIterators() {
    return std::make_pair(params_values_.begin(), params_values_.end());
  }

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
  int FormatVersion() const noexcept {
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
  /// Outputs Lora Parameters on CPU, their names and values
  /// into the supplied output iterators.
  /// </summary>
  /// <typeparam name="NamesOutputIter"></typeparam>
  /// <typeparam name="TensorOutputIter"></typeparam>
  /// <param name="names_out">output iterator that accepts const char*</param>
  /// <param name="tensor_out">output iterator that accepts const OrtValue*</param>
  template <class NamesOutputIter, class TensorOutputIter>
  void OutputAdapterParameters(NamesOutputIter names_out,
                               TensorOutputIter tensor_out) const {
    for (const auto& [name, param] : params_values_) {
      *names_out = name.c_str();
      ++names_out;
      *tensor_out = &param.GetDeviceOrMapped();
      ++tensor_out;
    }
  }

 private:
  void InitializeParamsValues();

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

  AllocatorPtr device_allocator_;
  const adapters::Adapter* adapter_{nullptr};
  std::unordered_map<std::string, Param> params_values_;
};

}  // namespace lora
}  // namespace onnxruntime

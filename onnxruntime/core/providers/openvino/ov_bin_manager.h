// Copyright (C) Intel Corporation
// Licensed under the MIT License

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <vector>
#include <filesystem>
#include <fstream>
#include <shared_mutex>

#include "openvino/runtime/core.hpp"
#include "weak_singleton.h"

namespace onnxruntime {
namespace openvino_ep {

// Forward declaration
class SharedContext;

// Manages native compiled model blobs and binary file serialization/deserialization
class BinManager {
 public:
  BinManager() = default;
  BinManager(const std::filesystem::path& external_bin_path) : external_bin_path_(external_bin_path) {}
  ~BinManager() = default;

  // Blob management
  void AddNativeBlob(const std::string& name, const ov::CompiledModel& compiled_model);
  ov::Tensor GetNativeBlob(const std::string& blob_name);
  std::unique_ptr<std::istream> GetNativeBlobAsStream(const std::string& blob_name);

  // Serialization/Deserialization
  void Serialize(std::ostream& stream, std::shared_ptr<SharedContext> shared_context = nullptr);
  void Deserialize(std::istream& stream, std::shared_ptr<SharedContext> shared_context = nullptr);

  void Serialize(std::shared_ptr<SharedContext> shared_context = nullptr);
  void Deserialize(std::shared_ptr<SharedContext> shared_context = nullptr);

  // Path management
  void TrySetExternalBinPath(const std::filesystem::path& bin_path) {
    std::unique_lock lock(mutex_);
    if (!external_bin_path_) {
      external_bin_path_ = bin_path;
    }
  }
  std::filesystem::path GetExternalBinPath() const {
    std::shared_lock lock(mutex_);
    return external_bin_path_.value_or("");
  }

  static std::filesystem::path GetBinPathForModel(const std::filesystem::path& model_path);

 private:
  struct BlobContainer {
    ov::CompiledModel compiled_model;
    ov::Tensor tensor;
    std::vector<uint8_t> data;  // For embedded blobs when no external file exists
    struct {
      uint64_t file_offset{0};
      uint64_t size{0};
    } serialized_info;
  };

  void DeserializeImpl(std::istream& stream, const std::shared_ptr<SharedContext>& shared_context);

  mutable std::shared_mutex mutex_;
  std::optional<std::filesystem::path> external_bin_path_;
  ov::Tensor mapped_bin_;
  std::unordered_map<std::string, BlobContainer> native_blobs_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime

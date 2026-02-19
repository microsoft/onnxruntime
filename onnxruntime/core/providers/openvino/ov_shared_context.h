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
#include <mutex>

#include "openvino/runtime/core.hpp"
#include "ov_bin_manager.h"
#include "weak_singleton.h"

namespace onnxruntime {
namespace openvino_ep {

class WeightFileManager;

class SharedContext : public std::enable_shared_from_this<SharedContext> {
 public:
  explicit SharedContext(const std::filesystem::path& bin_path);
  SharedContext() : SharedContext("") {}
  virtual ~SharedContext() {}

  struct Metadata {
    struct Value {
      struct {
        std::filesystem::path location{};
        size_t data_offset{0};
        size_t size{0};
      } serialized;

      std::shared_ptr<const ov::Tensor> tensor;
    };
    using Map = std::unordered_map<std::string, Value>;
  };

  bool IsSharedWeight(const std::string& name) const {
    std::shared_lock lock(mutex_);
    return metadata_.contains(name);
  }

  void AddExternalWeight(const std::string& name, size_t offset, size_t size, const std::filesystem::path& location) {
    Metadata::Value value;
    value.serialized.data_offset = offset;
    value.serialized.size = size;
    value.serialized.location = location;
    std::unique_lock lock(mutex_);
    metadata_[name] = std::move(value);
  }

  Metadata::Map GetMetadataCopy() const {
    std::shared_lock lock(mutex_);
    return metadata_;
  }

  void SetSharedWeightsOnInferRequest(ov::InferRequest& ir, const std::filesystem::path& model_dir);

  void AddNativeBlob(const std::string& name, const ov::CompiledModel& compiled_model) {
    bin_manager_.AddNativeBlob(name, compiled_model);
  }

  ov::Tensor GetNativeBlob(const std::string& blob_name) {
    return bin_manager_.GetNativeBlob(blob_name);
  }

  std::unique_ptr<std::istream> GetNativeBlobAsStream(const std::string& blob_name) {
    return bin_manager_.GetNativeBlobAsStream(blob_name);
  }

  void Serialize(std::ostream& stream);
  void Deserialize(std::istream& stream);
  void Serialize();
  void Deserialize();

  std::filesystem::path GetBinPath() const {
    return bin_manager_.GetExternalBinPath();
  }

  static std::filesystem::path GetBinPathForModel(const std::filesystem::path& model_path) {
    return BinManager::GetBinPathForModel(model_path);
  }

  struct WeightsFile {
    ORT_DISALLOW_COPY_AND_ASSIGNMENT(WeightsFile);
    WeightsFile() = delete;
    virtual ~WeightsFile() = default;
    explicit WeightsFile(const std::filesystem::path& filename);
    void LoadWeights(size_t file_offset, void* data, size_t size);
    const void* TryGetOrCreateDeviceMapping(std::optional<ov::RemoteContext>& remote_context);
    size_t Size() const { return weights_size_; }

   private:
    std::ifstream file_;
    std::filesystem::path file_path_;
    size_t weights_size_;
    struct MappingContainer {
      const void* ptr_{nullptr};
      ov::Tensor tensor_;
    };
    std::map<std::string, MappingContainer> imported_device_tensors_;
  };

 private:
  void
  LoadTensorFromFile(
      Metadata::Value& value,
      const std::filesystem::path& model_dir,
      std::optional<ov::RemoteContext>& remote_context,
      const ov::element::Type& element_type,
      const ov::Shape& dimensions);

  mutable std::shared_mutex mutex_;
  std::filesystem::path bin_path_;
  BinManager bin_manager_;
  std::shared_ptr<WeightFileManager> weight_file_manager_;
  std::unordered_map<std::filesystem::path, std::shared_ptr<WeightsFile>> weight_files_;
  Metadata::Map metadata_;
};

class WeightFileManager : public WeakSingleton<WeightFileManager> {
 public:
  using WeightsFile = SharedContext::WeightsFile;
  std::shared_ptr<WeightsFile> GetOrCreateWeightsFile(const std::filesystem::path& weights_path) {
    auto absolute_path = std::filesystem::absolute(weights_path);
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = files_.try_emplace(absolute_path, nullptr);
    if (inserted) {
      it->second = std::make_shared<WeightsFile>(absolute_path);
    }
    return it->second;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::filesystem::path, std::shared_ptr<WeightsFile>> files_;
};

class SharedContextManager : public WeakSingleton<SharedContextManager> {
 public:
  std::shared_ptr<SharedContext> GetOrCreateActiveSharedContext(const std::filesystem::path& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_context_) {
      return active_context_;
    }
    auto [it, inserted] = contexts_.try_emplace(model_path, nullptr);
    if (inserted) {
      it->second = std::make_shared<SharedContext>(model_path);
    }
    active_context_ = it->second;
    active_context_path_ = model_path;
    return it->second;
  }

  std::shared_ptr<SharedContext> GetOrCreateSharedContext(const std::filesystem::path& model_path) {
    std::lock_guard<std::mutex> lock(mutex_);
    auto [it, inserted] = contexts_.try_emplace(model_path, nullptr);
    if (inserted) {
      it->second = std::make_shared<SharedContext>(model_path);
    }
    return it->second;
  }

  void ClearActiveSharedContext() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (active_context_) {
      contexts_.erase(active_context_path_);
      active_context_path_.clear();
    }
    active_context_ = nullptr;
  }

 private:
  mutable std::mutex mutex_;
  std::unordered_map<std::filesystem::path, std::shared_ptr<SharedContext>> contexts_;
  std::shared_ptr<SharedContext> active_context_;
  std::filesystem::path active_context_path_;
};

}  // namespace openvino_ep
}  // namespace onnxruntime

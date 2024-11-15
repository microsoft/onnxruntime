// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <functional>
#include <memory>
#include <mutex>
#include <vector>

#include <QnnInterface.h>

#include "core/common/common.h"
#include "core/providers/qnn/builder/qnn_model.h"

#pragma once

namespace onnxruntime {

class SharedMemHandles {
 public:
  Qnn_MemHandle_t Get(const void* addr) {
    std::lock_guard g{mutex_};
    const auto it = qnn_mem_handles_.find(addr);
    ORT_ENFORCE(it != qnn_mem_handles_.end(), "Failed to find mem handle associated with address (", addr, ").");
    return it->second;
  }

  void Add(const void* addr, Qnn_MemHandle_t mem_handle) {
    std::lock_guard g{mutex_};
    auto [it, added] = qnn_mem_handles_.emplace(addr, mem_handle);
    ORT_ENFORCE(added,
                "There is already a mem handle (", mem_handle, ") associated with the address (", addr, ").");
  }

  Qnn_MemHandle_t GetAndRemove(const void* addr) {
    std::lock_guard g{mutex_};
    const auto it = qnn_mem_handles_.find(addr);
    ORT_ENFORCE(it != qnn_mem_handles_.end(), "Failed to find mem handle associated with address (", addr, ").");
    const auto qnn_mem_handle = it->second;
    qnn_mem_handles_.erase(it);
    return qnn_mem_handle;
  }

 private:
  std::unordered_map<const void*, Qnn_MemHandle_t> qnn_mem_handles_;
  std::mutex mutex_;
};

class SharedContext {
 public:
  static SharedContext& GetInstance() {
    static SharedContext instance_;
    return instance_;
  }

  bool HasSharedQnnModels() {
    const std::lock_guard<std::mutex> lock(mtx_);
    return !shared_qnn_models_.empty();
  }

  bool HasQnnModel(const std::string& model_name) {
    auto it = find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                      [&model_name](const std::unique_ptr<qnn::QnnModel>& qnn_model) { return qnn_model->Name() == model_name; });
    return it != shared_qnn_models_.end();
  }

  std::unique_ptr<qnn::QnnModel> GetSharedQnnModel(const std::string& model_name) {
    const std::lock_guard<std::mutex> lock(mtx_);
    auto it = find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                      [&model_name](const std::unique_ptr<qnn::QnnModel>& qnn_model) { return qnn_model->Name() == model_name; });
    if (it == shared_qnn_models_.end()) {
      return nullptr;
    }
    auto qnn_model = std::move(*it);
    shared_qnn_models_.erase(it);
    return qnn_model;
  }

  bool SetSharedQnnModel(std::vector<std::unique_ptr<qnn::QnnModel>>&& shared_qnn_models,
                         std::string& duplicate_graph_names) {
    const std::lock_guard<std::mutex> lock(mtx_);
    bool graph_exist = false;
    for (auto& shared_qnn_model : shared_qnn_models) {
      auto& model_name = shared_qnn_model->Name();
      auto it = find_if(shared_qnn_models_.begin(), shared_qnn_models_.end(),
                        [&model_name](const std::unique_ptr<qnn::QnnModel>& qnn_model) { return qnn_model->Name() == model_name; });
      if (it == shared_qnn_models_.end()) {
        shared_qnn_models_.push_back(std::move(shared_qnn_model));
      } else {
        duplicate_graph_names.append(model_name + " ");
        graph_exist = true;
      }
    }

    return graph_exist;
  }

  SharedMemHandles& GetSharedMemHandles() { return shared_mem_handles_; }

 private:
  SharedContext() = default;
  ~SharedContext() = default;
  SharedContext(const SharedContext&) = delete;
  SharedContext& operator=(const SharedContext&) = delete;

  std::vector<std::unique_ptr<qnn::QnnModel>> shared_qnn_models_;
  // Producer sessions can be in parallel
  // Consumer sessions have to be after producer sessions initialized
  std::mutex mtx_;

  // TODO can we avoid keeping mem handles in SharedContext?
  SharedMemHandles shared_mem_handles_;
};

}  // namespace onnxruntime

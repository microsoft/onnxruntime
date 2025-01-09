// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License

#include <memory>
#include <mutex>
#include <vector>

#include "core/common/common.h"
#include "core/providers/qnn/builder/qnn_model.h"

#pragma once

namespace onnxruntime {

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

 private:
  SharedContext() = default;
  ~SharedContext() = default;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(SharedContext);

  std::vector<std::unique_ptr<qnn::QnnModel>> shared_qnn_models_;
  // Producer sessions can be in parallel
  // Consumer sessions have to be after producer sessions initialized
  std::mutex mtx_;
};

}  // namespace onnxruntime

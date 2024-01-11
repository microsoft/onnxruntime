// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/basic_types.h"
namespace onnxruntime {
class GraphViewer;

// helper to generate ids that are unique to model and deterministic, even if the execution provider is shared across
// multiple sessions.
class ModelMetadefIdGenerator {
 public:
  int GenerateId(const onnxruntime::GraphViewer& graph_viewer, HashValue& model_hash);

 private:
  std::unordered_map<HashValue, HashValue> main_graph_hash_;  // map graph instance hash to model contents hash
  std::unordered_map<HashValue, int> model_metadef_id_;       // current unique id for model
};

}  // namespace onnxruntime

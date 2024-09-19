// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <unordered_map>
#include "core/common/basic_types.h"
namespace onnxruntime {
class GraphViewer;

/// <summary>
/// helper to generate ids that are unique to model and deterministic, even if the execution provider is shared across
/// multiple sessions.
/// </summary>
class ModelMetadefIdGenerator {
 public:
  /** Generate a unique id that can be used in a MetaDef name. Values are unique for a model instance.
   The model hash is also returned if you wish to include that in the MetaDef name to ensure uniqueness across models.
   @param graph_viewer[in] Graph viewer that GetCapability was called with. Can be for the main graph or nested graph.
   @param model_hash[out] Returns the hash for the main (i.e. top level) graph in the model.
                          This is created using the model path if available,
                          or the model input names and the output names from all nodes in the main graph.
   */
  int GenerateId(const onnxruntime::GraphViewer& graph_viewer, HashValue& model_hash) const;

 private:
  // mutable as these are caches so we can minimize the hashing required on each usage of GenerateId
  mutable std::unordered_map<HashValue, HashValue> main_graph_hash_;  // map graph instance hash to model contents hash
  mutable std::unordered_map<HashValue, int> model_metadef_id_;       // current unique id for model
};

}  // namespace onnxruntime

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
  /** Generate a unique id that can be used in a MetaDef name. Values are unique for a model instance.
   The model hash is also returned if you wish to include that in the MetaDef name to ensure uniqueness across models.
   @param graph_viewer[in] Graph viewer that GetCapability was called with. Can be for the main graph or nested graph.
   @param model_hash[out] Returns the hash for the main (i.e. top level) graph in the model.
                          This is created using the model path if available,
                          or the model input names and the output names from all nodes in the main graph.
   @remarks e.g. the TensorRT Execution Provider is used in multiple sessions and the underlying infrastructure caches
            compiled kernels, so the name must be unique and deterministic across models and sessions.
            NOTE: Ideally this would be a protected method, but to work across the EP bridge it has to be public and
                  virtual, and ModelMetadefIdGenerator but be defined in the header as well.
   */
  int GenerateId(const onnxruntime::GraphViewer& graph_viewer, HashValue& model_hash);

 private:
  std::unordered_map<HashValue, HashValue> main_graph_hash_;  // map graph instance hash to model contents hash
  std::unordered_map<HashValue, int> model_metadef_id_;       // current unique id for model
};

}  // namespace onnxruntime

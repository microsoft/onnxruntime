// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include <unordered_map>
#include "model_metadef_id_generator.h"
#include "core/platform/ort_mutex.h"
#include "core/graph/graph_viewer.h"
#include "core/framework/murmurhash3.h"

namespace onnxruntime {
int GenerateMetaDefId(const onnxruntime::GraphViewer& graph_viewer, HashValue& model_hash) {
  static std::unordered_map<HashValue, HashValue> main_graph_hash_;  // map graph instance hash to model contents hash
  static std::unordered_map<HashValue, int> model_metadef_id_;       // current unique id for model

  // if the EP is shared across multiple sessions there's a very small potential for concurrency issues.
  // use a lock when generating an id to be paranoid
  static OrtMutex mutex;
  std::lock_guard<OrtMutex> lock(mutex);
  model_hash = 0;

  // find the top level graph
  const Graph* cur_graph = &graph_viewer.GetGraph();
  while (cur_graph->IsSubgraph()) {
    cur_graph = cur_graph->ParentGraph();
  }

  uint32_t instance_hash[4] = {0, 0, 0, 0};

  const Graph& main_graph = *cur_graph;

  // hash the bytes in the Graph instance. we can't just use the address as a new Graph instance may use
  // the same memory (unit tests prove this can occur). the raw bytes of the Graph instance should be a unique
  // fingerprint for the instance that can use used as the key to the hash of the model path/contents.
  MurmurHash3::x86_128(&main_graph, gsl::narrow_cast<int32_t>(sizeof(Graph)), instance_hash[0], &instance_hash);
  HashValue graph_instance_hash = instance_hash[0] | (uint64_t(instance_hash[1]) << 32);

  // if we've already hashed this main graph instance use the cached value
  auto entry = main_graph_hash_.find(graph_instance_hash);
  if (entry != main_graph_hash_.cend()) {
    model_hash = entry->second;
  } else {
    uint32_t hash[4] = {0, 0, 0, 0};

    // prefer path the model was loaded from
    // this may not be available if the model was loaded from a stream or in-memory bytes
    const auto& model_path_str = main_graph.ModelPath().ToPathString();
    if (!model_path_str.empty()) {
      MurmurHash3::x86_128(model_path_str.data(), gsl::narrow_cast<int32_t>(model_path_str.size()), hash[0], &hash);
    } else {
      auto hash_str = [&hash](const std::string& str) {
        MurmurHash3::x86_128(str.data(), gsl::narrow_cast<int32_t>(str.size()), hash[0], &hash);
      };

      // fingerprint the main graph by hashing graph inputs and the ordered outputs from each node
      for (const auto* node_arg : main_graph.GetInputsIncludingInitializers()) {
        hash_str(node_arg->Name());
      }

      // note: process nodes in order defined in model to be deterministic
      for (const auto& node : main_graph.Nodes()) {
        for (const auto* node_arg : node.OutputDefs()) {
          if (node_arg->Exists()) {
            hash_str(node_arg->Name());
          }
        }
      }
    }

    model_hash = hash[0] | (uint64_t(hash[1]) << 32);

    main_graph_hash_[graph_instance_hash] = model_hash;
  }

  // return the current unique id, and increment to update
  return model_metadef_id_[model_hash]++;
}

}  // namespace onnxruntime

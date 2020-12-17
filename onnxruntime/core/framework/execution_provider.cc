// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/framework/execution_provider.h"

#include "core/graph/graph_viewer.h"
#include "core/framework/compute_capability.h"
#include "core/framework/kernel_registry.h"
#include "core/framework/kernel_registry_manager.h"
#include "core/framework/murmurhash3.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {

namespace {
//It assumes max(OrtMemType) <= 1, min(OrtMemType) = -2
inline int MakeKey(int id, OrtMemType mem_type) {
  return id << 2 | (mem_type + 2);
}
}  // namespace

AllocatorPtr IExecutionProvider::GetAllocator(int id, OrtMemType mem_type) const {
  auto iter = allocators_.find(MakeKey(id, mem_type));
  if (iter != allocators_.end()) {
    return iter->second;
  }
  return nullptr;
}

std::vector<std::unique_ptr<ComputeCapability>>
IExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                  const std::vector<const KernelRegistry*>& kernel_registries) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
#if !defined(ORT_MINIMAL_BUILD)
  for (auto& node : graph.Nodes()) {
    for (auto registry : kernel_registries) {
      if (KernelRegistry::HasImplementationOf(*registry, node, Type())) {
        std::unique_ptr<IndexedSubGraph> sub_graph = onnxruntime::make_unique<IndexedSubGraph>();
        sub_graph->nodes.push_back(node.Index());
        result.push_back(onnxruntime::make_unique<ComputeCapability>(std::move(sub_graph)));
        break;
      }
    }
  }

  return result;
#else
  // We have saved hashes to lookup static kernels in an ORT format model so the default behavior is to return an
  // empty vector to leave that in place. An EP that compiles nodes can override this in a minimal build.
  ORT_UNUSED_PARAMETER(graph);
  ORT_UNUSED_PARAMETER(kernel_registries);
  return result;
#endif
}

// Update allocator in the provider if already present; ignore if not.
void IExecutionProvider::ReplaceAllocator(AllocatorPtr allocator) {
  const auto& info = allocator->Info();
  auto ite = mem_info_set_.find(info);
  if (ite != mem_info_set_.end()) {
    const int key = MakeKey(info.id, info.mem_type);
    allocators_[key] = allocator;
  }
}

void IExecutionProvider::InsertAllocator(AllocatorPtr allocator) {
  const OrtMemoryInfo& info = allocator->Info();
  auto ite = mem_info_set_.find(info);
  if (ite != mem_info_set_.end()) {
    ORT_THROW("duplicated allocator");
  }
  const int key = MakeKey(info.id, info.mem_type);
  allocators_.insert({key, allocator});
  mem_info_set_.insert(ite, info);
  allocator_list_.push_back(allocator);
}

#if !defined(ORT_MINIMAL_BUILD)
common::Status IExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& /*fused_node*/,
                                           std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
}

common::Status IExecutionProvider::Compile(const std::vector<onnxruntime::Node*>& /*fused_node*/,
                                           std::string& /*dll_path*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
}
#endif

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
common::Status IExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& /*fused_nodes_and_graphs*/,
                                           std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED);
}
#endif

/** Helper class for creating ids that are unique for a model to use when generating MetaDef names. */
class IExecutionProvider::ModelMetadefIdGenerator {
 public:
  int GenerateId(const onnxruntime::GraphViewer& graph_viewer, uint64_t& model_hash) {
    model_hash = 0;

    // find the top level graph
    const Graph* cur_graph = &graph_viewer.GetGraph();
    while (cur_graph->IsSubgraph()) {
      cur_graph = cur_graph->ParentGraph();
    }

    const Graph& main_graph = *cur_graph;
    intptr_t main_graph_addr = reinterpret_cast<intptr_t>(cur_graph);

    // if we've already hashed this main graph instance use the cached value
    auto entry = main_graph_hash_.find(main_graph_addr);
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

      main_graph_hash_[main_graph_addr] = model_hash;
    }

    // return the current unique id, and increment to update
    return model_metadef_id_[model_hash]++;
  }

 private:
  std::unordered_map<intptr_t, int64_t> main_graph_hash_;  // cache of hashes for main graph in model
  std::unordered_map<int64_t, int> model_metadef_id_;      // current unique id for model
};

int IExecutionProvider::GenerateMetaDefId(const onnxruntime::GraphViewer& graph_viewer, uint64_t& model_hash) const {
  return metadef_id_generator_->GenerateId(graph_viewer, model_hash);
}

IExecutionProvider::IExecutionProvider(const std::string& type) : type_{type} {
  metadef_id_generator_ = onnxruntime::make_unique<ModelMetadefIdGenerator>();
}

IExecutionProvider::~IExecutionProvider() = default;

}  // namespace onnxruntime

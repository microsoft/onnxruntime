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
// It assumes max(OrtMemType) <= 1, min(OrtMemType) = -2
inline int MakeKey(int id, OrtMemType mem_type) {
  return id << 2 | (mem_type + 2);
}
}  // namespace

AllocatorPtr IExecutionProvider::GetAllocator(int device_id, OrtMemType mem_type) const {
  auto iter = allocators_.find(MakeKey(device_id, mem_type));
  if (iter != allocators_.end()) {
    return iter->second;
  }
  return nullptr;
}

std::vector<std::unique_ptr<ComputeCapability>>
IExecutionProvider::GetCapability(const onnxruntime::GraphViewer& graph,
                                  const IKernelLookup& kernel_lookup) const {
  std::vector<std::unique_ptr<ComputeCapability>> result;
  for (const auto& node : graph.Nodes()) {
    if (const KernelCreateInfo* kernel_create_info = kernel_lookup.LookUpKernel(node);
        kernel_create_info != nullptr) {
      std::unique_ptr<IndexedSubGraph> sub_graph = std::make_unique<IndexedSubGraph>();
      sub_graph->nodes.push_back(node.Index());
      result.push_back(std::make_unique<ComputeCapability>(std::move(sub_graph)));
    }
  }

  return result;
}

// Update allocator in the provider if already present; ignore if not.
// We match using the device id, OrtMemType and OrtDevice info.
// We ignore the allocator name, and OrtAllocatorType (whether internally an arena is used or not).
// TODO: We should remove OrtAllocatorType from OrtMemoryInfo as it's an implementation detail of the allocator.
void IExecutionProvider::ReplaceAllocator(AllocatorPtr allocator) {
  const auto& info = allocator->Info();

  // TODO: This only works on allocators that are stored in this class. If a derived class overrides GetAllocator
  // (e.g. the CUDA EP) and stores AllocatorPtr instances in the derived class we know nothing about them.
  // In theory we could call GetAllocator instead of allocators_.find, however the CUDA EP does things this way to
  // return a per-thread allocator from the GetAllocator override, and it's not clear if that could/should be replaced.
  auto iter = allocators_.find(MakeKey(info.id, info.mem_type));
  if (iter != allocators_.end()) {
    // check device as mem_type is relative to the device
    // e.g. OrtMemTypeDefault is CPU for a CPU EP and GPU for a CUDA EP. An individual EP will only have one
    // allocator for an OrtMemType value, so this check is to ensure we don't replace with an incompatible allocator.
    if (iter->second->Info().device == info.device) {
      IAllocator* existing_alloc = iter->second.get();
      for (auto& entry : allocator_list_) {
        if (entry.get() == existing_alloc) {
          entry = allocator;
          break;
        }
      }

      iter->second = allocator;
    }
  }
}

void IExecutionProvider::InsertAllocator(AllocatorPtr allocator) {
  const OrtMemoryInfo& info = allocator->Info();
  const int key = MakeKey(info.id, info.mem_type);

  auto iter = allocators_.find(key);
  if (iter != allocators_.end()) {
    ORT_THROW("Duplicate allocator for OrtMemType:", info.mem_type, " device:", info.device.ToString(),
              " Existing allocator: ", iter->second->Info().name,
              " New allocator: ", allocator->Info().name);
  } else {
    allocators_.insert({key, allocator});
    allocator_list_.push_back(allocator);
  }
}

void IExecutionProvider::RegisterAllocator(AllocatorManager&) {
}

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)
common::Status IExecutionProvider::Compile(const std::vector<FusedNodeAndGraph>& /*fused_nodes_and_graphs*/,
                                           std::vector<NodeComputeInfo>& /*node_compute_funcs*/) {
  return common::Status(common::ONNXRUNTIME, common::NOT_IMPLEMENTED,
                        "IExecutionProvider::Compile with FusedNodeAndGraph is not implemented by " + type_);
}

#endif

int IExecutionProvider::ModelMetadefIdGenerator::GenerateId(const onnxruntime::GraphViewer& graph_viewer,
                                                            HashValue& model_hash) {
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

int IExecutionProvider::GenerateMetaDefId(const onnxruntime::GraphViewer& graph_viewer, HashValue& model_hash) const {
  ORT_ENFORCE(metadef_id_generator_,
              "IExecutionProvider constructor must be called with true for use_metadef_id_creator");

  // if the EP is shared across multiple sessions there's a very small potential for concurrency issues.
  // use a lock when generating an id to be paranoid
  static OrtMutex mutex;
  std::lock_guard<OrtMutex> lock(mutex);
  return metadef_id_generator_->GenerateId(graph_viewer, model_hash);
}

}  // namespace onnxruntime

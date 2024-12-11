// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/prepacked_weights_container.h"
#include "core/framework/allocator_utils.h"
#include "core/graph/graph.h"

namespace onnxruntime {

PrePackedWeights PrePackedWeights::CreateReferringCopy() const {
  PrePackedWeights copy;
  for (const auto& prepacked_buffer : buffers_) {
    // BufferDeleter is nullptr because we do not own the data in this case
    copy.buffers_.emplace_back(prepacked_buffer.get(), BufferDeleter(nullptr));
  }

  copy.buffer_sizes_ = buffer_sizes_;
  return copy;
}

AllocatorPtr PrepackedWeightsContainer::GetOrCreateAllocator(const std::string& device_name) {
  auto iter = allocators_.find(device_name);

  if (iter != allocators_.end())
    return iter->second;

  // Support only CPU based allocators for now.
  // as pre-packing is only supported by CPU kernels for now.
  if (device_name == CPU) {
    // TODO: Investigate benefits of using an arena based allocator
    // For now, we go with a non-arena based allocator
    AllocatorCreationInfo device_info{[](int) { return std::make_unique<CPUAllocator>(); },
                                      0, false};
    auto allocator = CreateAllocator(device_info);

    allocators_[device_name] = allocator;

    return allocator;

  } else {
    ORT_THROW("Unsupported device allocator in the context of pre-packed weights caching: ", device_name);
  }
}

const PrePackedWeights& PrepackedWeightsContainer::GetWeight(const std::string& key) const {
  // .at() will throw if the key doesn't exist
  return prepacked_weights_map_.at(key);
}

bool PrepackedWeightsContainer::WriteWeight(const std::string& key, PrePackedWeights&& packed_weight) {
  auto ret = prepacked_weights_map_.insert(std::make_pair(key, std::move(packed_weight)));
  return ret.second;
}

bool PrepackedWeightsContainer::HasWeight(const std::string& key) const {
  return prepacked_weights_map_.find(key) !=
         prepacked_weights_map_.end();
}

size_t PrepackedWeightsContainer::GetNumberOfElements() const {
  return prepacked_weights_map_.size();
}

PrepackedForSerialization::PrepackedForSerialization()
    : main_graph_(nullptr, key_to_blobs_, false) {
}

PrepackedForSerialization::~PrepackedForSerialization() = default;

void PrepackedForSerialization::Subgraph::InsertFromDisk(const std::string& key, PrePackedWeights&& packed_weight) {
  // We may have duplicate entries mapped from disk if the same weight is pre-packed from subgraphs and
  // up the tree by the same kernel with the same result. The map prevents this from happening.
  key_to_blobs_.emplace(key, std::move(packed_weight));
}

void PrepackedForSerialization::Subgraph::WritePacked(const std::string& weight_name, const std::string& key,
                                                      PrePackedWeights&& packed_weight) {
  auto hit = key_to_blobs_.find(key);
  if (hit == key_to_blobs_.end()) {
    // new key
    key_to_blobs_.emplace(key, std::move(packed_weight));
    if (save_mode_on_) {
      sorted_by_weight_for_writing_[weight_name].insert(key);
    }
    return;
  }

  // Key existed, but may or may not have a reference in this subgraph
  if (save_mode_on_) {
    auto& list = sorted_by_weight_for_writing_[weight_name];
    list.insert(key);
  }
  hit->second = std::move(packed_weight);
}

const PrePackedWeights* PrepackedForSerialization::Subgraph::GetPrepackedWeights(const std::string& key) const {
  auto it = key_to_blobs_.find(key);
  if (it == key_to_blobs_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::optional<PrePackedWeights> PrepackedForSerialization::Subgraph::ReplaceWithReferenceIfSaving(
    const std::string& weight_name,
    const std::string& key,
    const PrePackedWeights& refer_if_absent) {
  auto it = key_to_blobs_.find(key);
  if (it == key_to_blobs_.end()) {
    if (save_mode_on_) {
      key_to_blobs_.emplace(key, refer_if_absent.CreateReferringCopy());
      sorted_by_weight_for_writing_[weight_name].insert(key);
    }
    return std::nullopt;
  }

  PrePackedWeights result = std::move(it->second);
  if (save_mode_on_) {
    it->second = result.CreateReferringCopy();
    auto& list = sorted_by_weight_for_writing_[weight_name];
    list.insert(key);
  } else {
    key_to_blobs_.erase(it);
  }
  return result;
}

PrepackedForSerialization::Subgraph& PrepackedForSerialization::FindOrCreatePrepackedGraph(const Graph& graph) {
  if (graph.ParentGraph() == nullptr) {
    return main_graph_;
  }
  auto& parent = FindOrCreatePrepackedGraph(*graph.ParentGraph());
  return parent.GetOrCreateSubgraph(graph);
}

const PrepackedForSerialization::Subgraph* PrepackedForSerialization::FindPrepackedGraph(const Graph& graph) const {
  if (graph.ParentGraph() == nullptr) {
    return &main_graph_;
  }
  auto* parent = FindPrepackedGraph(*graph.ParentGraph());
  if (parent != nullptr) {
    parent = parent->GetSubgraph(graph);
  }
  return parent;
}

}  // namespace onnxruntime

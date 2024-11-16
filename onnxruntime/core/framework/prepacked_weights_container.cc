// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/prepacked_weights_container.h"
#include "core/framework/allocator_utils.h"
#include "core/graph/graph.h"

namespace onnxruntime {

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

void PrepackedForSerialization::Subgraph::Insert(std::string key, PrePackedWeights&& packed_weight) {
  auto result = key_to_blobs_.emplace(std::move(key), std::move(packed_weight));
  ORT_ENFORCE(result.second, "Duplicate pre-packed weight from disk");
}

bool PrepackedForSerialization::Subgraph::CreateOrOverWrite(const std::string& weight_name, std::string key,
                                                            PrePackedWeights&& packed_weight) {
  // We overwrite the existing key. This is necessary in case we already have a pre-packed weight
  // mapped from disk, but we want to overwrite it with our most recent pre-packed version.
  auto result = key_to_blobs_.insert_or_assign(std::move(key), std::move(packed_weight));
  weight_to_pre_packs_[weight_name].push_back(result.first);
  return result.second;
}

const PrePackedWeights* PrepackedForSerialization::Subgraph::GetPrepackedWeights(const std::string& key) const {
  auto it = key_to_blobs_.find(key);
  if (it == key_to_blobs_.end()) {
    return nullptr;
  }
  return &it->second;
}

PrePackedWeights* PrepackedForSerialization::Subgraph::GetPrepackedWeights(const std::string& key) {
  auto it = key_to_blobs_.find(key);
  if (it == key_to_blobs_.end()) {
    return nullptr;
  }
  return &it->second;
}

std::optional<PrePackedWeights> PrepackedForSerialization::TakePrepackedWeights(const std::string& key) {
  auto it = key_to_blobs_.find(key);
  if (it == key_to_blobs_.end()) {
    return std::nullopt;
  }
  PrePackedWeights result = std::move(it->second);
  key_to_blobs_.erase(it);
  return result;
}

PrepackedForSerialization::Subgraph& PrepackedForSerialization::FindOrCreateSubgraph(const Graph& graph) {
  if (graph.ParentGraph() == nullptr) {
    return main_graph_;
  }
  auto& parent = FindOrCreateSubgraph(*graph.ParentGraph());
  return parent.GetOrCreateSubgraph(graph);
}

}  // namespace onnxruntime

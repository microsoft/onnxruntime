// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/framework/allocator.h"
#include "prepacked_weights.h"

#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>

namespace onnxruntime {

#ifndef SHARED_PROVIDER
class Graph;
#else
struct Graph;
#endif

class PrepackedWeightsContainer final {
 public:
  PrepackedWeightsContainer() {
  }

  ~PrepackedWeightsContainer() = default;

  // Returns an allocator keyed by device name.
  // If an allocator doesn't exist for that specific device, an allocator
  // is created and stored in a member to be returned on subsequent calls.
  // Currently, the only supported device is "Cpu".
  AllocatorPtr GetOrCreateAllocator(const std::string& device_name);

  // Returns the PrePackedWeights instance pertaining to the provided key.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  // Throws an exception if the key doesn't exist
  const PrePackedWeights& GetWeight(const std::string& key) const;

  // Writes the PrePackedWeights instance pertaining to the provided key.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  // Returns a boolean indicating if the insertion took place.
  bool WriteWeight(const std::string& key, PrePackedWeights&& packed_weight);

  // Returns a boolean indicating if there is a PrePackedWeights instance
  // pertaining to the provided key.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  bool HasWeight(const std::string& key) const;

  // Returns the number of elements in the container
  size_t GetNumberOfElements() const;

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PrepackedWeightsContainer);

  // Resource to be acquired by the method that is going to invoke calls to the kernels'
  // PrePack() methods and does the read/write into the pre-packed weights' container.
  // We only want to invoke PrePack() on a kernel that doesn't have a cached version
  // of its pre-packed weight.
  std::mutex mutex_;

  // Define allocators ahead of the container containing tensors because the allocators
  // needs to destructed after the container containing the pre-packed cached tensors
  // because the Tensor buffers will be de-allocated using these allocators
  std::unordered_map<std::string, AllocatorPtr> allocators_;

  // This is an unordered map that holds a mapping between a composite key
  // to PrePackedWeights instances.
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  std::unordered_map<std::string, PrePackedWeights> prepacked_weights_map_;
};

/// <summary>
/// This class has a dual purpose.
/// When saving to disk is ON (IsOverWriteForSave() true)
/// it provides a storage container for PrePackedWeights instances. The pre-packed
/// data is collected using PrepackConstaitInitializers(). In this case newly pre-pack
/// data is used for writing to disk, unless old data matches.
///
/// If saving is OFF, it is used to contain the weights memory mapped from disk.
/// Those weights are then moved to the shared container if weight sharing is enabled.
/// And also the interested kernels.
/// </summary>
class PrepackedForSerialization final {
 public:
  explicit PrepackedForSerialization();
  ~PrepackedForSerialization();

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PrepackedForSerialization);

  using KeyToBlobMap = std::unordered_map<std::string, PrePackedWeights>;
  using KeyToBlobMapIterator = KeyToBlobMap::iterator;
  using BlobsInderect = std::vector<KeyToBlobMapIterator>;
  using BlobsConstIterator = BlobsInderect::const_iterator;

  // Maps weight name to iterators in key_to_blobs_. It associates a weight name with its pre-packs.
  // Normally, a single weight produces a single PrePackedWeights. But it is possible that a weight
  // is pre-packed by different kernels.
  using WeightToPrePacksMap = std::unordered_map<std::string, BlobsInderect>;

  class Subgraph {
   public:
    Subgraph(Subgraph* par, KeyToBlobMap& key_blobs, bool overwrite_for_save)
        : save_mode_on_(overwrite_for_save), parent_(par), key_to_blobs_(key_blobs) {
    }

    ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(Subgraph);

    Subgraph* Parent() noexcept {
      return parent_;
    }

    Subgraph& GetOrCreateSubgraph(const Graph& graph) {
      auto result = subgraph_prepacks_.emplace(&graph, nullptr);
      if (result.second) {
        result.first->second = std::make_unique<Subgraph>(this, key_to_blobs_, save_mode_on_);
      }
      return *result.first->second;
    }

    const Subgraph* GetSubgraph(const Graph& graph) const {
      auto it = subgraph_prepacks_.find(&graph);
      return it == subgraph_prepacks_.end() ? nullptr : it->second.get();
    }

    Subgraph* GetSubgraph(const Graph& graph) {
      auto it = subgraph_prepacks_.find(&graph);
      return it == subgraph_prepacks_.end() ? nullptr : it->second.get();
    }

    // This does not populate per-initializer structures.
    void Insert(std::string key, PrePackedWeights&& packed_weight);

    bool CreateOrOverWrite(const std::string& weight_name, std::string key,
                           PrePackedWeights&& packed_weight);

    const PrePackedWeights* GetPrepackedWeights(const std::string& key) const;

    PrePackedWeights* GetPrepackedWeights(const std::string& key);

    bool IsSaveModeOn() const noexcept {
      return save_mode_on_;
    }

    void SetSaveMode(bool value) noexcept {
      save_mode_on_ = value;
    }

   private:
    bool save_mode_on_;
    Subgraph* parent_ = nullptr;
    KeyToBlobMap& key_to_blobs_;
    WeightToPrePacksMap weight_to_pre_packs_;
    // Map Graph ptr to subgraphs
    std::unordered_map<const Graph*, std::unique_ptr<Subgraph>> subgraph_prepacks_;
  };

  const Subgraph& MainGraph() const noexcept {
    return main_graph_;
  }

  Subgraph& MainGraph() noexcept {
    return main_graph_;
  }

  size_t GetNumberOfKeyedBlobs() const noexcept {
    return key_to_blobs_.size();
  }

  void SetSaveMode(bool value) noexcept {
    main_graph_.SetSaveMode(value);
  }

  bool IsSaveModeOn() const noexcept {
    return main_graph_.IsSaveModeOn();
  }

  std::optional<PrePackedWeights> TakePrepackedWeights(const std::string& key);

  Subgraph& FindOrCreateSubgraph(const Graph& graph);

 private:
  // Map of key to pre-packed blobs.This is common for all subgraphs
  // The key is : op_type + "+" + hash_of_prepacked_buffers_in_the_PrepackedWeights_instance.
  // as defined above. We store keys for all scopes (main graph and subgraphs)
  KeyToBlobMap key_to_blobs_;
  Subgraph main_graph_;
};
}  // namespace onnxruntime

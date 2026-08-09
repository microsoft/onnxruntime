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
#include <unordered_set>

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

// Maps a pre-packed weight blob key to PrepackedWeights instance
using PrepackedKeyToBlobMap = std::unordered_map<std::string, PrePackedWeights>;

/// <summary>
/// This class has a dual purpose.
/// If saving is OFF (IsSaveModeOn() false), it is used to contain the weights memory mapped from disk.
/// Those weights are then moved to the shared container if weight sharing is enabled.
/// If cross-session weight sharing is not enabled, the weights are stored in this container,
/// and shared with the interested kernels.
///
/// When saving to disk is ON (IsSaveModeOn() true)
/// It records the pre-packed weights blobs and associates them with the weight name.
/// When saving the model with external initializers, the weights are written to disk along
/// with the pre-packed blobs.
///
/// </summary>
class PrepackedWeightsForGraph {
 public:
  PrepackedWeightsForGraph(PrepackedKeyToBlobMap& key_blobs, bool save_mode_on_)
      : key_to_blobs_(key_blobs), save_mode_on_(save_mode_on_) {
  }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(PrepackedWeightsForGraph);

  // WeightToPrePacksMap maps weight name to a set of pre-packed
  // keys contained in the KeyToBlobMap
  using KeysPerWeight = std::unordered_set<std::string>;  // blob keys
  using WeightToPrePacksMap = std::unordered_map<std::string, KeysPerWeight>;

  void InsertPrepackedWeights(const std::string& key, PrePackedWeights&& packed_weight);

  // Overwrites the existing weights and associates key with weight_name
  void WritePackedMaybeForSave(const std::string& weight_name, const std::string& key,
                               PrePackedWeights&& packed_weight);

  const PrePackedWeights* GetPrepackedWeights(const std::string& key) const;

  // The function would add or replace existing entry with references to it.
  // If the entry is present, it would replace it with references to the existing entry.
  // If the entry is not present, it would add reference to refer_if_absent
  // If the entry is present it would return the existing entry otherwise std::nullopt
  // Reference in this context means a non-owning smart pointer. Essentially, this function
  // replaces the existing entry with the same entry, but transfers the ownership outside
  // the container.
  std::optional<PrePackedWeights> ReplaceWithReferenceIfSaving(const std::string& weight_name,
                                                               const std::string& key,
                                                               const PrePackedWeights& refer_to_if_absent);

  bool IsSaveModeOn() const noexcept {
    return save_mode_on_;
  }

  void SetSaveMode(bool value) noexcept {
    save_mode_on_ = value;
  }

  const KeysPerWeight* GetKeysForWeightForSaving(const std::string& weight_name) const {
    auto hit = weight_prepacks_for_saving_.find(weight_name);
    if (hit != weight_prepacks_for_saving_.end()) {
      return &hit->second;
    }
    return nullptr;
  }

  size_t GetNumberOfWeightsForWriting() const noexcept {
    return weight_prepacks_for_saving_.size();
  }

  size_t GetNumberOfKeyedBlobsForWriting() const noexcept {
    size_t result = 0;
    for (const auto& [_, keys] : weight_prepacks_for_saving_) {
      result += keys.size();
    }
    return result;
  }

  const WeightToPrePacksMap& GetWeightToPrepack() const noexcept {
    return weight_prepacks_for_saving_;
  }

  PrepackedKeyToBlobMap& GetKeyToBlob() noexcept {
    return key_to_blobs_;
  }

  const PrepackedKeyToBlobMap& GetKeyToBlob() const noexcept {
    return key_to_blobs_;
  }

 private:
  PrepackedKeyToBlobMap& key_to_blobs_;
  bool save_mode_on_;
  WeightToPrePacksMap weight_prepacks_for_saving_;
};

}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>
#include <stdint.h>
#include <utility>

#include "core/platform/ort_mutex.h"

namespace onnxruntime {

/**
 * Random seed generator used to generate new seeds for random engines
 * such as std::default_random_engine.  The default (global) generator
 * will use the seed provided by the user to SetRandomSeed().
 */
class RandomGenerator {
 public:
  explicit RandomGenerator(int64_t seed) : seed_(seed) {}

  /**
   * Resets the generator seed.
   */
  void SetSeed(int64_t seed) {
    seed_.store(seed);
  }

  /**
   * Gets the next seed, optionally incrementing it by the specified count.
   */
  int64_t NextSeed(int64_t count = 1) {
    return seed_.fetch_add(count);
  }

  /**
   * Gets the default global random generator.  This generator will use the
   * seed provided in SetRandomSeed(), and will update if the seed is reset.
   */
  static RandomGenerator& Default();

 private:
  std::atomic<int64_t> seed_;
};

/**
 * Philox pseudo-random number generator.  Philox uses a counter-based design.
 * This generator provides the seed and counter to initialize a Philox random
 * engine such as the CUDA Philox_4x32_10 generator.
 */
class PhiloxGenerator {
 public:
  explicit PhiloxGenerator(uint64_t seed) : seed_(seed), offset_(0) {}

  /**
   * Resets the seed and offset.
   */
  void SetSeed(uint64_t seed) {
    std::lock_guard<OrtMutex> lock(mutex_);
    seed_ = seed;
    offset_ = 0;
  }

  /**
   * Gets the seed and offset pair, incrementing the offset by the specified count.
   */
  std::pair<uint64_t, uint64_t> NextPhiloxSeeds(uint64_t count) {
    std::lock_guard<OrtMutex> lock(mutex_);
    auto seeds = std::make_pair(seed_, offset_);
    offset_ += count;
    return seeds;
  }

  /**
   * Get the default global random generator.  This generator will use the
   * seed provided in SetRandomSeed(), and will update if the seed is reset.
   */
  static PhiloxGenerator& Default();

 private:
  OrtMutex mutex_;
  uint64_t seed_;
  uint64_t offset_;
};

}  // namespace onnxruntime

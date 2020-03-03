// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <atomic>

namespace onnxruntime {
namespace cuda {

class DropoutGenerator {
 public:
  DropoutGenerator() : seed_(0), offset_(0) {}

  DropoutGenerator(uint64_t seed) : seed_(seed), offset_(0) {}

  void SetSeed(uint64_t seed) {
    seed_ = seed;
  }

  std::pair<uint64_t, uint64_t> GetPhiloxSeeds(uint64_t count) {
    uint64_t offset = offset_.fetch_add(count);
    return std::pair<uint64_t, uint64_t>(seed_, offset);
  }

 private:
  uint64_t seed_;
  std::atomic<uint64_t> offset_;
};

template <typename T>
void DropoutKernelImpl(
  const cudaDeviceProp& prop,
  const int64_t N,
  const float ratio,
  DropoutGenerator& generator,
  const T* X_data,
  T* Y_data,
  bool* mask_data);

template <typename T>
void DropoutGradientKernelImpl(
  const int64_t N,
  const T* dY_data,
  const bool* mask_data,
  const float ratio,
  T* dX_data);

}  // namespace cuda
}  // namespace onnxruntime

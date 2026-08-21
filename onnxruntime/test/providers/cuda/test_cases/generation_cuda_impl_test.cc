// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "contrib_ops/cuda/transformers/generation_device_helper.h"
#include "contrib_ops/cuda/transformers/generation_cuda_impl.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace test {
namespace {

struct CudaDeviceMemoryDeleter {
  template <typename T>
  void operator()(T* ptr) const {
    cudaFree(ptr);
  }
};

template <typename T>
std::unique_ptr<T, CudaDeviceMemoryDeleter> AllocateDeviceMemory(size_t count) {
  T* ptr = nullptr;
  CUDA_CALL_THROW(cudaMalloc(&ptr, count * sizeof(T)));
  return std::unique_ptr<T, CudaDeviceMemoryDeleter>(ptr);
}

TEST(GenerationCudaImplTest, EmptyCrossQKPairsSkipLaunches) {
  CUDA_CALL_THROW(cudaGetLastError());

  IAllocatorUniquePtr<float*> qk_layer_pointers;
  EXPECT_STATUS_OK(::onnxruntime::contrib::GenerationCudaDeviceHelper::UpdateDecoderCrossQK(
      0, nullptr, nullptr, qk_layer_pointers, 0, 0, nullptr, nullptr, 0, nullptr));

  EXPECT_STATUS_OK(::onnxruntime::contrib::GenerationCudaDeviceHelper::FinalizeDecoderCrossQK(
      nullptr, 0, 0, 0, 0, 0, 0, nullptr, 0, nullptr, nullptr, 0, nullptr, {}));

  LaunchCopyCrossQKSingleDecodeStep(nullptr, nullptr, nullptr, 0, 1, 1, 1, 0, nullptr, 1, 1);
  CUDA_CALL_THROW(cudaGetLastError());

  LaunchFinalizeCrossQK(nullptr, 2, 1, 1, 1, 1, 0, nullptr, 1, nullptr, nullptr, 1, nullptr, {});
  CUDA_CALL_THROW(cudaGetLastError());
}

TEST(GenerationCudaImplTest, CrossQKPairsAllowDuplicatesAndZeroInvalidIndices) {
  constexpr int frames = 3;
  constexpr int pair_count = 6;
  const std::vector<float> source{1.0f, 2.0f, 3.0f};
  const std::vector<int> pairs{0, 0, 0, 0, -1, 0, 1, 0, 0, -1, 0, 1};
  const std::vector<float> initial_target(pair_count * frames, -1.0f);
  std::vector<float> expected = source;
  expected.insert(expected.end(), source.begin(), source.end());
  expected.resize(initial_target.size(), 0.0f);

  auto device_source = AllocateDeviceMemory<float>(source.size());
  auto device_layer_pointers = AllocateDeviceMemory<float*>(1);
  auto device_pairs = AllocateDeviceMemory<int>(pairs.size());
  auto device_target = AllocateDeviceMemory<float>(initial_target.size());

  float* source_ptr = device_source.get();
  CUDA_CALL_THROW(cudaMemcpy(device_source.get(), source.data(), source.size() * sizeof(float), cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(device_layer_pointers.get(), &source_ptr, sizeof(source_ptr), cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(device_pairs.get(), pairs.data(), pairs.size() * sizeof(int), cudaMemcpyHostToDevice));
  CUDA_CALL_THROW(cudaMemcpy(device_target.get(), initial_target.data(), initial_target.size() * sizeof(float), cudaMemcpyHostToDevice));

  LaunchCopyCrossQKSingleDecodeStep(nullptr,
                                    device_target.get(),
                                    device_layer_pointers.get(),
                                    0,
                                    1,
                                    1,
                                    1,
                                    pair_count,
                                    device_pairs.get(),
                                    frames,
                                    1);
  CUDA_CALL_THROW(cudaGetLastError());
  CUDA_CALL_THROW(cudaDeviceSynchronize());

  std::vector<float> actual(expected.size());
  CUDA_CALL_THROW(cudaMemcpy(actual.data(), device_target.get(), actual.size() * sizeof(float), cudaMemcpyDeviceToHost));
  EXPECT_EQ(actual, expected);
}

}  // namespace
}  // namespace test
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

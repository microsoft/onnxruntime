// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdio>
#include <memory>
#include <vector>

#include "asserts.h"
#include "core/framework/execution_provider.h"
#include "core/framework/tensor.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/file_util.h"

namespace onnxruntime {
namespace test {
namespace {

constexpr size_t kParallelReadThreshold = 16 * 1024 * 1024;
constexpr size_t kStagingBufferSize = 64 * 1024 * 1024;
constexpr size_t kFilePrefixSize = 13;

uint8_t TestValue(size_t index) {
  return static_cast<uint8_t>((index * 31 + 7) & 0xff);
}

void CreateExternalDataFile(size_t length, PathString& path) {
  FILE* file = nullptr;
  path = ORT_TSTR("cuda_external_data_loader_XXXXXX");
  CreateTestFile(file, path);

  std::vector<uint8_t> chunk(1024 * 1024);
  ASSERT_EQ(kFilePrefixSize, fwrite(chunk.data(), 1, kFilePrefixSize, file));
  for (size_t offset = 0; offset < length;) {
    const size_t chunk_size = std::min(chunk.size(), length - offset);
    for (size_t i = 0; i < chunk_size; ++i) {
      chunk[i] = TestValue(offset + i);
    }
    ASSERT_EQ(chunk_size, fwrite(chunk.data(), 1, chunk_size, file));
    offset += chunk_size;
  }
  EXPECT_EQ(0, fclose(file));
}

void VerifyLoad(size_t length, size_t load_count = 1, size_t reading_thread_count = 4) {
  PathString path;
  CreateExternalDataFile(length, path);
  ScopedFileDeleter file_deleter{path};
  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.use_tf32 = false;
  provider_options.external_data_loader_reading_threads = reading_thread_count;
  auto execution_provider = CudaExecutionProviderWithOptions(&provider_options);
  ASSERT_NE(execution_provider, nullptr);
  auto loader = execution_provider->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  auto allocators = execution_provider->CreatePreferredAllocators();
  const auto allocator = std::find_if(allocators.begin(), allocators.end(), [](const AllocatorPtr& candidate) {
    return candidate->Info().device.Type() == OrtDevice::GPU &&
           candidate->Info().mem_type == OrtMemTypeDefault;
  });
  ASSERT_NE(allocator, allocators.end());
  Tensor tensor(DataTypeImpl::GetType<uint8_t>(), TensorShape({static_cast<int64_t>(length)}), *allocator);

  for (size_t load = 0; load < load_count; ++load) {
    ASSERT_STATUS_OK(loader->LoadTensor(Env::Default(), path, kFilePrefixSize, length, tensor));
  }

  std::vector<uint8_t> output(length);
  ASSERT_EQ(cudaSuccess, cudaMemcpy(output.data(), tensor.DataRaw(), length, cudaMemcpyDeviceToHost));
  for (size_t i = 0; i < length; ++i) {
    ASSERT_EQ(TestValue(i), output[i]) << "Mismatch at byte " << i;
  }
}

}  // namespace

TEST(CudaExternalDataLoaderTest, LoadsBelowParallelReadThreshold) {
  VerifyLoad(kParallelReadThreshold - 1);
}

TEST(CudaExternalDataLoaderTest, LoadsAtParallelReadThreshold) {
  VerifyLoad(kParallelReadThreshold);
}

TEST(CudaExternalDataLoaderTest, LoadsSynchronouslyWhenConfiguredWithOneReadingThread) {
  VerifyLoad(kParallelReadThreshold, 1, 1);
}

TEST(CudaExternalDataLoaderTest, DisablesLoaderWhenConfiguredWithZeroReadingThreads) {
  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.use_tf32 = false;
  provider_options.external_data_loader_reading_threads = 0;
  auto execution_provider = CudaExecutionProviderWithOptions(&provider_options);
  ASSERT_NE(execution_provider, nullptr);
  EXPECT_EQ(execution_provider->GetExternalDataLoader(), nullptr);
}

TEST(CudaExternalDataLoaderTest, RejectsTooManyReadingThreadsFromStructOptions) {
  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.external_data_loader_reading_threads =
      OrtCUDAProviderOptionsV2::kMaxExternalDataLoaderReadingThreadCount;
  Ort::SessionOptions valid_session_options;
  EXPECT_NO_THROW(valid_session_options.AppendExecutionProvider_CUDA_V2(provider_options));

  provider_options.external_data_loader_reading_threads =
      OrtCUDAProviderOptionsV2::kMaxExternalDataLoaderReadingThreadCount + 1;
  Ort::SessionOptions invalid_session_options;
  try {
    invalid_session_options.AppendExecutionProvider_CUDA_V2(provider_options);
    FAIL() << "Expected an invalid external_data_loader_reading_threads value to be rejected.";
  } catch (const Ort::Exception& ex) {
    EXPECT_THAT(ex.what(), testing::HasSubstr("external_data_loader_reading_threads"));
  }

  EXPECT_EQ(CudaExecutionProviderWithOptions(&provider_options), nullptr);
}

TEST(CudaExternalDataLoaderTest, ReusesAlternatingBuffersAcrossRepeatedLoads) {
  VerifyLoad(2 * kStagingBufferSize + 1, 2);
}

TEST(CudaExternalDataLoaderTest, RestoresCurrentDevice) {
  int device_count = 0;
  ASSERT_EQ(cudaSuccess, cudaGetDeviceCount(&device_count));
  if (device_count < 2) {
    GTEST_SKIP() << "Test requires at least two CUDA devices.";
  }

  constexpr int kLoaderDeviceId = 0;
  constexpr int kCallerDeviceId = 1;
  constexpr size_t kLength = 1024;
  PathString path;
  CreateExternalDataFile(kLength, path);
  ScopedFileDeleter file_deleter{path};
  auto execution_provider = DefaultCudaExecutionProvider();
  ASSERT_NE(execution_provider, nullptr);
  auto loader = execution_provider->GetExternalDataLoader();
  ASSERT_NE(loader, nullptr);
  auto allocators = execution_provider->CreatePreferredAllocators();
  const auto allocator = std::find_if(allocators.begin(), allocators.end(), [](const AllocatorPtr& candidate) {
    return candidate->Info().device.Type() == OrtDevice::GPU &&
           candidate->Info().mem_type == OrtMemTypeDefault;
  });
  ASSERT_NE(allocator, allocators.end());
  Tensor tensor(DataTypeImpl::GetType<uint8_t>(), TensorShape({static_cast<int64_t>(kLength)}), *allocator);

  ASSERT_EQ(cudaSuccess, cudaSetDevice(kCallerDeviceId));
  ASSERT_STATUS_OK(loader->LoadTensor(Env::Default(), path, kFilePrefixSize, kLength, tensor));

  int current_device = -1;
  ASSERT_EQ(cudaSuccess, cudaGetDevice(&current_device));
  EXPECT_EQ(kCallerDeviceId, current_device);
  ASSERT_EQ(cudaSuccess, cudaSetDevice(kLoaderDeviceId));
}
}  // namespace test
}  // namespace onnxruntime

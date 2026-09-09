// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "asserts.h"
#include "core/common/inlined_containers.h"
#include "core/framework/execution_provider.h"
#include "core/framework/session_state.h"
#include "core/framework/tensor.h"
#include "core/graph/onnx_protobuf.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "cuda_runtime.h"
#include "gtest/gtest.h"
#include "test/test_environment.h"
#include "test/unittest_util/framework_test_utils.h"
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

#if !defined(ORT_MINIMAL_BUILD)
TEST(CudaExternalDataLoaderTest, ExternalInitializerSessionMatchesWithLoaderEnabledAndDisabled) {
  constexpr int64_t kLength = kStagingBufferSize + 17;
  const InlinedVector<int64_t> indices{0, 1, kParallelReadThreshold - 1, kParallelReadThreshold,
                                       kParallelReadThreshold + 1, kStagingBufferSize - 1,
                                       kStagingBufferSize, kStagingBufferSize + 1, kLength - 1};
  const auto output_length = static_cast<int64_t>(indices.size());
  PathString data_path;
  ASSERT_NO_FATAL_FAILURE(CreateExternalDataFile(kLength, data_path));
  ScopedFileDeleter data_deleter{data_path};

  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::IR_VERSION);
  model.add_opset_import()->set_version(13);
  auto* graph = model.mutable_graph();
  graph->set_name("cuda_external_initializer");
  auto* weights = graph->add_initializer();
  weights->set_name("weights");
  weights->set_data_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  weights->add_dims(kLength);
  weights->set_data_location(ONNX_NAMESPACE::TensorProto_DataLocation_EXTERNAL);
  auto add_external_data = [&](const char* key, const std::string& value) {
    auto* entry = weights->add_external_data();
    entry->set_key(key);
    entry->set_value(value);
  };
  add_external_data("location", ToUTF8String(std::filesystem::path(data_path).filename().native()));
  add_external_data("offset", std::to_string(kFilePrefixSize));
  add_external_data("length", std::to_string(kLength));

  auto* input = graph->add_input();
  input->set_name("indices");
  auto* input_type = input->mutable_type()->mutable_tensor_type();
  input_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_INT64);
  input_type->mutable_shape()->add_dim()->set_dim_value(output_length);
  auto* output = graph->add_output();
  output->set_name("output");
  auto* output_type = output->mutable_type()->mutable_tensor_type();
  output_type->set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_UINT8);
  output_type->mutable_shape()->add_dim()->set_dim_value(output_length);
  auto* gather = graph->add_node();
  gather->set_op_type("Gather");
  gather->add_input("weights");
  gather->add_input("indices");
  gather->add_output("output");

  PathString model_path = ORT_TSTR("cuda_external_initializer_model_XXXXXX");
  FILE* model_file = nullptr;
  ASSERT_NO_FATAL_FAILURE(CreateTestFile(model_file, model_path));
  ScopedFileDeleter model_deleter{model_path};
  std::unique_ptr<FILE, int (*)(FILE*)> model_file_owner(model_file, fclose);
  const auto model_bytes = model.SerializeAsString();
  ASSERT_EQ(model_bytes.size(), fwrite(model_bytes.data(), 1, model_bytes.size(), model_file));
  ASSERT_EQ(0, fclose(model_file_owner.release()));

  OrtValue indices_value;
  CreateMLValue<int64_t>(std::make_shared<CPUAllocator>(), {output_length},
                         gsl::span<const int64_t>(indices), &indices_value);
  const InlinedVector<std::string> output_names{"output"};
  InlinedVector<uint8_t> baseline;
  for (const size_t reader_count : {0, 4}) {
    SCOPED_TRACE(reader_count);
    SessionOptions session_options;
    session_options.graph_optimization_level = TransformerLevel::Default;
    ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));
    InferenceSession session(session_options, GetEnvironment());
    OrtCUDAProviderOptionsV2 provider_options{};
    provider_options.do_copy_in_default_stream = true;
    provider_options.external_data_loader_reading_threads = reader_count;
    auto execution_provider = CudaExecutionProviderWithOptions(&provider_options);
    ASSERT_NE(execution_provider, nullptr);
    ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(execution_provider)));
    ASSERT_STATUS_OK(session.Load(model_path));
    ASSERT_STATUS_OK(session.Initialize());

    const auto& session_state = session.GetSessionState();
    int weights_index = -1;
    ASSERT_STATUS_OK(session_state.GetOrtValueNameIdxMap().GetIdx("weights", weights_index));
    const auto& initialized_weights = session_state.GetInitializedTensors().at(weights_index).Get<Tensor>();
    ASSERT_EQ(initialized_weights.Location().device.Type(), OrtDevice::GPU);
    const auto* loader =
        session.GetExternalDataLoaderManager().GetExternalDataLoader(initialized_weights.Location());
    ASSERT_EQ(loader != nullptr, reader_count != 0);

    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, {{"indices", indices_value}}, output_names, &fetches));
    ASSERT_EQ(fetches.size(), 1U);
    const auto& result = fetches[0].Get<Tensor>();
    ASSERT_EQ(result.Shape(), TensorShape({output_length}));
    ASSERT_EQ(result.Location().device.Type(), OrtDevice::CPU);
    const auto values = result.DataAsSpan<uint8_t>();
    for (size_t i = 0; i < indices.size(); ++i) {
      ASSERT_EQ(values[i], TestValue(static_cast<size_t>(indices[i]))) << "Index " << indices[i];
    }
    if (reader_count == 0) {
      baseline.assign(values.begin(), values.end());
    } else {
      EXPECT_TRUE(std::equal(baseline.begin(), baseline.end(), values.begin(), values.end()));
    }
  }
}
#endif

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

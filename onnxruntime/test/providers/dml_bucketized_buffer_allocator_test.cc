// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_DML

#include "gtest/gtest.h"

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config_options.h"
#include "core/framework/execution_provider.h"
#include "core/graph/onnx_protobuf.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"
#include "test/util/include/test_utils.h"

#include <wrl/client.h>
#include <wrl/implements.h>
#include <wil/wrl.h>
#include <wil/result.h>
#include <gsl/gsl>
#include "directx/d3d12.h"
#include <DirectML.h>

using Microsoft::WRL::ComPtr;

#include "core/providers/dml/DmlExecutionProvider/src/ErrorHandling.h"
#include "core/providers/dml/DmlExecutionProvider/src/GraphicsUnknownHelper.h"
#include "core/providers/dml/DmlExecutionProvider/inc/DmlExecutionProvider.h"
#include "core/providers/dml/DmlExecutionProvider/src/BucketizedBufferAllocator.h"
#include "core/providers/dml/dml_provider_factory_creator.h"
#include "core/providers/dml/dml_session_options_config_keys.h"

namespace onnxruntime {
namespace test {

namespace {

// The size of the smallest bucket in Dml::BucketizedBufferAllocator's pool.
constexpr size_t kMinBucketSize = size_t{1} << 16;

// Returns nullptr if no DirectML device is available. Each EP instance creates its own buffer allocator, whose pool
// starts out empty.
std::unique_ptr<IExecutionProvider> CreateDmlExecutionProvider(bool disable_memory_arena) {
  ConfigOptions config_options{};
  if (disable_memory_arena) {
    ORT_THROW_IF_ERROR(config_options.AddConfigEntry(kOrtSessionOptionsConfigDisableMemoryArena, "1"));
  }

  if (auto factory = DMLProviderFactoryCreator::CreateFromDeviceOptions(config_options, nullptr, false, false)) {
    return factory->CreateProvider();
  }

  return nullptr;
}

Dml::BucketizedBufferAllocator& GetBufferAllocator(const AllocatorPtr& allocator) {
  const OrtDevice& device = allocator->Info().device;
  ORT_ENFORCE(device.Type() == OrtDevice::GPU && device.Vendor() == OrtDevice::VendorIds::MICROSOFT,
              "Expected the DML device allocator.");
  return static_cast<Dml::BucketizedBufferAllocator&>(*allocator);
}

uint64_t GetResourceId(Dml::BucketizedBufferAllocator& allocator, void* allocation) {
  return allocator.DecodeDataHandle(allocation)->GetPooledResourceId();
}

// Builds a single-node model: Y = Relu(X), where X and Y are FLOAT16[element_count].
std::string BuildReluModel(int64_t element_count) {
  ONNX_NAMESPACE::ModelProto model;
  model.set_ir_version(ONNX_NAMESPACE::Version::IR_VERSION);
  ONNX_NAMESPACE::OperatorSetIdProto& opset = *model.add_opset_import();
  opset.set_domain("");
  opset.set_version(14);

  ONNX_NAMESPACE::GraphProto& graph = *model.mutable_graph();
  graph.set_name("dml_bucketized_buffer_allocator_test");

  for (auto* value_info : {graph.add_input(), graph.add_output()}) {
    ONNX_NAMESPACE::TypeProto_Tensor& tensor_type = *value_info->mutable_type()->mutable_tensor_type();
    tensor_type.set_elem_type(ONNX_NAMESPACE::TensorProto_DataType_FLOAT16);
    tensor_type.mutable_shape()->add_dim()->set_dim_value(element_count);
  }
  graph.mutable_input(0)->set_name("X");
  graph.mutable_output(0)->set_name("Y");

  ONNX_NAMESPACE::NodeProto& node = *graph.add_node();
  node.set_name("relu0");
  node.set_op_type("Relu");
  node.add_input("X");
  node.add_output("Y");

  std::string serialized;
  EXPECT_TRUE(model.SerializeToString(&serialized));
  return serialized;
}

}  // namespace

// The bucket for an allocation must be the smallest one that can hold it. Bucket indices used to be computed with
// floating-point log2, which rounds sizes slightly above 2^n (n >= 49) down to n and so chose a bucket that was too
// small.
TEST(DmlBucketizedBufferAllocatorTest, BucketIsSmallestThatHoldsSize) {
  const auto get_bucket_size = [](uint64_t size) {
    return Dml::BucketizedBufferAllocator::GetBucketSizeFromIndex(
        Dml::BucketizedBufferAllocator::GetBucketIndexFromSize(size));
  };

  EXPECT_EQ(get_bucket_size(1), kMinBucketSize);

  for (uint32_t exponent = 16; exponent < 63; ++exponent) {
    const uint64_t power_of_two = uint64_t{1} << exponent;
    for (uint64_t size : {power_of_two - 1, power_of_two, power_of_two + 1}) {
      SCOPED_TRACE(size);
      const uint64_t bucket_size = get_bucket_size(size);
      EXPECT_GE(bucket_size, size);
      EXPECT_TRUE(bucket_size == kMinBucketSize || bucket_size / 2 < size);
    }
  }

  constexpr uint64_t kLargestBucketSize = uint64_t{1} << 63;
  EXPECT_EQ(get_bucket_size(kLargestBucketSize - 1), kLargestBucketSize);
  EXPECT_EQ(get_bucket_size(kLargestBucketSize), kLargestBucketSize);
}

// Allocations made while rounding is disabled get a resource whose size is rounded up to a multiple of 4 bytes, so
// for sizes 1-3 bytes below a bucket size the resource is exactly bucket-sized. Alloc used to treat such allocations
// as unpooled (it compared the requested size against the bucket size) while FreeResource returned them to the pool
// (it compared the resource size), indexing a bucket that Alloc never created. Here the pool is empty, so that write
// went through m_pool[0] of an empty vector.
TEST(DmlBucketizedBufferAllocatorTest, FreeUnroundedAllocationJustBelowBucketSizeIntoEmptyPool) {
  for (size_t bytes_below_bucket_size : {size_t{1}, size_t{2}, size_t{3}}) {
    SCOPED_TRACE(bytes_below_bucket_size);

    std::unique_ptr<IExecutionProvider> dml_ep = CreateDmlExecutionProvider(/*disable_memory_arena*/ false);
    if (dml_ep == nullptr) {
      GTEST_SKIP() << "DirectML execution provider is not available on this machine.";
    }

    const AllocatorPtr allocator_ptr = dml_ep->CreatePreferredAllocators()[0];
    Dml::BucketizedBufferAllocator& allocator = GetBufferAllocator(allocator_ptr);

    void* allocation = allocator.Alloc(kMinBucketSize - bytes_below_bucket_size, AllocatorRoundingMode::Disabled);
    EXPECT_EQ(allocator.DecodeDataHandle(allocation)->GetResource()->GetDesc().Width, kMinBucketSize);
    const uint64_t resource_id = GetResourceId(allocator, allocation);
    allocator.Free(allocation);

    // The resource is bucket-sized, so it is pooled and reused by the next allocation from that bucket.
    void* reused = allocator.Alloc(kMinBucketSize, AllocatorRoundingMode::Enabled);
    EXPECT_EQ(GetResourceId(allocator, reused), resource_id);
    allocator.Free(reused);
  }
}

// Mirrors the lifetime of an initializer in a session using the default configuration: it is allocated during session
// initialization while rounding is still disabled, rounding is enabled once initialization ends, a later (rounded)
// allocation grows the pool to hold only its first bucket, and the initializer is freed when the session is destroyed.
// Freeing the initializer used to write to m_pool[4] of a pool with a single bucket.
TEST(DmlBucketizedBufferAllocatorTest, FreeUnroundedAllocationJustBelowBucketSizeAfterPoolGrows) {
  constexpr size_t kBucketSize = size_t{1} << 20;

  for (size_t bytes_below_bucket_size : {size_t{1}, size_t{2}, size_t{3}}) {
    SCOPED_TRACE(bytes_below_bucket_size);

    std::unique_ptr<IExecutionProvider> dml_ep = CreateDmlExecutionProvider(/*disable_memory_arena*/ false);
    if (dml_ep == nullptr) {
      GTEST_SKIP() << "DirectML execution provider is not available on this machine.";
    }

    const AllocatorPtr allocator_ptr = dml_ep->CreatePreferredAllocators()[0];
    Dml::BucketizedBufferAllocator& allocator = GetBufferAllocator(allocator_ptr);

    void* initializer = allocator.Alloc(kBucketSize - bytes_below_bucket_size);
    const uint64_t initializer_resource_id = GetResourceId(allocator, initializer);

    ASSERT_STATUS_OK(dml_ep->OnSessionInitializationEnd());

    void* transient = allocator.Alloc(1);
    allocator.Free(transient);

    allocator.Free(initializer);

    void* reused = allocator.Alloc(kBucketSize);
    EXPECT_EQ(GetResourceId(allocator, reused), initializer_resource_id);
    allocator.Free(reused);
  }
}

#ifndef ORT_NO_EXCEPTIONS
// The largest bucket is 2^63 bytes. Rounding a larger size up to a bucket size used to overflow the bucket size
// computation, producing a 1-byte resource for an allocation of nearly 2^64 bytes.
TEST(DmlBucketizedBufferAllocatorTest, RejectsAllocationLargerThanLargestBucket) {
  if (sizeof(size_t) <= sizeof(uint32_t)) {
    GTEST_SKIP() << "This regression needs a size_t wider than uint32_t.";
  }

  std::unique_ptr<IExecutionProvider> dml_ep = CreateDmlExecutionProvider(/*disable_memory_arena*/ false);
  if (dml_ep == nullptr) {
    GTEST_SKIP() << "DirectML execution provider is not available on this machine.";
  }

  const AllocatorPtr allocator_ptr = dml_ep->CreatePreferredAllocators()[0];
  Dml::BucketizedBufferAllocator& allocator = GetBufferAllocator(allocator_ptr);

  EXPECT_ANY_THROW((void)allocator.Alloc(std::numeric_limits<size_t>::max(), AllocatorRoundingMode::Enabled));
}
#endif

// With the memory arena disabled, rounding stays disabled while running a model, so any tensor 1-3 bytes smaller than
// a bucket takes the same path. Here the copy of the input to the GPU and the output are both 2 bytes below 1MB.
TEST(DmlBucketizedBufferAllocatorTest, RunWithMemoryArenaDisabledAndTensorsJustBelowBucketSize) {
  std::unique_ptr<IExecutionProvider> dml_ep = CreateDmlExecutionProvider(/*disable_memory_arena*/ true);
  if (dml_ep == nullptr) {
    GTEST_SKIP() << "DirectML execution provider is not available on this machine.";
  }

  constexpr int64_t kElementCount = (int64_t{1} << 19) - 1;
  const std::string model = BuildReluModel(kElementCount);

  SessionOptions session_options;
  session_options.session_logid = "DmlBucketizedBufferAllocatorTest";
  ASSERT_STATUS_OK(session_options.config_options.AddConfigEntry(kOrtSessionOptionsDisableCPUEPFallback, "1"));

  InferenceSession session{session_options, GetEnvironment()};
  ASSERT_STATUS_OK(session.RegisterExecutionProvider(std::move(dml_ep)));
  ASSERT_STATUS_OK(session.Load(model.data(), static_cast<int>(model.size())));
  ASSERT_STATUS_OK(session.Initialize());

  std::vector<MLFloat16> input;
  std::vector<float> expected;
  input.reserve(static_cast<size_t>(kElementCount));
  expected.reserve(static_cast<size_t>(kElementCount));
  for (int64_t i = 0; i < kElementCount; ++i) {
    const float value = static_cast<float>(i % 7) - 3.0f;
    input.emplace_back(value);
    expected.push_back(value > 0.0f ? value : 0.0f);
  }

  const std::vector<int64_t> dims = {kElementCount};
  NameMLValMap feeds;
  feeds.emplace("X", CreateInputOrtValueOnCPU<MLFloat16>(dims, input));

  // The second run reuses any resources the first run returned to the pool.
  for (int run = 0; run < 2; ++run) {
    SCOPED_TRACE(run);

    std::vector<OrtValue> fetches;
    ASSERT_STATUS_OK(session.Run(RunOptions{}, feeds, std::vector<std::string>{"Y"}, &fetches));
    ASSERT_EQ(fetches.size(), 1u);

    const auto output = fetches[0].Get<Tensor>().DataAsSpan<MLFloat16>();
    ASSERT_EQ(output.size(), expected.size());
    for (size_t i = 0; i < output.size(); ++i) {
      ASSERT_EQ(output[i].ToFloat(), expected[i]) << "at index " << i;
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // USE_DML

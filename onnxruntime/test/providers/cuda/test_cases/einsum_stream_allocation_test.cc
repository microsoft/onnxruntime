// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Regression tests for the intermediate tensors that Einsum's CUDA device helpers allocate.
//
// Those intermediates are filled by work queued on the run's stream, but they are released as soon
// as they go out of scope in Compute() - while that work may still be in flight. A stream aware
// arena can only keep such a buffer away from another stream if the allocation was tagged with the
// stream that uses it, which means the helpers have to allocate through IAllocator::AllocOnStream
// rather than the plain Alloc(). These tests wrap the device allocator in a recorder and assert
// that every intermediate is tagged with the stream in the Einsum assets, so a helper that goes
// back to an untagged allocation is caught here.
//
// This is a provider-world translation unit: it reaches into CUDA EP internals through the shared
// provider bridge, so it must not include the core framework headers.

#include "gtest/gtest.h"

#include <memory>
#include <vector>

#include "core/providers/shared_library/provider_api.h"
#include "core/framework/stream_handles.h"
#include "core/providers/cuda/cuda_allocator.h"
#include "core/providers/cuda/math/einsum_utils/einsum_auxiliary_ops.h"

namespace onnxruntime {
namespace test {

namespace {

// Forwards to a real device allocator and records which entry point each allocation came in on.
class StreamRecordingAllocator : public IAllocator {
 public:
  explicit StreamRecordingAllocator(AllocatorPtr inner)
      : IAllocator(inner->Info()), inner_(std::move(inner)) {}

  bool IsStreamAware() const override { return true; }

  void* Alloc(size_t size) override {
    ++untagged_allocs;
    return inner_->Alloc(size);
  }

  void* AllocOnStream(size_t size, Stream* stream) override {
    ++stream_allocs;
    last_stream = stream;
    return inner_->AllocOnStream(size, stream);
  }

  void Free(void* p) override { inner_->Free(p); }

  int untagged_allocs = 0;
  int stream_allocs = 0;
  Stream* last_stream = nullptr;

 private:
  AllocatorPtr inner_;
};

}  // namespace

class EinsumCudaIntermediateTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available";
    }

    CUDA_CALL_THROW(cudaSetDevice(0));
    CUDA_CALL_THROW(cudaGetDeviceProperties(&device_prop_, 0));
    CUDA_CALL_THROW(cudaStreamCreate(&cuda_stream_));

    device_allocator_ = std::make_shared<CUDAAllocator>(0, CUDA);
    recording_allocator_ = std::make_shared<StreamRecordingAllocator>(device_allocator_);

    device_ = device_allocator_->Info().device;  // Stream holds a reference to this
    stream_ = std::make_unique<Stream>(cuda_stream_, device_);

    assets_ = MakeAssets(stream_.get());
  }

  // The allocation stream is tracked separately from the stream the work is queued on, because a
  // plugin build hosted by an ORT that cannot hand out its framework stream has no stream to tag
  // allocations with. Passing null for `alloc_stream` models that host.
  //
  // The cuBLAS and cuDNN handles are left null: none of the helpers exercised here use them.
  std::unique_ptr<EinsumOp::EinsumCudaAssets> MakeAssets(Stream* alloc_stream) {
    return std::make_unique<EinsumOp::EinsumCudaAssets>(stream_.get(), alloc_stream, device_prop_,
                                                        nullptr, nullptr, device_allocator_, false);
  }

  void TearDown() override {
    assets_.reset();
    stream_.reset();
    if (cuda_stream_ != nullptr) {
      cudaStreamDestroy(cuda_stream_);
      cuda_stream_ = nullptr;
    }
  }

  // Uploads `values` into a fresh device tensor of the given shape.
  std::unique_ptr<Tensor> DeviceTensor(const TensorShape& shape, const std::vector<float>& values) {
    auto tensor = Tensor::Create(DataTypeImpl::GetType<float>(), shape, device_allocator_);
    CUDA_CALL_THROW(cudaMemcpy(tensor->MutableDataRaw(), values.data(), values.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
    return tensor;
  }

  std::vector<float> ReadBack(const Tensor& tensor) {
    std::vector<float> values(static_cast<size_t>(tensor.Shape().Size()));
    CUDA_CALL_THROW(cudaStreamSynchronize(cuda_stream_));
    CUDA_CALL_THROW(cudaMemcpy(values.data(), tensor.DataRaw(), values.size() * sizeof(float),
                               cudaMemcpyDeviceToHost));
    return values;
  }

  cudaDeviceProp device_prop_{};
  cudaStream_t cuda_stream_ = nullptr;
  OrtDevice device_;
  AllocatorPtr device_allocator_;
  std::shared_ptr<StreamRecordingAllocator> recording_allocator_;
  std::unique_ptr<Stream> stream_;
  std::unique_ptr<EinsumOp::EinsumCudaAssets> assets_;
};

// The helper every Transpose and MatMul intermediate goes through.
TEST_F(EinsumCudaIntermediateTest, CreateTensorAllocatesOnTheRunStream) {
  auto tensor = EinsumOp::DeviceHelpers::CudaDeviceHelpers::CreateTensor(
      DataTypeImpl::GetType<float>(), TensorShape({2, 3}), recording_allocator_, assets_.get());

  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(recording_allocator_->stream_allocs, 1);
  EXPECT_EQ(recording_allocator_->untagged_allocs, 0);
  EXPECT_EQ(recording_allocator_->last_stream, stream_.get());
}

// With no stream to tag allocations with, the intermediate has to fall back to the plain
// allocation rather than tagging it with the stream the work is queued on - in a plugin build that
// stream can be a shim that a stream aware arena must never be handed.
TEST_F(EinsumCudaIntermediateTest, CreateTensorAllocatesUntaggedWithoutAnAllocationStream) {
  auto assets = MakeAssets(/*alloc_stream*/ nullptr);

  auto tensor = EinsumOp::DeviceHelpers::CudaDeviceHelpers::CreateTensor(
      DataTypeImpl::GetType<float>(), TensorShape({2, 3}), recording_allocator_, assets.get());

  ASSERT_NE(tensor, nullptr);
  EXPECT_EQ(recording_allocator_->stream_allocs, 0);
  EXPECT_EQ(recording_allocator_->untagged_allocs, 1);
}

// Diagonal allocates its output itself instead of going through CreateTensor.
TEST_F(EinsumCudaIntermediateTest, DiagonalAllocatesOnTheRunStream) {
  auto input = DeviceTensor(TensorShape({3, 3}), {0.f, 1.f, 2.f,
                                                  3.f, 4.f, 5.f,
                                                  6.f, 7.f, 8.f});

  auto output = EinsumOp::DeviceHelpers::CudaDeviceHelpers::Diagonal(*input, 0, 1, recording_allocator_,
                                                                     assets_.get());

  ASSERT_NE(output, nullptr);
  EXPECT_EQ(recording_allocator_->stream_allocs, 1);
  EXPECT_EQ(recording_allocator_->untagged_allocs, 0);
  EXPECT_EQ(recording_allocator_->last_stream, stream_.get());

  EXPECT_EQ(output->Shape(), TensorShape({3}));
  EXPECT_EQ(ReadBack(*output), std::vector<float>({0.f, 4.f, 8.f}));
}

// The reduce path allocates inside cuda::ReductionOps::ReduceCompute. The reduction itself is a
// matrix reduction over the trailing axis, so it stays on the fast path and needs no cuDNN handle.
TEST_F(EinsumCudaIntermediateTest, ReduceSumAllocatesOnTheRunStream) {
  auto input = DeviceTensor(TensorShape({2, 3}), {0.f, 1.f, 2.f,
                                                  3.f, 4.f, 5.f});

  const std::vector<int64_t> reduce_axes{1};
  auto output = EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<float>(
      *input, reduce_axes, /*keep_dims*/ true, recording_allocator_, /*input_shape_override*/ nullptr,
      /*tp*/ nullptr, assets_.get());

  ASSERT_NE(output, nullptr);
  EXPECT_EQ(recording_allocator_->stream_allocs, 1);
  EXPECT_EQ(recording_allocator_->untagged_allocs, 0);
  EXPECT_EQ(recording_allocator_->last_stream, stream_.get());

  EXPECT_EQ(output->Shape(), TensorShape({2, 1}));
  EXPECT_EQ(ReadBack(*output), std::vector<float>({3.f, 12.f}));
}

}  // namespace test
}  // namespace onnxruntime

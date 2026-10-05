// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_WEBGPU) && !defined(ORT_USE_EP_API_ADAPTERS) && !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)

#include <array>
#include <latch>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "core/framework/op_kernel.h"
#include "core/framework/ort_value_name_idx_map.h"
#include "core/graph/model.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "test/test_environment.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

class UnusedKernel final : public OpKernel {
 public:
  explicit UnusedKernel(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext*) const override { return Status::OK(); }
};

class UnusedKernelContext final : public OpKernelContext {
 public:
  UnusedKernelContext() : OpKernelContext(nullptr, DefaultLoggingManager().DefaultLogger(), nullptr) {}
};

TEST(WebGpuContextTest, FillZeroClearsSessionBuffer) {
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = webgpu_ep.Recording();

  Model model("fill_zero", false, DefaultLoggingManager().DefaultLogger());
  auto& node = model.MainGraph().AddNode("unused", "Identity", "", {}, {});
  auto kernel_def = KernelDefBuilder().SetName("Identity").Provider(kWebGpuExecutionProvider).SinceVersion(1).Build();
  const std::unordered_map<int, OrtValue> initializers;
  const OrtValueNameIdxMap values;
  const DataTransferManager transfers;
  const AllocatorMap allocators;
  OpKernelInfo info(node, *kernel_def, *ep, initializers, values, transfers, allocators, options);
  UnusedKernel kernel(info);
  UnusedKernelContext kernel_context;
  webgpu::ComputeContext compute_context(context, webgpu_ep, kernel, kernel_context);

  wgpu::BufferDescriptor desc{};
  desc.size = 64;
  desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto buffer = context.Device().CreateBuffer(&desc);
  Tensor tensor(DataTypeImpl::GetType<uint32_t>(), TensorShape{16}, buffer.Get(),
                OrtMemoryInfo(WEBGPU_BUFFER, OrtDeviceAllocator, webgpu::WebGpuDevice, OrtMemTypeDefault));

  std::array<uint32_t, 16> data;
  data.fill(42);
  webgpu_ep.BufferManager().Upload(recording, data.data(), buffer.Get(), sizeof(data));
  compute_context.FillZero(tensor);
  webgpu_ep.BufferManager().Download(recording, buffer.Get(), data.data(), sizeof(data));
  for (auto value : data) {
    EXPECT_EQ(value, 0u);
  }
}

TEST(WebGpuContextTest, SharedDeviceErrorScopesRemainThreadLocal) {
  ConfigOptions options;
  auto first_ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto second_ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  ASSERT_NE(first_ep.get(), second_ep.get());
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  const auto& device = context.Device();
  struct Result {
    wgpu::PopErrorScopeStatus status{};
    wgpu::ErrorType type{};
    std::string message;
    Status wait_status;
  };
  std::array<Result, 2> results;
  std::latch first_pushed{1};
  std::latch second_pushed{1};
  std::latch first_error{1};
  std::latch second_popped{1};

  const auto pop = [&](size_t index) {
    auto future = device.PopErrorScope(
        wgpu::CallbackMode::WaitAnyOnly,
        [](wgpu::PopErrorScopeStatus status, wgpu::ErrorType type, wgpu::StringView message, Result* result) {
          result->status = status;
          result->type = type;
          result->message.assign(message.data, message.length);
        },
        &results[index]);
    results[index].wait_status = context.Wait(future);
  };
  std::thread first([&] {
    device.PushErrorScope(wgpu::ErrorFilter::Validation);
    first_pushed.count_down();
    second_pushed.wait();
    device.InjectError(wgpu::ErrorType::Validation, "first-session-error");
    first_error.count_down();
    second_popped.wait();
    pop(0);
  });
  std::thread second([&] {
    first_pushed.wait();
    device.PushErrorScope(wgpu::ErrorFilter::Validation);
    second_pushed.count_down();
    first_error.wait();
    device.InjectError(wgpu::ErrorType::Validation, "second-session-error");
    pop(1);
    second_popped.count_down();
  });
  first.join();
  second.join();

  for (const auto& result : results) {
    ASSERT_STATUS_OK(result.wait_status);
    EXPECT_EQ(result.status, wgpu::PopErrorScopeStatus::Success);
    EXPECT_EQ(result.type, wgpu::ErrorType::Validation);
  }
  EXPECT_EQ(results[0].message.substr(0, results[0].message.find('\n')), "first-session-error");
  EXPECT_EQ(results[1].message.substr(0, results[1].message.find('\n')), "second-session-error");
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
#endif

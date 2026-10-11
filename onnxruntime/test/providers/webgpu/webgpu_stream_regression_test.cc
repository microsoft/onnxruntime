// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if defined(USE_WEBGPU) && !defined(__wasm__) && !defined(USE_EXTERNAL_DAWN)

#include <array>
#include <latch>
#include <string>
#include <thread>

#include "gtest/gtest.h"
#include "test/providers/webgpu/internal_test_kernel.h"
#if !defined(ORT_USE_EP_API_ADAPTERS)
#include "test/test_environment.h"
#endif
#include "core/framework/op_kernel.h"
#include "core/providers/webgpu/allocator.h"
#include "core/providers/webgpu/compute_context.h"
#include "core/providers/webgpu/webgpu_provider_factory_creator.h"
#include "test/util/include/asserts.h"

namespace onnxruntime {
namespace test {
namespace {

TEST(WebGpuContextTest, FillZeroClearsSessionBuffer) {
  WebGpuTestKernel test_kernel;
  auto& webgpu_ep = test_kernel.Ep();
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = webgpu_ep.Recording();
#if defined(ORT_USE_EP_API_ADAPTERS)
  WebGpuTestOpKernelContext kernel_context(test_kernel.Context(), test_kernel.Kernel());
#else
  class UnusedKernelContext final : public OpKernelContext {
   public:
    UnusedKernelContext() : OpKernelContext(nullptr, DefaultLoggingManager().DefaultLogger(), nullptr) {}
  } kernel_context;
#endif
  webgpu::ComputeContext compute_context(context, webgpu_ep, test_kernel.Kernel(), kernel_context);

  wgpu::BufferDescriptor desc{};
  desc.size = 64;
  desc.usage = wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
  auto buffer = context.Device().CreateBuffer(&desc);
  Tensor tensor(DataTypeImpl::GetType<uint32_t>(), TensorShape{16}, buffer.Get(),
                OrtMemoryInfo(WEBGPU_BUFFER, OrtDeviceAllocator, webgpu::WebGpuDevice(0), OrtMemTypeDefault));

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

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

#if !defined(BUILD_DAWN_SHARED_LIBRARY)
#include "dawn/dawn_proc.h"
#endif
#include "dawn/native/DawnNative.h"

namespace onnxruntime {
namespace test {
namespace {

#if !defined(BUILD_DAWN_SHARED_LIBRARY)
class UnusedKernel final : public OpKernel {
 public:
  explicit UnusedKernel(const OpKernelInfo& info) : OpKernel(info) {}
  Status Compute(OpKernelContext*) const override { return Status::OK(); }
};

class UnusedKernelContext final : public OpKernelContext {
 public:
  UnusedKernelContext() : OpKernelContext(nullptr, DefaultLoggingManager().DefaultLogger(), nullptr) {}
};

thread_local webgpu::CommandRecordingState* observed_recording = nullptr;
thread_local bool* observed_exclusion = nullptr;
thread_local size_t* observed_clears = nullptr;

void ObserveClearBuffer(WGPUCommandEncoder encoder, WGPUBuffer buffer, uint64_t offset, uint64_t size) {
  if (observed_recording != nullptr) {
    // Probe from another thread while the real FillZero is inside ClearBuffer. Joining the
    // nonblocking probe makes this independent of sleeps or which thread is scheduled first.
    auto* recording = observed_recording;
    bool acquired = false;
    std::thread contender([&] {
      acquired = recording->mutex.try_lock();
      if (acquired) {
        recording->mutex.unlock();
      }
    });
    contender.join();
    *observed_exclusion = !acquired;
    ++*observed_clears;
  }
  dawn::native::GetProcs().commandEncoderClearBuffer(encoder, buffer, offset, size);
}
#endif

TEST(WebGpuContextTest, FillZeroExcludesConcurrentSessionAllocator) {
#if defined(BUILD_DAWN_SHARED_LIBRARY)
  GTEST_SKIP() << "Shared Dawn calls bypass the replaceable proc table required by this mutex probe.";
#else
  ConfigOptions options;
  auto ep = WebGpuProviderFactoryCreator::Create(options)->CreateProvider();
  auto& webgpu_ep = static_cast<WebGpuExecutionProvider&>(*ep);
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);
  auto& recording = webgpu_ep.Recording();

  Model model("fill_zero_lock", false, DefaultLoggingManager().DefaultLogger());
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

  auto procs = dawn::native::GetProcs();
  procs.commandEncoderClearBuffer = ObserveClearBuffer;
  bool excluded = false;
  size_t clears = 0;
  observed_recording = &recording;
  observed_exclusion = &excluded;
  observed_clears = &clears;
  struct RestoreProcs {
    ~RestoreProcs() {
      observed_recording = nullptr;
      observed_exclusion = nullptr;
      observed_clears = nullptr;
      dawnProcSetProcs(&dawn::native::GetProcs());
    }
  } restore;
  dawnProcSetProcs(&procs);
  compute_context.FillZero(tensor);

  EXPECT_EQ(clears, 1u);
  EXPECT_TRUE(excluded) << "FillZero encoded a clear without holding the Session recording mutex";
  ASSERT_STATUS_OK(context.Flush(webgpu_ep.BufferManager(), recording));
#endif
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

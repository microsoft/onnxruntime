// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <algorithm>
#include <array>
#include <atomic>
#include <barrier>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <thread>
#include <vector>
#include "gtest/gtest.h"
#include "core/common/path_string.h"
#include "core/providers/webgpu/data_transfer.h"
#include "core/providers/webgpu/webgpu_context.h"
#include "core/providers/webgpu/program_manager.h"
#include "test/providers/webgpu/internal_test_kernel.h"
#include "test/util/include/temp_dir.h"

namespace onnxruntime::test {
namespace {
class ErrorSink {
 public:
  void Record(const std::string& message) {
    bool expected = false;
    if (failed_.compare_exchange_strong(expected, true)) {
      std::lock_guard<std::mutex> lock(mutex_);
      first_error_ = message;
    }
  }

  bool Failed() const {
    return failed_.load();
  }

  std::string FirstError() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return first_error_;
  }

 private:
  std::atomic<bool> failed_{false};
  mutable std::mutex mutex_;
  std::string first_error_;
};
TEST(WebGpuConcurrentContextTestStandalone, ShaderDumpWritesRemainComplete) {
  TemporaryDirectory temp_dir{ORT_TSTR("webgpu_shader_dump_test")};
  const auto dump_path = std::filesystem::path(temp_dir.Path()) / ORT_TSTR("shaders.txt");
  constexpr int kThreads = 8;
  constexpr int kRecordsPerThread = 100;
  constexpr size_t kPayloadSize = 4096;

  {
    auto dump_shader = webgpu::detail::CreateShaderDumpFunction(ToUTF8String(dump_path.native()));
    std::barrier start{kThreads};
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
      threads.emplace_back([&, thread_id] {
        start.arrive_and_wait();
        for (int record_id = 0; record_id < kRecordsPerThread; ++record_id) {
          const std::string marker = "record-" + std::to_string(thread_id) + "-" + std::to_string(record_id);
          dump_shader(marker + "-start\n" + std::string(kPayloadSize, static_cast<char>('a' + thread_id)) +
                      "\n" + marker + "-end");
        }
      });
    }
    for (auto& thread : threads) {
      thread.join();
    }
  }

  std::ifstream dump_file(dump_path);
  ASSERT_TRUE(dump_file.is_open());
  const std::string contents{std::istreambuf_iterator<char>{dump_file}, std::istreambuf_iterator<char>{}};
  for (int thread_id = 0; thread_id < kThreads; ++thread_id) {
    for (int record_id = 0; record_id < kRecordsPerThread; ++record_id) {
      const std::string marker = "record-" + std::to_string(thread_id) + "-" + std::to_string(record_id);
      const std::string record = marker + "-start\n" +
                                 std::string(kPayloadSize, static_cast<char>('a' + thread_id)) +
                                 "\n" + marker + "-end\n";
      const size_t position = contents.find(record);
      ASSERT_NE(position, std::string::npos) << "Missing or corrupted shader dump record " << marker;
      EXPECT_EQ(contents.find(record, position + 1), std::string::npos)
          << "Duplicate shader dump record " << marker;
    }
  }
}
#if defined(ORT_USE_EP_API_ADAPTERS)
TEST(WebGpuConcurrentContextTest, IndependentDataTransfersMultiThreadCopy) {
  WebGpuTestKernel keepalive;
  constexpr int kThreads = 4;
  constexpr int kIters = 30;
  constexpr size_t kElements = 4096;
  auto& context = webgpu::WebGpuContextFactory::GetContext(0);

  std::array<webgpu::CommandRecordingState, kThreads> recordings;
  std::array<std::unique_ptr<webgpu::DataTransferImpl>, kThreads> data_transfers;
  std::array<wgpu::Buffer, kThreads> gpu_buffers;
  for (int t = 0; t < kThreads; ++t) {
    data_transfers[t] = std::make_unique<webgpu::DataTransferImpl>(context.BufferManager(), recordings[t]);
    wgpu::BufferDescriptor desc{};
    desc.size = kElements * sizeof(float);
    desc.usage = wgpu::BufferUsage::Storage | wgpu::BufferUsage::CopySrc | wgpu::BufferUsage::CopyDst;
    gpu_buffers[t] = context.Device().CreateBuffer(&desc);
  }

  ErrorSink sink;
  std::barrier start{kThreads};
  std::vector<std::thread> threads;
  for (int t = 0; t < kThreads; ++t) {
    threads.emplace_back([&, t]() {
      std::vector<float> input(kElements, static_cast<float>(t + 1));
      std::vector<float> output(kElements);
      start.arrive_and_wait();
      try {
        for (int i = 0; i < kIters && !sink.Failed(); ++i) {
          ORT_THROW_IF_ERROR(data_transfers[t]->CopyTensor(input.data(), false, gpu_buffers[t].Get(), true,
                                                           input.size() * sizeof(float)));
          ORT_THROW_IF_ERROR(data_transfers[t]->CopyTensor(gpu_buffers[t].Get(), true, output.data(), false,
                                                           output.size() * sizeof(float)));
          if (!std::all_of(output.begin(), output.end(), [&](float value) { return value == input[0]; })) {
            sink.Record("F.thread" + std::to_string(t) + " copied incorrect data");
          }
        }
      } catch (const std::exception& e) {
        sink.Record("F.thread" + std::to_string(t) + " threw: " + e.what());
      }
    });
  }
  for (auto& thread : threads) thread.join();

  ASSERT_FALSE(sink.Failed()) << sink.FirstError();
}
#endif
}  // namespace
}  // namespace onnxruntime::test

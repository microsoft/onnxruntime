// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include "gtest/gtest.h"
#include "contrib_ops/cuda/bert/xqa/xqa_loader.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "test/util/include/scoped_env_vars.h"

namespace onnxruntime::contrib::cuda::test {
namespace {

struct DeviceDeleter {
  void operator()(void* buffer) const { cudaFree(buffer); }
};

std::unique_ptr<void, DeviceDeleter> DeviceBuffer(size_t bytes, const void* source = nullptr) {
  void* buffer = nullptr;
  CUDA_CALL_THROW(cudaMalloc(&buffer, bytes));
  std::unique_ptr<void, DeviceDeleter> result(buffer);
  if (source != nullptr) {
    CUDA_CALL_THROW(cudaMemcpy(buffer, source, bytes, cudaMemcpyHostToDevice));
  }
  return result;
}

class XqaMultiBlockTest : public ::testing::Test {
 protected:
  XqaMultiBlockTest() = default;

  void SetUp() override {
#ifdef _WIN32
    const char* original_splits = std::getenv("XQA_NB_SUB_SEQ");
    original_splits_ = original_splits == nullptr ? "" : original_splits;
#endif
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "CUDA device unavailable";
    }
    int device = 0;
    CUDA_CALL_THROW(cudaGetDevice(&device));
    CUDA_CALL_THROW(cudaGetDeviceProperties(&properties_, device));
    if (properties_.major < 8 || properties_.multiProcessorCount < 4) {
      GTEST_SKIP() << "XQA multi-block regression requires SM80+ with at least four SMs";
    }
    CUDA_CALL_THROW(cudaStreamCreate(&stream_));
  }

  void TearDown() override {
#ifdef _WIN32
    EXPECT_EQ(_putenv_s("XQA_NB_SUB_SEQ", original_splits_.c_str()), 0);
#endif
    if (executable_ != nullptr) CUDA_CALL_THROW(cudaGraphExecDestroy(executable_));
    if (graph_ != nullptr) CUDA_CALL_THROW(cudaGraphDestroy(graph_));
    if (stream_ != nullptr) CUDA_CALL_THROW(cudaStreamDestroy(stream_));
  }

  template <typename T>
  void Run(bool capture) {
    constexpr int head_size = 64;
    constexpr int capacity = 2503;
    const size_t shared_memory = GetXQARequiredSharedMemoryBytes(properties_, head_size, 1, 1);
    if (shared_memory > properties_.sharedMemPerBlockOptin) {
      GTEST_SKIP() << "XQA kernel shared memory exceeds the device limit";
    }

    std::vector<T> query(head_size, static_cast<T>(0.0f));
    std::vector<T> key(capacity * head_size, static_cast<T>(0.0f));
    std::vector<T> value(key.size());
    for (int token = 0; token < capacity; ++token) {
      for (int channel = 0; channel < head_size; ++channel) {
        // Four interleaved partitions have distinct means, none equal to the full attention result.
        value[token * head_size + channel] = static_cast<T>(
            1.0f + static_cast<float>((token / 256) % 4) + static_cast<float>(channel % 8) / 16.0f);
      }
    }
    auto device_query = DeviceBuffer(query.size() * sizeof(T), query.data());
    auto device_key = DeviceBuffer(key.size() * sizeof(T), key.data());
    auto device_value = DeviceBuffer(value.size() * sizeof(T), value.data());
    auto device_output = DeviceBuffer(query.size() * sizeof(T));
    int past_length = 504;
    auto device_length = DeviceBuffer(sizeof(past_length), &past_length);
    const size_t workspace_bytes = GetXQAScratchSize(properties_, 1, 1, 1, head_size, capacity,
                                                     XqaQuantType::kNone, std::is_same_v<T, __nv_bfloat16>);
    auto workspace = DeviceBuffer(workspace_bytes);
    const auto launch = [&]() {
      return LaunchXQAKernel<T>(
          properties_, stream_, device_query.get(), device_key.get(), device_value.get(), device_output.get(),
          1, 1, 1, head_size, capacity, 0.125f, -1, false,
          static_cast<const int*>(device_length.get()), nullptr, nullptr, nullptr,
          XqaQuantType::kNone, workspace.get(), workspace_bytes);
    };

    for (int splits : {1, 4}) {
      SCOPED_TRACE(splits);
      onnxruntime::test::ScopedEnvironmentVariables scoped_env_vars{{{"XQA_NB_SUB_SEQ", std::to_string(splits)}}};
#ifdef _WIN32
      // XQA uses getenv, whose CRT environment is separate from the Win32 environment.
      ASSERT_EQ(_putenv_s("XQA_NB_SUB_SEQ", std::to_string(splits).c_str()), 0);
#endif
      ORT_THROW_IF_ERROR(launch());
      CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
      if (capture) {
        CUDA_CALL_THROW(cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal));
        ORT_THROW_IF_ERROR(launch());
        CUDA_CALL_THROW(cudaStreamEndCapture(stream_, &graph_));
        CUDA_CALL_THROW(cudaGraphInstantiate(&executable_, graph_, nullptr, nullptr, 0));

        size_t node_count = 0;
        CUDA_CALL_THROW(cudaGraphGetNodes(graph_, nullptr, &node_count));
        std::vector<cudaGraphNode_t> nodes(node_count);
        CUDA_CALL_THROW(cudaGraphGetNodes(graph_, nodes.data(), &node_count));
        int kernel_count = 0;
        for (auto node : nodes) {
          cudaGraphNodeType type;
          CUDA_CALL_THROW(cudaGraphNodeGetType(node, &type));
          if (type == cudaGraphNodeTypeKernel) {
            cudaKernelNodeParams params{};
            CUDA_CALL_THROW(cudaGraphKernelNodeGetParams(node, &params));
            EXPECT_EQ(params.gridDim.x, static_cast<unsigned int>(splits));
            ++kernel_count;
          }
        }
        ASSERT_EQ(kernel_count, 1);
      }

      for (int length : {505, 1025, capacity}) {
        SCOPED_TRACE(length);
        past_length = length - 1;
        CUDA_CALL_THROW(cudaMemcpyAsync(device_length.get(), &past_length, sizeof(past_length),
                                        cudaMemcpyHostToDevice, stream_));
        std::vector<double> expected(head_size, 0.0);
        for (int token = 0; token < length; ++token) {
          for (int channel = 0; channel < head_size; ++channel) {
            expected[channel] += static_cast<float>(value[token * head_size + channel]) / static_cast<double>(length);
          }
        }
        std::vector<T> actual(head_size);
        std::vector<T> first_output;
        for (int repeat = 0; repeat < 200; ++repeat) {
          SCOPED_TRACE(repeat);
          CUDA_CALL_THROW(cudaMemsetAsync(device_output.get(), 0xff, query.size() * sizeof(T), stream_));
          if (capture) {
            CUDA_CALL_THROW(cudaGraphLaunch(executable_, stream_));
          } else {
            ORT_THROW_IF_ERROR(launch());
          }
          CUDA_CALL_THROW(cudaMemcpyAsync(actual.data(), device_output.get(), actual.size() * sizeof(T),
                                          cudaMemcpyDeviceToHost, stream_));
          CUDA_CALL_THROW(cudaStreamSynchronize(stream_));
          for (int channel = 0; channel < head_size; ++channel) {
            ASSERT_TRUE(std::isfinite(static_cast<float>(actual[channel]))) << channel;
            ASSERT_NEAR(static_cast<float>(actual[channel]), expected[channel],
                        (std::is_same_v<T, __nv_bfloat16> ? 0.03 : 0.003))
                << channel;
          }
          if (first_output.empty()) {
            first_output = actual;
          } else {
            ASSERT_EQ(std::memcmp(actual.data(), first_output.data(), actual.size() * sizeof(T)), 0);
          }
        }
      }
      if (capture) {
        CUDA_CALL_THROW(cudaGraphExecDestroy(executable_));
        executable_ = nullptr;
        CUDA_CALL_THROW(cudaGraphDestroy(graph_));
        graph_ = nullptr;
      }
    }
  }

 private:
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(XqaMultiBlockTest);

  cudaDeviceProp properties_{};
  cudaStream_t stream_ = nullptr;
  cudaGraph_t graph_ = nullptr;
  cudaGraphExec_t executable_ = nullptr;
#ifdef _WIN32
  std::string original_splits_;
#endif
};

TEST_F(XqaMultiBlockTest, Float16) { Run<half>(false); }
TEST_F(XqaMultiBlockTest, Float16GraphReplay) { Run<half>(true); }
TEST_F(XqaMultiBlockTest, BFloat16) { Run<__nv_bfloat16>(false); }
TEST_F(XqaMultiBlockTest, BFloat16GraphReplay) { Run<__nv_bfloat16>(true); }

}  // namespace
}  // namespace onnxruntime::contrib::cuda::test

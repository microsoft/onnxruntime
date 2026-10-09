#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <memory>
#include <random>
#include <type_traits>
#include <vector>

#include <cuda_bf16.h>
#include "gtest/gtest.h"
#include "contrib_ops/cuda/bert/xqa/xqa_loader.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {
namespace test {
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

template <typename T>
void CheckH512Reference() {
  int device_count = 0;
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count == 0) GTEST_SKIP() << "CUDA device unavailable";
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp properties{};
  ASSERT_EQ(cudaGetDeviceProperties(&properties, device), cudaSuccess);
  if (properties.major < 8) GTEST_SKIP() << "H512 requires SM80 or newer";

  constexpr int batch_size = 2;
  constexpr int kv_heads = 2;
  constexpr int head_size = 512;
  const float scale = 1.0f / std::sqrt(static_cast<float>(head_size));
  struct Case {
    int length;
    int ratio;
    int window;
    bool bsnh;
    bool sink;
  };
  for (const auto& config : std::vector<Case>{{1, 1, -1, false, false}, {7, 3, 1, true, true}, {31, 8, 32, false, true}, {32, 3, -1, true, false}, {33, 8, 32, true, true}, {257, 1, 512, false, false}, {793, 8, 512, false, true}, {793, 3, -1, true, false}}) {
    SCOPED_TRACE(testing::Message() << "length=" << config.length << " ratio=" << config.ratio
                                    << " window=" << config.window << " bsnh=" << config.bsnh
                                    << " sink=" << config.sink);
    const int num_heads = kv_heads * config.ratio;
    const int capacity = config.length + 9;
    const std::vector<int> lengths{config.length - 1, std::max(1, config.length - 3) - 1};
    std::vector<T> query(batch_size * num_heads * head_size);
    std::vector<T> key(batch_size * kv_heads * capacity * head_size);
    std::vector<T> value(key.size());
    std::vector<float> sinks(num_heads);
    std::mt19937 random(42);
    std::uniform_real_distribution<float> distribution(-1.0f, 1.0f);
    for (auto& element : query) element = static_cast<T>(distribution(random));
    for (auto& element : key) element = static_cast<T>(distribution(random));
    for (auto& element : value) element = static_cast<T>(distribution(random));
    for (auto& element : sinks) element = distribution(random) * 4.0f;
    const auto cache_index = [&](int batch, int token, int head, int channel) {
      return config.bsnh ? ((batch * capacity + token) * kv_heads + head) * head_size + channel
                         : ((batch * kv_heads + head) * capacity + token) * head_size + channel;
    };
    for (int batch = 0; batch < batch_size; ++batch) {
      for (int token = lengths[batch] + 1; token < capacity; ++token) {
        for (int head = 0; head < kv_heads; ++head) {
          for (int channel = 0; channel < head_size; ++channel) {
            value[cache_index(batch, token, head, channel)] = static_cast<T>(100.0f);
          }
        }
      }
    }

    std::vector<double> expected(query.size());
    for (int batch = 0; batch < batch_size; ++batch) {
      const int length = lengths[batch] + 1;
      const int first = config.window < 0 ? 0 : std::max(0, length - config.window);
      for (int head = 0; head < num_heads; ++head) {
        std::vector<double> scores(length);
        double maximum = config.sink ? sinks[head] : -std::numeric_limits<double>::infinity();
        for (int token = first; token < length; ++token) {
          double score = 0.0;
          for (int channel = 0; channel < head_size; ++channel) {
            score += static_cast<double>(static_cast<float>(query[(batch * num_heads + head) * head_size + channel])) *
                     static_cast<float>(key[cache_index(batch, token, head / config.ratio, channel)]);
          }
          scores[token] = score * scale;
          maximum = std::max(maximum, scores[token]);
        }
        double denominator = config.sink ? std::exp(sinks[head] - maximum) : 0.0;
        for (int token = first; token < length; ++token) {
          scores[token] = std::exp(scores[token] - maximum);
          denominator += scores[token];
        }
        for (int channel = 0; channel < head_size; ++channel) {
          double weighted_value = 0.0;
          for (int token = first; token < length; ++token) {
            weighted_value += scores[token] * static_cast<float>(value[cache_index(batch, token, head / config.ratio, channel)]);
          }
          expected[(batch * num_heads + head) * head_size + channel] = weighted_value / denominator;
        }
      }
    }

    auto device_query = DeviceBuffer(query.size() * sizeof(T), query.data());
    auto device_key = DeviceBuffer(key.size() * sizeof(T), key.data());
    auto device_value = DeviceBuffer(value.size() * sizeof(T), value.data());
    auto device_lengths = DeviceBuffer(lengths.size() * sizeof(int), lengths.data());
    auto device_sinks = DeviceBuffer(sinks.size() * sizeof(float), sinks.data());
    auto device_output = DeviceBuffer(query.size() * sizeof(T));
    const size_t workspace_bytes = GetXQAScratchSize(properties, batch_size, num_heads, kv_heads,
                                                     head_size, capacity, XqaQuantType::kNone,
                                                     std::is_same_v<T, __nv_bfloat16>);
    auto workspace = DeviceBuffer(workspace_bytes);
    const auto status = LaunchXQAKernel<T>(properties, nullptr, device_query.get(), device_key.get(),
                                           device_value.get(), device_output.get(), batch_size, num_heads,
                                           kv_heads, head_size, capacity, scale, config.window, config.bsnh,
                                           static_cast<const int*>(device_lengths.get()),
                                           config.sink ? static_cast<const float*>(device_sinks.get()) : nullptr,
                                           nullptr, nullptr, XqaQuantType::kNone, workspace.get(), workspace_bytes);
    ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
    std::vector<T> actual(query.size());
    ASSERT_EQ(cudaMemcpy(actual.data(), device_output.get(), actual.size() * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    const double tolerance = std::is_same_v<T, __nv_bfloat16> ? 0.004 : 0.0005;
    for (size_t index = 0; index < actual.size(); ++index) {
      ASSERT_TRUE(std::isfinite(static_cast<float>(actual[index]))) << index;
      ASSERT_NEAR(static_cast<float>(actual[index]), expected[index], tolerance) << index;
    }
    std::vector<T> unchanged(key.size());
    ASSERT_EQ(cudaMemcpy(unchanged.data(), device_key.get(), key.size() * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(std::memcmp(unchanged.data(), key.data(), key.size() * sizeof(T)), 0);
    ASSERT_EQ(cudaMemcpy(unchanged.data(), device_value.get(), value.size() * sizeof(T), cudaMemcpyDeviceToHost), cudaSuccess);
    EXPECT_EQ(std::memcmp(unchanged.data(), value.data(), value.size() * sizeof(T)), 0);
  }
}

// Compare FP16 H512 decode with FP64 reference across cache layouts, windows, and attention sinks.
TEST(XqaH512Test, Float16Reference) { CheckH512Reference<half>(); }
// Apply the same reference and cache-immutability checks to the BF16 H512 loader path.
TEST(XqaH512Test, BFloat16Reference) { CheckH512Reference<__nv_bfloat16>(); }

}  // namespace
}  // namespace test
}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
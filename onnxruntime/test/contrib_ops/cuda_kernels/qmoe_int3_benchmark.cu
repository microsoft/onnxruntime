#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <type_traits>
#include <vector>

#include "contrib_ops/cuda/moe/qmoe_int3.h"

using namespace onnxruntime::contrib::cuda;

static void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

struct Buffer {
  void* data = nullptr;
  explicit Buffer(size_t bytes) {
    Check(cudaMalloc(&data, bytes));
    Check(cudaMemset(data, 0x35, bytes));
  }
  ~Buffer() { Check(cudaFree(data)); }
  Buffer(const Buffer&) = delete;
  Buffer& operator=(const Buffer&) = delete;
};

template <typename Action>
static float Time(Action action, int repeat = 1) {
  cudaEvent_t start, end;
  Check(cudaEventCreate(&start));
  Check(cudaEventCreate(&end));
  Check(cudaEventRecord(start));
  for (int iteration = 0; iteration < repeat; ++iteration) action();
  Check(cudaEventRecord(end));
  Check(cudaEventSynchronize(end));
  float milliseconds = 0;
  Check(cudaEventElapsedTime(&milliseconds, start, end));
  Check(cudaEventDestroy(start));
  Check(cudaEventDestroy(end));
  return milliseconds / repeat;
}

template <typename T>
static void Run(int rows, int hidden, int intermediate, int experts, int top_k,
                int block, int first_bits, int second_bits) {
  const size_t first_bytes = size_t(experts) * 2 * intermediate * hidden * first_bits / 8;
  const size_t second_bytes = size_t(experts) * hidden * intermediate * second_bits / 8;
  Buffer first(first_bytes), second(second_bytes), first_private(first_bytes), second_private(second_bytes);
  Buffer first_scale(size_t(experts) * 2 * intermediate * (hidden / block) * sizeof(T));
  Buffer second_scale(size_t(experts) * hidden * (intermediate / block) * sizeof(T));
  Buffer input(size_t(rows) * hidden * sizeof(T)), output(size_t(rows) * hidden * sizeof(T));
  Buffer activation(size_t(rows) * top_k * intermediate * sizeof(T));
  Buffer selected(size_t(rows) * top_k * sizeof(int)), routing(size_t(rows) * top_k * sizeof(float));
  Check(cudaMemset(selected.data, 0, size_t(rows) * top_k * sizeof(int)));
  std::vector<float> routing_host(rows * top_k, 1.0f / top_k);
  Check(cudaMemcpy(routing.data, routing_host.data(), routing_host.size() * sizeof(float), cudaMemcpyHostToDevice));
  Int3MoeParams canonical{};
  canonical.input = input.data;
  canonical.output = output.data;
  canonical.activation = activation.data;
  canonical.experts = static_cast<const int*>(selected.data);
  canonical.routing_scales = static_cast<const float*>(routing.data);
  canonical.fc1 = {static_cast<const uint8_t*>(first.data), first_scale.data, nullptr, nullptr, first_bits};
  canonical.fc2 = {static_cast<const uint8_t*>(second.data), second_scale.data, nullptr, nullptr, second_bits};
  canonical.rows = rows;
  canonical.hidden = hidden;
  canonical.intermediate = intermediate;
  canonical.top_k = top_k;
  canonical.block_size = block;
  canonical.alpha = 1;
  canonical.limit = 10;
  canonical.is_bfloat16 = std::is_same_v<T, __nv_bfloat16>;
  auto packed = canonical;
  packed.fc1.weights = static_cast<const uint8_t*>(first_private.data);
  packed.fc2.weights = static_cast<const uint8_t*>(second_private.data);
  packed.fc1.layout_version = packed.fc2.layout_version = 1;
  const float prepack = Time([&] {
    Check(PackInt3MoeWeights(canonical.fc1.weights, static_cast<uint8_t*>(first_private.data),
                             size_t(experts) * 2 * intermediate, hidden, first_bits, nullptr));
    Check(PackInt3MoeWeights(canonical.fc2.weights, static_cast<uint8_t*>(second_private.data),
                             size_t(experts) * hidden, intermediate, second_bits, nullptr));
  });
  const float first_run = Time([&] { Check(LaunchInt3Moe(packed, nullptr)); });
  std::vector<T> packed_output(size_t(rows) * hidden), canonical_output(packed_output.size());
  Check(cudaMemcpy(packed_output.data(), output.data, packed_output.size() * sizeof(T), cudaMemcpyDeviceToHost));
  Check(LaunchInt3Moe(canonical, nullptr));
  Check(cudaMemcpy(canonical_output.data(), output.data, canonical_output.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (size_t index = 0; index < packed_output.size(); ++index) {
    if (float(packed_output[index]) != float(canonical_output[index])) {
      std::fprintf(stderr, "benchmark parity failure at %zu\n", index);
      std::exit(2);
    }
  }
  for (int warmup = 0; warmup < 10; ++warmup) {
    Check(LaunchInt3Moe(canonical, nullptr));
    Check(LaunchInt3Moe(packed, nullptr));
  }
  std::vector<float> baseline, optimized;
  for (int sample = 0; sample < 9; ++sample) {
    float canonical_ms = 0, packed_ms = 0;
    const auto canonical_time = [&] { canonical_ms = Time([&] { Check(LaunchInt3Moe(canonical, nullptr)); }, 20); };
    const auto packed_time = [&] { packed_ms = Time([&] { Check(LaunchInt3Moe(packed, nullptr)); }, 20); };
    if (sample % 2) {
      packed_time();
      canonical_time();
    } else {
      canonical_time();
      packed_time();
    }
    baseline.push_back(canonical_ms);
    optimized.push_back(packed_ms);
  }
  std::sort(baseline.begin(), baseline.end());
  std::sort(optimized.begin(), optimized.end());
  std::printf(
      "{\"dtype\":\"%s\",\"rows\":%d,\"hidden\":%d,\"intermediate\":%d,"
      "\"experts\":%d,\"top_k\":%d,\"block\":%d,\"bits\":[%d,%d],"
      "\"weight_bytes\":%zu,\"private_weight_bytes\":%zu,\"coexist_weight_bytes\":%zu,"
      "\"prepack_gpu_ms\":%.6f,\"first_private_gpu_ms\":%.6f,"
      "\"canonical_median_ms\":%.6f,\"private_median_ms\":%.6f,"
      "\"canonical_min_ms\":%.6f,\"canonical_max_ms\":%.6f,"
      "\"private_min_ms\":%.6f,\"private_max_ms\":%.6f}\n",
      canonical.is_bfloat16 ? "bf16" : "fp16", rows, hidden, intermediate, experts, top_k,
      block, first_bits, second_bits, first_bytes + second_bytes, first_bytes + second_bytes,
      2 * (first_bytes + second_bytes), prepack, first_run, baseline[4], optimized[4],
      baseline.front(), baseline.back(), optimized.front(), optimized.back());
}

int main() {
  int device = 0, runtime = 0, driver = 0;
  cudaDeviceProp properties{};
  Check(cudaGetDevice(&device));
  Check(cudaGetDeviceProperties(&properties, device));
  Check(cudaRuntimeGetVersion(&runtime));
  Check(cudaDriverGetVersion(&driver));
  std::printf("{\"device\":\"%s\",\"ordinal\":%d,\"runtime\":%d,\"driver\":%d,\"uuid_hex\":\"",
              properties.name, device, runtime, driver);
  for (unsigned char byte : properties.uuid.bytes) std::printf("%02x", byte);
  std::printf("\",\"warmup\":10,\"samples\":9,\"launches_per_sample\":20,\"layout_version\":1}\n");
  for (int rows : {1, 4, 16}) {
    for (auto bits : {std::pair{3, 3}, std::pair{3, 4}, std::pair{4, 3}}) {
      Run<half>(rows, 2048, 768, 128, 8, 64, bits.first, bits.second);
      Run<__nv_bfloat16>(rows, 2048, 768, 128, 8, 64, bits.first, bits.second);
    }
  }
  for (int block : {32, 128}) {
    Run<half>(1, 256, 384, 3, 2, block, 3, 3);
    Run<__nv_bfloat16>(1, 256, 384, 3, 2, block, 3, 3);
  }
}
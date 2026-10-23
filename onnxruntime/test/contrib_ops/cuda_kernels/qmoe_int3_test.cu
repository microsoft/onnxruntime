#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "contrib_ops/cuda/moe/qmoe_int3.h"

using namespace onnxruntime::contrib::cuda;

static void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(1);
  }
}

struct DeviceBuffer {
  void* data = nullptr;
  explicit DeviceBuffer(size_t bytes, const void* source = nullptr) {
    Check(cudaMalloc(&data, bytes));
    if (source) Check(cudaMemcpy(data, source, bytes, cudaMemcpyHostToDevice));
  }
  ~DeviceBuffer() { Check(cudaFree(data)); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

static void Pack(std::vector<uint8_t>& bytes, size_t row, size_t count,
                 size_t index, int bits, unsigned int code) {
  const size_t row_bytes = (count * bits + 7) / 8;
  for (int bit = 0; bit < bits; ++bit) {
    const size_t position = index * bits + bit;
    bytes[row * row_bytes + position / 8] |= ((code >> bit) & 1u) << (position % 8);
  }
}

template <typename T>
struct Projection {
  std::vector<uint8_t> packed;
  std::vector<uint8_t> zeros;
  std::vector<T> scales;
  std::vector<T> bias;
  std::vector<float> dense;
  Projection(int rows, int features, int block, int bits, bool explicit_zeros)
      : packed(rows * ((features * bits + 7) / 8)),
        zeros(rows * (((features / block) * bits + 7) / 8)),
        scales(rows * (features / block)),
        bias(rows),
        dense(rows * features) {
    for (int row = 0; row < rows; ++row) {
      bias[row] = T((row % 7 - 3) * 0.003f);
      for (int group = 0; group < features / block; ++group) {
        scales[row * (features / block) + group] = T(0.007f * (1 + (row + group) % 3));
        const unsigned int zero = bits == 3 ? 4 : (explicit_zeros ? 6 + (row + group) % 3 : 8);
        Pack(zeros, row, features / block, group, bits, zero);
        for (int offset = 0; offset < block; ++offset) {
          const int feature = group * block + offset;
          const unsigned int code = (row * 11 + feature * 7) % (1 << bits);
          Pack(packed, row, features, feature, bits, code);
          dense[row * features + feature] = float(T((int(code) - int(zero)) *
                                                    float(scales[row * (features / block) + group])));
        }
      }
    }
  }
};

template <typename T>
bool RunCase(int rows, int block, int fc1_bits, int fc2_bits, bool with_bias, bool with_zeros) {
  constexpr int experts = 3;
  constexpr int top_k = 2;
  const int hidden = block * 2;
  const int inter = block * 3;
  Projection<T> first(experts * inter * 2, hidden, block, fc1_bits, with_zeros);
  Projection<T> second(experts * hidden, inter, block, fc2_bits, with_zeros);
  std::vector<T> input(rows * hidden);
  for (size_t index = 0; index < input.size(); ++index) input[index] = T((int(index % 17) - 8) * 0.04f);
  std::vector<int> selected(rows * top_k);
  std::vector<float> routing(rows * top_k);
  for (int route = 0; route < rows * top_k; ++route) {
    selected[route] = (route / top_k + route % top_k) % experts;
    routing[route] = route % top_k ? 0.3f : 0.6f;
  }
  DeviceBuffer input_device(input.size() * sizeof(T), input.data());
  DeviceBuffer experts_device(selected.size() * sizeof(int), selected.data());
  DeviceBuffer routing_device(routing.size() * sizeof(float), routing.data());
  DeviceBuffer first_weights(first.packed.size(), first.packed.data());
  DeviceBuffer first_scales(first.scales.size() * sizeof(T), first.scales.data());
  DeviceBuffer first_zeros(first.zeros.size(), first.zeros.data());
  DeviceBuffer first_bias(first.bias.size() * sizeof(T), first.bias.data());
  DeviceBuffer second_weights(second.packed.size(), second.packed.data());
  DeviceBuffer second_scales(second.scales.size() * sizeof(T), second.scales.data());
  DeviceBuffer second_zeros(second.zeros.size(), second.zeros.data());
  DeviceBuffer second_bias(second.bias.size() * sizeof(T), second.bias.data());
  DeviceBuffer activation(rows * top_k * inter * sizeof(T));
  DeviceBuffer output(rows * hidden * sizeof(T));
  Int3MoeParams params{};
  params.input = input_device.data;
  params.output = output.data;
  params.activation = activation.data;
  params.experts = static_cast<const int*>(experts_device.data);
  params.routing_scales = static_cast<const float*>(routing_device.data);
  params.fc1 = {static_cast<const uint8_t*>(first_weights.data), first_scales.data,
                with_zeros ? static_cast<const uint8_t*>(first_zeros.data) : nullptr,
                with_bias ? first_bias.data : nullptr, fc1_bits};
  params.fc2 = {static_cast<const uint8_t*>(second_weights.data), second_scales.data,
                with_zeros ? static_cast<const uint8_t*>(second_zeros.data) : nullptr,
                with_bias ? second_bias.data : nullptr, fc2_bits};
  params.rows = rows;
  params.hidden = hidden;
  params.intermediate = inter;
  params.top_k = top_k;
  params.block_size = block;
  params.alpha = 1.2f;
  params.beta = 0.1f;
  params.limit = 0.05f;
  params.is_bfloat16 = std::is_same_v<T, __nv_bfloat16>;
  DeviceBuffer invalid_device(sizeof(int));
  Check(cudaMemset(invalid_device.data, 0, sizeof(int)));
  Check(ValidateInt3MoeContent(params.fc1, experts * inter * 2, hidden, block,
                               params.is_bfloat16, static_cast<int*>(invalid_device.data), nullptr));
  Check(ValidateInt3MoeContent(params.fc2, experts * hidden, inter, block,
                               params.is_bfloat16, static_cast<int*>(invalid_device.data), nullptr));
  int invalid = -1;
  Check(cudaMemcpy(&invalid, invalid_device.data, sizeof(int), cudaMemcpyDeviceToHost));
  if (invalid) return false;
  for (int repeat = 0; repeat < 3; ++repeat) Check(LaunchInt3Moe(params, nullptr));
  std::vector<T> actual(rows * hidden);
  Check(cudaMemcpy(actual.data(), output.data, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
  DeviceBuffer first_private(first.packed.size());
  DeviceBuffer second_private(second.packed.size());
  Check(PackInt3MoeWeights(params.fc1.weights, static_cast<uint8_t*>(first_private.data),
                           experts * inter * 2, hidden, fc1_bits, nullptr));
  Check(PackInt3MoeWeights(params.fc2.weights, static_cast<uint8_t*>(second_private.data),
                           experts * hidden, inter, fc2_bits, nullptr));
  params.fc1.weights = static_cast<const uint8_t*>(first_private.data);
  params.fc2.weights = static_cast<const uint8_t*>(second_private.data);
  params.fc1.layout_version = params.fc2.layout_version = 1;
  for (int repeat = 0; repeat < 3; ++repeat) Check(LaunchInt3Moe(params, nullptr));
  std::vector<T> private_actual(actual.size());
  Check(cudaMemcpy(private_actual.data(), output.data, actual.size() * sizeof(T), cudaMemcpyDeviceToHost));
  for (size_t index = 0; index < actual.size(); ++index) {
    if (float(actual[index]) != float(private_actual[index])) {
      std::fprintf(stderr, "Private bitplane output differs at %zu\n", index);
      return false;
    }
  }
  std::vector<T> reference_activation(rows * top_k * inter);
  for (int route = 0; route < rows * top_k; ++route) {
    for (int column = 0; column < inter; ++column) {
      const int weight_row = selected[route] * inter * 2 + column * 2;
      float gate = with_bias ? float(first.bias[weight_row]) : 0;
      float up = with_bias ? float(first.bias[weight_row + 1]) : 0;
      for (int feature = 0; feature < hidden; ++feature) {
        gate += float(input[(route / top_k) * hidden + feature]) * first.dense[weight_row * hidden + feature];
        up += float(input[(route / top_k) * hidden + feature]) * first.dense[(weight_row + 1) * hidden + feature];
      }
      gate = std::min(gate, params.limit);
      up = std::clamp(up, -params.limit, params.limit);
      reference_activation[route * inter + column] = T(gate / (1 + std::exp(-params.alpha * gate)) * (up + params.beta));
    }
  }
  float max_error = 0;
  for (int token = 0; token < rows; ++token) {
    for (int column = 0; column < hidden; ++column) {
      float expected = 0;
      for (int slot = 0; slot < top_k; ++slot) {
        const int route = token * top_k + slot;
        const int weight_row = selected[route] * hidden + column;
        float value = 0;
        for (int feature = 0; feature < inter; ++feature) {
          value += float(reference_activation[route * inter + feature]) * second.dense[weight_row * inter + feature];
        }
        if (with_bias) value += float(second.bias[weight_row]);
        expected += float(T(value)) * routing[route];
      }
      const float rounded = float(T(expected));
      const float observed = float(actual[token * hidden + column]);
      max_error = std::max(max_error, std::abs(rounded - observed));
      if (!std::isfinite(observed) || std::abs(rounded - observed) > 0.00015f + std::abs(rounded) * 0.02f) {
        std::fprintf(stderr, "FAIL type=%s rows=%d block=%d bits=%d/%d bias=%d zeros=%d expected=%g actual=%g\n",
                     params.is_bfloat16 ? "bf16" : "fp16", rows, block, fc1_bits, fc2_bits,
                     with_bias, with_zeros, rounded, observed);
        return false;
      }
    }
  }
  return true;
}

template <typename T>
static bool CheckMutableContent() {
  const bool bfloat = std::is_same_v<T, __nv_bfloat16>;
  std::vector<T> scales{T(0.5f)};
  uint8_t zero = 4;
  DeviceBuffer scale_device(sizeof(T), scales.data());
  DeviceBuffer zero_device(1, &zero);
  DeviceBuffer invalid_device(sizeof(int));
  Int3Projection projection{nullptr, scale_device.data, static_cast<const uint8_t*>(zero_device.data), nullptr, 3};
  const auto validate = [&](int expected) {
    Check(cudaMemset(invalid_device.data, 0, sizeof(int)));
    Check(ValidateInt3MoeContent(projection, 1, 32, 32, bfloat,
                                 static_cast<int*>(invalid_device.data), nullptr));
    int actual = -1;
    Check(cudaMemcpy(&actual, invalid_device.data, sizeof(int), cudaMemcpyDeviceToHost));
    return actual == expected;
  };
  if (!validate(0)) return false;
  for (float value : {-1.0f, INFINITY, NAN}) {
    scales[0] = T(value);
    Check(cudaMemcpy(scale_device.data, scales.data(), sizeof(T), cudaMemcpyHostToDevice));
    if (!validate(1)) return false;
  }
  projection.scales = nullptr;
  if (!validate(0)) return false;
  for (uint8_t malformed : {uint8_t{5}, uint8_t{0x84}}) {
    Check(cudaMemcpy(zero_device.data, &malformed, 1, cudaMemcpyHostToDevice));
    if (!validate(1)) return false;
  }
  projection.zeros = nullptr;
  if (!validate(0)) return false;
  scales[0] = T(0.0f);
  Check(cudaMemcpy(scale_device.data, scales.data(), sizeof(T), cudaMemcpyHostToDevice));
  projection.scales = scale_device.data;
  if (!validate(0)) return false;
  scales[0] = T(NAN);
  Check(cudaMemcpy(scale_device.data, scales.data(), sizeof(T), cudaMemcpyHostToDevice));
  return validate(1);
}

int main() {
  if (!CheckMutableContent<half>() || !CheckMutableContent<__nv_bfloat16>()) {
    std::fprintf(stderr, "FAIL independent mutable scale/zero content validation\n");
    return 1;
  }
  Int3MoeParams malformed{};
  malformed.rows = 1;
  malformed.hidden = malformed.intermediate = malformed.block_size = 32;
  malformed.top_k = 1;
  malformed.fc1.bits = malformed.fc2.bits = 3;
  malformed.fc1.layout_version = 2;
  if (LaunchInt3Moe(malformed, nullptr) != cudaErrorInvalidValue ||
      PackInt3MoeWeights(nullptr, nullptr, 1, 32, 3, nullptr) != cudaErrorInvalidValue) return 1;
  int cases = 0;
  for (int rows : {1, 5, 17}) {
    for (int block : {32, 64, 128}) {
      for (auto bits : {std::pair{3, 3}, std::pair{3, 4}, std::pair{4, 3}}) {
        for (bool bias : {false, true}) {
          for (bool zeros : {false, true}) {
            if (!RunCase<half>(rows, block, bits.first, bits.second, bias, zeros) ||
                !RunCase<__nv_bfloat16>(rows, block, bits.first, bits.second, bias, zeros)) return 1;
            cases += 2;
          }
        }
      }
    }
  }
  std::printf("PASS: %d FP16/BF16 packed dual-projection cases; each repeated three times\n", cases);
}
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <vector>

#include "contrib_ops/cuda/moe/packed_code.cuh"

template <int Bits>
__global__ void DecodeRows(const uint8_t* packed, unsigned int* codes,
                           size_t rows, size_t count, size_t row_bytes) {
  const size_t index = blockIdx.x * blockDim.x + threadIdx.x;
  if (index < rows * count) {
    codes[index] = onnxruntime::contrib::cuda::LoadRowLocalPackedCode<Bits>(
        packed + (index / count) * row_bytes, row_bytes, index % count);
  }
}

static bool CheckCuda(cudaError_t status) {
  if (status == cudaSuccess) return true;
  std::fprintf(stderr, "%s\n", cudaGetErrorString(status));
  return false;
}

template <int Bits>
bool CheckWidth() {
  for (size_t count : {size_t{1}, size_t{3}, size_t{7}, size_t{8}, size_t{9},
                       size_t{31}, size_t{32}, size_t{33}, size_t{64}, size_t{129}}) {
    constexpr size_t rows = 5;
    const size_t row_bytes = (count * Bits + 7) / 8;
    std::vector<uint8_t> packed(rows * row_bytes, 0);
    std::vector<unsigned int> expected(rows * count);
    for (size_t row = 0; row < rows; ++row) {
      for (size_t element = 0; element < count; ++element) {
        const unsigned int code = static_cast<unsigned int>((row * 13 + element * 7) % (1u << Bits));
        expected[row * count + element] = code;
        for (int bit = 0; bit < Bits; ++bit) {
          const size_t position = element * Bits + bit;
          packed[row * row_bytes + position / 8] |= ((code >> bit) & 1u) << (position % 8);
        }
      }
    }
    uint8_t* device_packed = nullptr;
    unsigned int* device_codes = nullptr;
    if (!CheckCuda(cudaMalloc(&device_packed, packed.size()))) return false;
    if (!CheckCuda(cudaMalloc(&device_codes, expected.size() * sizeof(unsigned int)))) {
      cudaFree(device_packed);
      return false;
    }
    bool passed = CheckCuda(cudaMemcpy(device_packed, packed.data(), packed.size(), cudaMemcpyHostToDevice));
    if (passed) {
      DecodeRows<Bits><<<(expected.size() + 127) / 128, 128>>>(device_packed, device_codes, rows, count, row_bytes);
      passed = CheckCuda(cudaGetLastError());
    }
    std::vector<unsigned int> actual(expected.size());
    passed = passed && CheckCuda(cudaMemcpy(actual.data(), device_codes, actual.size() * sizeof(unsigned int),
                                            cudaMemcpyDeviceToHost));
    passed = CheckCuda(cudaFree(device_codes)) && passed;
    passed = CheckCuda(cudaFree(device_packed)) && passed;
    if (!passed || actual != expected) {
      std::fprintf(stderr, "FAIL: bits=%d count=%zu\n", Bits, count);
      return false;
    }
  }
  return true;
}

int main() {
  if (!(CheckWidth<1>() && CheckWidth<2>() && CheckWidth<3>() && CheckWidth<4>() &&
        CheckWidth<5>() && CheckWidth<6>() && CheckWidth<7>() && CheckWidth<8>())) {
    return 1;
  }
  std::puts("PASS: 80 GPU row-local packed-code cases, widths 1..8, five rows per case");
  return 0;
}
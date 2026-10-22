#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

void Check(cudaError_t status) {
  if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
}

template <typename Value>
struct DeviceBuffer {
  Value* data = nullptr;
  explicit DeviceBuffer(size_t count) { Check(cudaMalloc(&data, count * sizeof(Value))); }
  ~DeviceBuffer() { cudaFree(data); }
  DeviceBuffer(const DeviceBuffer&) = delete;
  DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

__host__ __device__ unsigned Code(size_t row, int feature) {
  uint32_t value = static_cast<uint32_t>(row) * 747796405u + static_cast<uint32_t>(feature) * 2891336453u;
  value ^= value >> 13;
  value *= 277803737u;
  return (value ^ (value >> 17)) & 7u;
}

__global__ void Initialize(uint8_t* weights, half* scales, half* input,
                           size_t rows, int reduction, int block, int tokens) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < rows * reduction * 3 / 8) {
    const size_t row = index / (reduction * 3 / 8);
    const int first_bit = (index % (reduction * 3 / 8)) * 8;
    unsigned value = 0;
    for (int offset = 0; offset < 8; ++offset) {
      const int bit = first_bit + offset;
      value |= ((Code(row, bit / 3) >> (bit % 3)) & 1u) << offset;
    }
    weights[index] = value;
  }
  if (index < rows * (reduction / block)) {
    scales[index] = __float2half(static_cast<float>((index / (reduction / block)) % 7 + 1) / 128);
  }
  if (index < static_cast<size_t>(tokens) * reduction) {
    input[index] = __float2half(static_cast<float>(static_cast<int>((index * 13) % 16) - 8) / 16);
  }
}

__global__ void Repack(const uint8_t* raw, uint8_t* padded, size_t count,
                       int reduction, int block, int stride) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= count) return;
  const int blocks = reduction / block;
  const size_t row = index / (blocks * stride);
  const int position = index % (blocks * stride);
  const int offset = position % stride;
  padded[index] = offset < block * 3 / 8
                      ? raw[row * (reduction * 3 / 8) + (position / stride) * (block * 3 / 8) + offset]
                      : 0xa5;
}

template <int Block, bool Padded>
__device__ float Load(const uint8_t* weights, const half* scales, size_t row,
                      int feature, int reduction, int stride) {
  const int blocks = reduction / Block;
  const size_t byte = Padded ? row * blocks * stride + (feature / Block) * stride + (feature % Block) * 3 / 8
                             : row * (reduction * 3 / 8) + feature * 3 / 8;
  const int shift = (feature * 3) % 8;
  unsigned word = weights[byte];
  if (shift > 5) word |= static_cast<unsigned>(weights[byte + 1]) << 8;
  const int code = static_cast<int>((word >> shift) & 7u) - 4;
  return __half2float(__float2half(code * __half2float(scales[row * blocks + feature / Block])));
}

template <int Block, bool Padded>
__global__ void Dequantize(const uint8_t* weights, const half* scales, half* output,
                           int rows_per_expert, int reduction, int stride, int expert_start) {
  const size_t index = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index >= static_cast<size_t>(8) * rows_per_expert * reduction) return;
  const size_t row = static_cast<size_t>(expert_start) * rows_per_expert + index / reduction;
  output[index] = __float2half(Load<Block, Padded>(weights, scales, row, index % reduction, reduction, stride));
}

template <int Block, bool Padded>
__global__ void Project(const uint8_t* weights, const half* scales, const half* input,
                        half* output, int rows_per_expert, int reduction, int stride,
                        int expert_start, int tokens) {
  const size_t item = static_cast<size_t>(blockIdx.x) * 4 + threadIdx.x / 32;
  if (item >= static_cast<size_t>(tokens) * 8 * rows_per_expert) return;
  const int token = item / (8 * rows_per_expert);
  const size_t row = static_cast<size_t>(expert_start) * rows_per_expert + item % (8 * rows_per_expert);
  float sum = 0;
  for (int feature = threadIdx.x % 32; feature < reduction; feature += 32) {
    sum += __half2float(input[token * reduction + feature]) *
           Load<Block, Padded>(weights, scales, row, feature, reduction, stride);
  }
  for (int delta = 16; delta; delta /= 2) sum += __shfl_down_sync(0xffffffffu, sum, delta);
  if (threadIdx.x % 32 == 0) output[item] = __float2half(sum);
}

template <int Block, bool Padded>
void Launch(const std::string& operation, const uint8_t* weights, const half* scales,
            const half* input, half* output, int rows, int reduction, int stride,
            int expert_start, int tokens) {
  if (operation == "dequantize") {
    const size_t count = static_cast<size_t>(8) * rows * reduction;
    Dequantize<Block, Padded><<<(count + 255) / 256, 256>>>(weights, scales, output, rows, reduction, stride, expert_start);
  } else {
    const size_t count = static_cast<size_t>(tokens) * 8 * rows;
    Project<Block, Padded><<<(count + 3) / 4, 128>>>(weights, scales, input, output, rows, reduction, stride, expert_start, tokens);
  }
  Check(cudaGetLastError());
}

template <int Block>
void Dispatch(const std::string& operation, const uint8_t* weights, const half* scales,
              const half* input, half* output, int rows, int reduction, int stride,
              int expert_start, int tokens) {
  if (stride == Block * 3 / 8)
    Launch<Block, false>(operation, weights, scales, input, output, rows, reduction, stride, expert_start, tokens);
  else
    Launch<Block, true>(operation, weights, scales, input, output, rows, reduction, stride, expert_start, tokens);
}

template <typename Action>
float Time(Action action) {
  cudaEvent_t begin, end;
  Check(cudaEventCreate(&begin));
  Check(cudaEventCreate(&end));
  Check(cudaEventRecord(begin));
  action();
  Check(cudaEventRecord(end));
  Check(cudaEventSynchronize(end));
  float elapsed;
  Check(cudaEventElapsedTime(&elapsed, begin, end));
  Check(cudaEventDestroy(begin));
  Check(cudaEventDestroy(end));
  return elapsed;
}

void PrintSamples(const std::vector<float>& samples) {
  std::cout << '[';
  for (size_t index = 0; index < samples.size(); ++index) {
    if (index) std::cout << ',';
    std::cout << samples[index];
  }
  std::cout << ']';
}

template <int Block>
void Benchmark(const std::string& projection, int rows, int reduction, bool quick, bool& first) {
  const int experts = quick ? 16 : 128;
  const int prefill_tokens = quick ? 4 : 32;
  const int trials = quick ? 3 : 7;
  const size_t total_rows = static_cast<size_t>(experts) * rows;
  const size_t raw_bytes = total_rows * reduction * 3 / 8;
  const size_t scale_count = total_rows * (reduction / Block);
  DeviceBuffer<uint8_t> raw(raw_bytes);
  DeviceBuffer<half> scales(scale_count), input(prefill_tokens * reduction);
  Initialize<<<(raw_bytes + 255) / 256, 256>>>(raw.data, scales.data, input.data, total_rows, reduction, Block, prefill_tokens);
  Check(cudaDeviceSynchronize());
  const int alignments[] = {1, 16, 32};
  std::vector<DeviceBuffer<uint8_t>*> layouts;
  std::vector<int> strides;
  std::vector<size_t> byte_counts;
  std::vector<std::vector<float>> conversions(3);
  for (int alignment : alignments) {
    const int stride = ((Block * 3 / 8 + alignment - 1) / alignment) * alignment;
    strides.push_back(stride);
    const size_t count = total_rows * (reduction / Block) * stride;
    byte_counts.push_back(count);
    auto* buffer = new DeviceBuffer<uint8_t>(count);
    layouts.push_back(buffer);
    Repack<<<(count + 255) / 256, 256>>>(raw.data, buffer->data, count, reduction, Block, stride);
  }
  Check(cudaDeviceSynchronize());
  for (int trial = 0; trial < trials; ++trial) {
    for (int position = 0; position < 3; ++position) {
      const int layout = trial % 2 ? 2 - position : position;
      const size_t count = byte_counts[layout];
      conversions[layout].push_back(Time([&] {
        Repack<<<(count + 255) / 256, 256>>>(raw.data, layouts[layout]->data, count, reduction, Block, strides[layout]);
        Check(cudaGetLastError());
      }));
    }
  }
  for (const std::string operation : {"dequantize", "decode_projection", "prefill_projection"}) {
    const int tokens = operation == "prefill_projection" ? prefill_tokens : 1;
    const int iterations = quick ? 2 : (tokens == 1 ? 32 : 16);
    const size_t count = operation == "dequantize" ? static_cast<size_t>(8) * rows * reduction
                                                   : static_cast<size_t>(tokens) * 8 * rows;
    DeviceBuffer<half> output(count);
    std::vector<half> expected(count), actual(count);
    for (int expert_start : {0, experts - 8}) {
      Dispatch<Block>(operation, raw.data, scales.data, input.data, output.data, rows, reduction, Block * 3 / 8, expert_start, tokens);
      Check(cudaMemcpy(expected.data(), output.data, count * sizeof(half), cudaMemcpyDeviceToHost));
      for (int selected = 0; selected < 8; ++selected) {
        for (int row_index : {0, rows - 1}) {
          const size_t row = static_cast<size_t>(expert_start + selected) * rows + row_index;
          for (int token : {0, tokens - 1}) {
            float reference = 0;
            for (int feature = 0; feature < reduction; ++feature) {
              const float weight = (static_cast<int>(Code(row, feature)) - 4) * static_cast<float>(row % 7 + 1) / 128;
              if (operation == "dequantize") {
                const size_t index = static_cast<size_t>(selected * rows + row_index) * reduction + feature;
                if (__half2float(expected[index]) != weight) throw std::runtime_error("Scalar dequantization mismatch");
              } else {
                const int input_code = ((token * reduction + feature) * 13) % 16 - 8;
                reference += weight * static_cast<float>(input_code) / 16;
              }
            }
            if (operation != "dequantize") {
              const size_t index = static_cast<size_t>(token * 8 + selected) * rows + row_index;
              if (__half2float(expected[index]) != __half2float(__float2half(reference)))
                throw std::runtime_error("Scalar projection mismatch");
            }
          }
        }
      }
      for (int layout = 0; layout < 3; ++layout) {
        Dispatch<Block>(operation, layouts[layout]->data, scales.data, input.data, output.data, rows, reduction, strides[layout], expert_start, tokens);
        Check(cudaMemcpy(actual.data(), output.data, count * sizeof(half), cudaMemcpyDeviceToHost));
        for (size_t index = 0; index < count; ++index) {
          if (__half2float(actual[index]) != __half2float(expected[index]))
            throw std::runtime_error("Padding changed numerical output");
        }
      }
    }
    for (const std::string cache : {"fixed_8_experts", "rotating_128_experts"}) {
      std::vector<std::vector<float>> timings(3);
      for (int trial = 0; trial < trials; ++trial) {
        for (int position = 0; position < 3; ++position) {
          const int layout = trial % 2 ? 2 - position : position;
          auto launch = [&](int iteration) {
            const int start = cache == "fixed_8_experts" ? 0 : (iteration * 8) % experts;
            Dispatch<Block>(operation, layouts[layout]->data, scales.data, input.data, output.data, rows, reduction, strides[layout], start, tokens);
          };
          for (int iteration = 0; iteration < 3; ++iteration) launch(iteration);
          Check(cudaDeviceSynchronize());
          timings[layout].push_back(Time([&] {
                                      for (int iteration = 0; iteration < iterations; ++iteration) launch(iteration);
                                    }) /
                                    iterations);
        }
      }
      for (int layout = 0; layout < 3; ++layout) {
        if (!first) std::cout << ",\n";
        first = false;
        auto sorted = timings[layout];
        std::sort(sorted.begin(), sorted.end());
        std::cout << "{\"projection\":\"" << projection << "\",\"operation\":\"" << operation
                  << "\",\"access\":\"" << cache << "\",\"experts\":" << experts << ",\"selected_experts\":8"
                  << ",\"rows\":" << rows << ",\"k\":" << reduction << ",\"tokens\":" << tokens
                  << ",\"block_size\":" << Block << ",\"block_alignment_bytes\":" << alignments[layout]
                  << ",\"block_stride_bytes\":" << strides[layout] << ",\"weight_bytes\":" << byte_counts[layout]
                  << ",\"scale_bytes\":" << scale_count * sizeof(half) << ",\"iterations_per_trial\":" << iterations
                  << ",\"median_ms\":" << sorted[sorted.size() / 2] << ",\"trial_ms\":";
        PrintSamples(timings[layout]);
        std::cout << ",\"conversion_trial_ms\":";
        PrintSamples(conversions[layout]);
        std::cout << ",\"layout_bit_exact\":true,\"scalar_samples_passed\":true}";
      }
    }
  }
  for (auto* layout : layouts) delete layout;
}

int main(int argc, char** argv) {
  try {
    if (argc > 2 || (argc == 2 && std::string(argv[1]) != "--quick")) throw std::runtime_error("Usage: benchmark [--quick]");
    const bool quick = argc == 2;
    cudaDeviceProp properties;
    Check(cudaGetDeviceProperties(&properties, 0));
    int runtime_version, driver_version;
    Check(cudaRuntimeGetVersion(&runtime_version));
    Check(cudaDriverGetVersion(&driver_version));
    std::cout << std::fixed << std::setprecision(6) << "{\"gpu\":\"" << properties.name
              << "\",\"sm\":" << properties.major * 10 + properties.minor << ",\"l2_bytes\":" << properties.l2CacheSize
              << ",\"cuda_runtime\":" << runtime_version << ",\"cuda_driver\":" << driver_version
              << ",\"quick\":" << (quick ? "true" : "false") << ",\"results\":[\n";
    bool first = true;
    for (const std::string projection : {"fc1_gate_up", "fc2_down"}) {
      const int rows = quick ? 64 : (projection == "fc1_gate_up" ? 1536 : 2048);
      const int reduction = quick ? 128 : (projection == "fc1_gate_up" ? 2048 : 768);
      Benchmark<32>(projection, rows, reduction, quick, first);
      Benchmark<64>(projection, rows, reduction, quick, first);
      Benchmark<128>(projection, rows, reduction, quick, first);
    }
    std::cout << "\n]}\n";
  } catch (const std::exception& exception) {
    std::cerr << exception.what() << '\n';
    return EXIT_FAILURE;
  }
}
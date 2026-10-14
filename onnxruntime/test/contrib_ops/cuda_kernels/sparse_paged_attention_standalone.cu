// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "sparse_device.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

using namespace onnxruntime::contrib::cuda;

void Check(cudaError_t status) {
  if (status != cudaSuccess) {
    fprintf(stderr, "%s\n", cudaGetErrorString(status));
    std::exit(2);
  }
}

template <typename T>
T* Copy(const std::vector<T>& values, int offset = 0) {
  T* ptr;
  Check(cudaMalloc(&ptr, (values.size() + offset) * sizeof(T)));
  Check(cudaMemcpy(ptr + offset, values.data(), values.size() * sizeof(T), cudaMemcpyHostToDevice));
  return ptr + offset;
}

template <typename T>
int Run(int tokens, int selected, int splits, bool edges, bool misaligned) {
  constexpr int heads = 24, kv_heads = 2, dim = 256, length = 8192, block_size = 32;
  const int past = length - tokens;
  std::mt19937 random(881);
  std::normal_distribution<float> normal(0.0f, 0.5f);
  auto activation = [&](size_t size) {
    std::vector<T> data(size);
    for (auto& value : data) value = static_cast<T>(normal(random));
    return data;
  };
  auto q = activation(static_cast<size_t>(tokens) * heads * dim);
  auto k = activation(static_cast<size_t>(tokens) * kv_heads * dim);
  auto v = activation(k.size());
  auto kc = activation(static_cast<size_t>(length) * kv_heads * dim);
  auto vc = activation(kc.size());
  auto sink = activation(heads);
  std::vector<int> indices(static_cast<size_t>(tokens) * selected), counts(tokens, selected);
  for (int t = 0; t < tokens; ++t) {
    for (int i = 0; i < selected; ++i) {
      indices[static_cast<size_t>(t) * selected + i] = random() % (past + t + 1);
      if (edges && i % 11 == 0) indices[static_cast<size_t>(t) * selected + i] = -1;
      if (edges && i % 17 == 0) indices[static_cast<size_t>(t) * selected + i] = length + 3;
      if (edges && i % 23 == 0) indices[static_cast<size_t>(t) * selected + i] = past + t;
    }
    if (edges && t % 5 == 0) counts[t] = 0;
    if (edges && t % 5 == 1) counts[t] = 1;
    if (edges && t % 5 == 2) counts[t] = std::min(selected, 65);
  }
  std::vector<int> blocks(length / block_size), slots(tokens);
  for (int i = 0; i < static_cast<int>(blocks.size()); ++i) blocks[i] = i;
  for (int i = 0; i < tokens; ++i) slots[i] = past + i;
  if (edges) {
    blocks[2] = -1;
    if (tokens > 5) slots[5] = -1;
  }
  auto dq = Copy(q), dk = Copy(k, misaligned), dv = Copy(v, misaligned);
  auto dkc = Copy(kc, misaligned), dvc = Copy(vc, misaligned);
  auto dsink = Copy(sink);
  auto dcumulative = Copy(std::vector<int>{0, tokens}), dpast = Copy(std::vector<int>{past});
  auto dblocks = Copy(blocks), dslots = Copy(slots), dindices = Copy(indices), dcounts = Copy(counts);
  T* output;
  float *partial, *maxima, *sums;
  Check(cudaMalloc(&output, q.size() * sizeof(T)));
  Check(cudaMalloc(&partial, q.size() * splits * sizeof(float)));
  Check(cudaMalloc(&maxima, static_cast<size_t>(tokens) * heads * splits * sizeof(float)));
  Check(cudaMalloc(&sums, static_cast<size_t>(tokens) * heads * splits * sizeof(float)));
  cudaStream_t stream;
  Check(cudaStreamCreate(&stream));
  const bool grouped = std::getenv("SPARSE_BENCH_GROUPED") != nullptr;
  auto launch = [&](int variant) {
#define LAUNCH(H, QK, PV, ASYNC) \
    SparsePagedAttentionGqaTiledKernel<T, H, (H == 12 ? 4 : 1), QK, PV, ASYNC> \
        <<<dim3(heads / H, tokens, splits), (PV ? 4 : H) * 32, 0, stream>>>( \
            dq, dk, dv, dkc, dvc, dcumulative, dpast, dblocks, dslots, dindices, dcounts, \
            output, edges ? dsink : nullptr, partial, maxima, sums, 1, tokens, heads, kv_heads, \
            block_size, length / block_size, length / block_size, selected, kv_heads * dim, \
            kv_heads * dim, 0.0625f, edges ? 1.7f : 0.0f, true, splits)
    if (variant == 0) {
      LAUNCH(4, false, false, false);
    } else if (variant == 1) {
      if (grouped) {
        LAUNCH(12, false, false, false);
      } else {
        LAUNCH(12, true, false, false);
      }
    } else if (variant == 2) {
      if constexpr (std::is_same<T, half>::value) {
        if (splits == 1) {
          LAUNCH(12, true, true, false);
        } else {
          LAUNCH(12, true, false, false);
        }
      } else {
        LAUNCH(12, true, false, false);
      }
    } else {
      if constexpr (std::is_same<T, half>::value) {
        if (splits == 1) {
          LAUNCH(12, true, true, true);
        } else {
          LAUNCH(12, true, false, true);
        }
      } else {
        LAUNCH(12, true, false, true);
      }
    }
#undef LAUNCH
    if (splits > 1) {
      SparsePagedAttentionReduceKernel<T><<<dim3(heads, tokens), 128, 0, stream>>>(
          output, partial, maxima, sums, edges ? dsink : nullptr, tokens, heads, dim, splits);
    }
    Check(cudaGetLastError());
  };
  std::vector<T> reference(q.size()), tensor_reference(q.size()), actual(q.size());
  int failures = 0;
  const bool sanitizer = std::getenv("SPARSE_BENCH_SANITIZER") != nullptr;
  const int iterations = sanitizer ? 1 : 20;
  for (int variant = 0; variant < 4; ++variant) {
    for (int i = 0; i < (sanitizer ? 1 : 5); ++i) launch(variant);
    Check(cudaStreamSynchronize(stream));
    Check(cudaMemcpy(actual.data(), output, q.size() * sizeof(T), cudaMemcpyDeviceToHost));
    if (variant == 0) reference = actual;
    if (variant == 2) tensor_reference = actual;
    double max_abs = 0.0;
    size_t mismatches = 0, outside_tolerance = 0, scalar_mismatches = 0;
    for (size_t i = 0; i < q.size(); ++i) {
      float a = static_cast<float>(actual[i]), b = static_cast<float>(reference[i]);
      max_abs = std::max(max_abs, static_cast<double>(std::abs(a - b)));
      const float relative_tolerance = std::is_same<T, half>::value ? 0.002f : 0.008f;
      outside_tolerance += !std::isfinite(a) ||
                           std::abs(a - b) > 0.002f + relative_tolerance * std::abs(b);
      if (variant == 3) mismatches += std::memcmp(&actual[i], &tensor_reference[i], sizeof(T)) != 0;
      if (variant == 1 && grouped) scalar_mismatches += std::memcmp(&actual[i], &reference[i], sizeof(T)) != 0;
    }
    failures += outside_tolerance != 0 || mismatches != 0 || scalar_mismatches != 0;
    cudaGraph_t graph;
    cudaGraphExec_t executable;
    Check(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal));
    for (int i = 0; i < iterations; ++i) launch(variant);
    Check(cudaStreamEndCapture(stream, &graph));
    Check(cudaGraphInstantiate(&executable, graph, nullptr, nullptr, 0));
    cudaEvent_t start, end;
    Check(cudaEventCreate(&start));
    Check(cudaEventCreate(&end));
    printf("{\"tokens\":%d,\"selected\":%d,\"splits\":%d,\"bf16\":%s,\"edges\":%s,"
           "\"misaligned\":%s,\"variant\":%d,\"max_abs\":%.9g,\"outside_tolerance\":%zu,"
           "\"async_bit_mismatches\":%zu,\"scalar_bit_mismatches\":%zu,\"grouped\":%s,\"samples_us\":[",
           tokens, selected, splits, std::is_same<T, half>::value ? "false" : "true",
           edges ? "true" : "false", misaligned ? "true" : "false",
           variant, max_abs, outside_tolerance, mismatches, scalar_mismatches, grouped ? "true" : "false");
    for (int sample = 0; sample < (sanitizer ? 1 : 7); ++sample) {
      Check(cudaEventRecord(start, stream));
      Check(cudaGraphLaunch(executable, stream));
      Check(cudaEventRecord(end, stream));
      Check(cudaEventSynchronize(end));
      float milliseconds;
      Check(cudaEventElapsedTime(&milliseconds, start, end));
      printf("%s%.6f", sample ? "," : "", milliseconds * 1000.0 / iterations);
    }
    printf("]}\n");
    fflush(stdout);
    Check(cudaGraphExecDestroy(executable));
    Check(cudaGraphDestroy(graph));
    Check(cudaEventDestroy(start));
    Check(cudaEventDestroy(end));
  }
  for (void* ptr : {static_cast<void*>(dq), static_cast<void*>(dk - misaligned),
                    static_cast<void*>(dv - misaligned), static_cast<void*>(dkc - misaligned),
                    static_cast<void*>(dvc - misaligned), static_cast<void*>(dsink),
                    static_cast<void*>(dcumulative), static_cast<void*>(dpast),
                    static_cast<void*>(dblocks), static_cast<void*>(dslots),
                    static_cast<void*>(dindices), static_cast<void*>(dcounts),
                    static_cast<void*>(output), static_cast<void*>(partial),
                    static_cast<void*>(maxima), static_cast<void*>(sums)}) {
    Check(cudaFree(ptr));
  }
  Check(cudaStreamDestroy(stream));
  return failures ? 1 : 0;
}

int main(int argc, char** argv) {
  if (argc != 7) return 2;
  int tokens = atoi(argv[1]), selected = atoi(argv[2]), splits = atoi(argv[3]);
  if (tokens < 1 || tokens > 8192 || selected < 1 || splits < 1 || splits > 32) return 2;
  return atoi(argv[4]) ? Run<__nv_bfloat16>(tokens, selected, splits, atoi(argv[5]), atoi(argv[6]))
                       : Run<half>(tokens, selected, splits, atoi(argv[5]), atoi(argv[6]));
}

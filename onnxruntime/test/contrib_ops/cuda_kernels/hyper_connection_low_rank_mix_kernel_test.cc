// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cmath>
#include <cstdio>
#include <memory>
#include <vector>

#include "contrib_ops/cuda/bert/hyper_connection_low_rank_mix_impl.h"
#include "core/common/float16.h"
#include "core/providers/cuda/cuda_common.h"
#include "gtest/gtest.h"

namespace onnxruntime::test {
cudaError_t LaunchHcIndependentProducer(const uint16_t* source, uint16_t* destination,
                                       size_t count, cudaStream_t stream,
                                       const uint16_t* weight_source, uint16_t* weight_destination,
                                       size_t weight_count);
namespace {

namespace hc = contrib::cuda;

struct HcBuffer {
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(HcBuffer);

  explicit HcBuffer(size_t bytes) : bytes(bytes) {
    CUDA_CALL_THROW(cudaMalloc(&data, bytes));
  }
  ~HcBuffer() {
    if (cudaFree(data) != cudaSuccess) std::fprintf(stderr, "HC test cudaFree failed\n");
  }
  template <typename T>
  T* As() { return static_cast<T*>(data); }
  template <typename T>
  void Upload(const std::vector<T>& values) {
    ASSERT_EQ(values.size() * sizeof(T), bytes);
    ASSERT_EQ(cudaMemcpy(data, values.data(), bytes, cudaMemcpyHostToDevice), cudaSuccess);
  }
  std::vector<uint16_t> Download() {
    std::vector<uint16_t> result(bytes / sizeof(uint16_t));
    CUDA_CALL_THROW(cudaMemcpy(result.data(), data, bytes, cudaMemcpyDeviceToHost));
    return result;
  }
  void* data = nullptr;
  size_t bytes;
};

struct HcGraph {
  HcGraph() { CUDA_CALL_THROW(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
  ~HcGraph() {
    if (exec) (void)cudaGraphExecDestroy(exec);
    if (graph) (void)cudaGraphDestroy(graph);
    (void)cudaStreamDestroy(stream);
  }
  cudaStream_t stream = nullptr;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t exec = nullptr;
};

template <typename T>
std::vector<T> HcData(size_t size, float scale, float bias = 0.0f, int seed = 0) {
  std::vector<T> result(size);
  for (size_t i = 0; i < size; ++i) {
    result[i] = T(bias + scale * std::sin(static_cast<float>((i * 73 + seed) % 1009)));
  }
  return result;
}

enum class HcDispatch { Automatic, NoCooperativeLaunch, InsufficientResidency };

// Compare to the original two-kernel arithmetic bit-for-bit, with fresh inputs and dirty scratch
// on every replay. The OpTester tests independently check the schema's CPU reference.
template <typename HostT = BFloat16, typename CudaT = __nv_bfloat16>
void RunHcGraphParity(int rows, int branches, int hidden, int rank, bool prologue, HcDispatch dispatch,
                      bool dynamic_weight = false, bool packed_weights = false) {
  static_assert(sizeof(HostT) == 2 && sizeof(CudaT) == 2);
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  cudaDeviceProp prop{};
  ASSERT_EQ(cudaGetDeviceProperties(&prop, device), cudaSuccess);
  if (prop.major < 8) GTEST_SKIP() << "Fused HC requires SM80+";

  hc::HyperConnectionLowRankMixConfig config;
  config.rows = rows;
  config.branches = branches;
  config.hidden = hidden;
  config.rank = rank;
  config.epsilon = 1e-6f;
  config.silu_alpha = 0.25f;
  config.reduction_scale = 0.25f;
  config.gate_div = 4.0f;
  config.gate_mul = 2.0f;
  config.weights_ready = !dynamic_weight;
  ASSERT_TRUE(hc::CanUseFusedHyperConnectionLowRankMix<CudaT>(prop, config));
  const size_t k = static_cast<size_t>(branches) * hidden;
  HcBuffer source(rows * k * 2), streams(rows * k * 2), norm(k * 2), down(k * rank * 2), up(rank * k * 2),
      inject(k * branches * 2), prev_out(rows * hidden * 2), prev_gate(rows * branches * 2);
  HcBuffer pre(rows * hidden * 2), gate(rows * branches * 2), streams_out(rows * k * 2),
      ref_pre(rows * hidden * 2), ref_gate(rows * branches * 2), ref_streams(rows * k * 2);
  HcBuffer scratch(hc::HyperConnectionLowRankMixFusedScratchElements(config) * sizeof(float));
  HcBuffer down_source(k * rank * 2);
  norm.Upload(HcData<HostT>(k, 0.1f, 1.0f));
  down.Upload(HcData<HostT>(k * rank, 1.0f / std::sqrt(static_cast<float>(k))));
  up.Upload(HcData<HostT>(rank * k, 2.0f / std::sqrt(static_cast<float>(rank))));
  inject.Upload(HcData<HostT>(k * branches, 1.0f / std::sqrt(static_cast<float>(k))));
  prev_out.Upload(HcData<HostT>(rows * hidden, 1.0f));
  prev_gate.Upload(HcData<HostT>(rows * branches, 0.5f, 1.0f));
  streams.Upload(HcData<HostT>(rows * k, 1.0f));

  hc::HyperConnectionLowRankMixTensors<CudaT> tensors;
  tensors.streams = streams.As<CudaT>();
  tensors.norm_scale = norm.As<CudaT>();
  tensors.down_weight = down.As<CudaT>();
  tensors.up_weight = up.As<CudaT>();
  tensors.inject_weight = inject.As<CudaT>();
  tensors.prev_block_output = prologue ? prev_out.As<CudaT>() : nullptr;
  tensors.prev_post_gate = prologue ? prev_gate.As<CudaT>() : nullptr;
  tensors.pre_output = pre.As<CudaT>();
  tensors.post_gate = gate.As<CudaT>();
  tensors.streams_out = streams_out.As<CudaT>();
  std::unique_ptr<HcBuffer> packed_down, packed_up;
  if (packed_weights) {
    ASSERT_EQ(branches, 4);
    ASSERT_EQ(hidden, 2560);
    ASSERT_EQ(rank, 320);
    ASSERT_FALSE(dynamic_weight);
    packed_down = std::make_unique<HcBuffer>(hc::kHyperConnectionPackedWeightBytes);
    packed_up = std::make_unique<HcBuffer>(hc::kHyperConnectionPackedWeightBytes);
    ASSERT_TRUE(hc::LaunchPackHyperConnectionWeight(packed_down->data, down.data, false, nullptr).IsOK());
    ASSERT_TRUE(hc::LaunchPackHyperConnectionWeight(packed_up->data, up.data, true, nullptr).IsOK());
    ASSERT_EQ(cudaDeviceSynchronize(), cudaSuccess);
    tensors.packed_down_weight = packed_down->As<CudaT>();
    tensors.packed_up_weight = packed_up->As<CudaT>();
  }
  auto reference = tensors;
  reference.packed_down_weight = nullptr;
  reference.packed_up_weight = nullptr;
  reference.pre_output = ref_pre.As<CudaT>();
  reference.post_gate = ref_gate.As<CudaT>();
  reference.streams_out = ref_streams.As<CudaT>();
  auto fallback_prop = prop;
  fallback_prop.cooperativeLaunch = 0;
  auto selected_prop = prop;
  if (dispatch == HcDispatch::NoCooperativeLaunch) selected_prop.cooperativeLaunch = 0;
  if (dispatch == HcDispatch::InsufficientResidency) selected_prop.multiProcessorCount = 0;

  HcGraph graph;
  // Warm both instantiations before capture (also covers ordinary repeated launches).
  for (const auto& p : {fallback_prop, selected_prop}) {
    ASSERT_TRUE(hc::LaunchHyperConnectionLowRankMixFused(
                    p, graph.stream, config, tensors, scratch.As<float>()).IsOK());
  }
  ASSERT_EQ(cudaStreamSynchronize(graph.stream), cudaSuccess);
  ASSERT_EQ(cudaStreamBeginCapture(graph.stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
  const auto producer_status = LaunchHcIndependentProducer(
      source.As<uint16_t>(), streams.As<uint16_t>(), rows * k, graph.stream,
      down_source.As<uint16_t>(), down.As<uint16_t>(), dynamic_weight ? k * rank : 0);
  const auto status = hc::LaunchHyperConnectionLowRankMixFused(
      selected_prop, graph.stream, config, tensors, scratch.As<float>());
  const auto capture_status = cudaStreamEndCapture(graph.stream, &graph.graph);
  ASSERT_EQ(producer_status, cudaSuccess);
  ASSERT_TRUE(status.IsOK()) << status.ErrorMessage();
  ASSERT_EQ(capture_status, cudaSuccess);
  size_t nodes = 0;
  ASSERT_EQ(cudaGraphGetNodes(graph.graph, nullptr, &nodes), cudaSuccess);
  if (dispatch != HcDispatch::Automatic || !prop.cooperativeLaunch || rows > 2) {
    EXPECT_EQ(nodes, 3u);
  } else if (branches == 2 && hidden == 64 && rank == 16 && prop.multiProcessorCount >= 4) {
    // This four-CTA grid fits even at one CTA/SM: ensure automatic selection really fuses.
    EXPECT_EQ(nodes, 2u);
  } else {
    // Automatic fallback on a smaller device is intentional, not a test failure.
    EXPECT_TRUE(nodes == 2 || nodes == 3) << nodes;
  }
  ASSERT_EQ(cudaGraphInstantiateWithFlags(&graph.exec, graph.graph, 0), cudaSuccess);
  for (int replay = 0; replay < 4; ++replay) {
    SCOPED_TRACE(replay);
    streams.Upload(HcData<HostT>(rows * k, 1.0f, 0.0f, replay * 17));
    source.Upload(HcData<HostT>(rows * k, 1.0f, 0.0f, replay * 17));
    if (dynamic_weight) {
      const auto weights = HcData<HostT>(k * rank, 1.0f / std::sqrt(static_cast<float>(k)),
                                        0.0f, 23 + replay * 31);
      down.Upload(weights);
      down_source.Upload(weights);
    }
    ASSERT_TRUE(hc::LaunchHyperConnectionLowRankMixFused(
                    fallback_prop, graph.stream, config, reference, scratch.As<float>()).IsOK());
    ASSERT_EQ(cudaStreamSynchronize(graph.stream), cudaSuccess);
    const auto expected_pre = ref_pre.Download();
    const auto expected_gate = ref_gate.Download();
    const auto expected_streams = ref_streams.Download();
    ASSERT_EQ(cudaMemsetAsync(streams.data, 0xff, streams.bytes, graph.stream), cudaSuccess);
    if (dynamic_weight) {
      ASSERT_EQ(cudaMemsetAsync(down.data, 0xff, down.bytes, graph.stream), cudaSuccess);
    }
    ASSERT_EQ(cudaMemsetAsync(scratch.data, 0xff, scratch.bytes, graph.stream), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(pre.data, 0xff, pre.bytes, graph.stream), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(gate.data, 0xff, gate.bytes, graph.stream), cudaSuccess);
    ASSERT_EQ(cudaMemsetAsync(streams_out.data, 0xff, streams_out.bytes, graph.stream), cudaSuccess);
    ASSERT_EQ(cudaGraphLaunch(graph.exec, graph.stream), cudaSuccess);
    ASSERT_EQ(cudaStreamSynchronize(graph.stream), cudaSuccess);
    EXPECT_EQ(pre.Download(), expected_pre);
    EXPECT_EQ(gate.Download(), expected_gate);
    EXPECT_EQ(streams_out.Download(), expected_streams);
  }
}

TEST(HyperConnectionLowRankMixCudaKernelTest, DecodeBFloat16GraphReplay) {
  for (int rows : {1, 2}) {
    for (bool prologue : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "rows=" << rows << " prologue=" << prologue);
      RunHcGraphParity(rows, 4, 2560, 320, prologue, HcDispatch::Automatic);
    }
  }
}

TEST(HyperConnectionLowRankMixCudaKernelTest, GenericBFloat16GraphReplay) {
  for (int rows : {1, 2}) {
    for (bool prologue : {false, true}) {
      RunHcGraphParity(rows, 2, 64, 16, prologue, HcDispatch::Automatic);
      RunHcGraphParity(rows, 8, 1288, 40, prologue, HcDispatch::Automatic);
    }
  }
}

TEST(HyperConnectionLowRankMixCudaKernelTest, DecodeFloat16GraphReplay) {
  for (int rows : {1, 2}) {
    for (bool prologue : {false, true}) {
      SCOPED_TRACE(::testing::Message() << "rows=" << rows << " prologue=" << prologue);
      RunHcGraphParity<MLFloat16, __half>(rows, 4, 2560, 320, prologue, HcDispatch::Automatic);
    }
  }
}

TEST(HyperConnectionLowRankMixCudaKernelTest, DynamicWeightGraphReplay) {
  for (int rows : {1, 2}) {
    for (bool prologue : {false, true}) {
      for (auto dispatch : {HcDispatch::Automatic, HcDispatch::NoCooperativeLaunch}) {
        RunHcGraphParity(rows, 4, 2560, 320, prologue, dispatch, true);
        RunHcGraphParity<MLFloat16, __half>(rows, 4, 2560, 320, prologue, dispatch, true);
      }
    }
  }
}

TEST(HyperConnectionLowRankMixCudaKernelTest, PackedWeightGraphReplay) {
  for (int rows : {1, 2, 3, 4}) {
    for (bool prologue : {false, true}) {
      for (auto dispatch : {HcDispatch::Automatic, HcDispatch::NoCooperativeLaunch,
                            HcDispatch::InsufficientResidency}) {
        RunHcGraphParity(rows, 4, 2560, 320, prologue, dispatch, false, true);
        RunHcGraphParity<MLFloat16, __half>(rows, 4, 2560, 320, prologue, dispatch, false, true);
      }
    }
  }
}

TEST(HyperConnectionLowRankMixCudaKernelTest, PackedWeightLayout) {
  std::vector<uint16_t> source(hc::kHyperConnectionPackedWeightBytes / sizeof(uint16_t));
  for (size_t i = 0; i < source.size(); ++i) source[i] = static_cast<uint16_t>(i * 73 + i / 65536);
  HcBuffer input(hc::kHyperConnectionPackedWeightBytes), output(hc::kHyperConnectionPackedWeightBytes);
  ASSERT_EQ(cudaMemcpy(input.data, source.data(), input.bytes, cudaMemcpyHostToDevice), cudaSuccess);
  for (bool up : {false, true}) {
    ASSERT_TRUE(hc::LaunchPackHyperConnectionWeight(output.data, input.data, up, nullptr).IsOK());
    const auto packed = output.Download();
    for (int tile = 0; tile < 160; ++tile) {
      for (int q = 0; q < 2560; ++q) {
        const int i = q / 256, tid = q % 256;
        const size_t offset = up
            ? (tid / 8 + 32 * i) * 10240 + ((tid % 8) / 2) * 2560 + tile * 16 + (tid % 2) * 8
            : ((tile / 20) * 1280 + tid / 2 + 128 * i) * 320 + (tile % 20) * 16 + (tid % 2) * 8;
        for (int m = 0; m < 8; ++m) {
          ASSERT_EQ(packed[(tile * 2560 + q) * 8 + m], source[offset + m]);
        }
      }
    }
  }
}

TEST(HyperConnectionLowRankMixCudaKernelTest, GenericFloat16GraphReplay) {
  for (int rows : {1, 2}) {
    for (bool prologue : {false, true}) {
      RunHcGraphParity<MLFloat16, __half>(rows, 2, 64, 16, prologue, HcDispatch::Automatic);
      RunHcGraphParity<MLFloat16, __half>(rows, 8, 1288, 40, prologue, HcDispatch::Automatic);
      RunHcGraphParity<MLFloat16, __half>(rows, 4, 64, 16, prologue, HcDispatch::NoCooperativeLaunch);
      RunHcGraphParity<MLFloat16, __half>(rows, 4, 64, 16, prologue, HcDispatch::InsufficientResidency);
    }
  }
}

TEST(HyperConnectionLowRankMixCudaKernelTest, TwoKernelFallbackGraphReplay) {
  for (bool prologue : {false, true}) {
    RunHcGraphParity(2, 4, 64, 16, prologue, HcDispatch::NoCooperativeLaunch);
    RunHcGraphParity(1, 4, 64, 16, prologue, HcDispatch::InsufficientResidency);
    RunHcGraphParity(3, 4, 64, 16, prologue, HcDispatch::Automatic);
  }
}

}  // namespace
}  // namespace onnxruntime::test

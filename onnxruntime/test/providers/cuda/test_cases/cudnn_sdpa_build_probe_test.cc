// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

#include <cuda_runtime.h>
#include <cudnn.h>

#include "contrib_ops/cuda/bert/cudnn_fmha/cudnn_flash_attention.h"
#include "gtest/gtest.h"

#if CUDNN_MAJOR >= 9

namespace onnxruntime {
namespace test {

namespace {

class CudnnHandle {
 public:
  CudnnHandle() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) == cudaSuccess && device_count > 0 &&
        cudnnCreate(&handle_) != CUDNN_STATUS_SUCCESS) {
      handle_ = nullptr;
    }
  }
  ~CudnnHandle() {
    if (handle_ != nullptr) {
      cudnnDestroy(handle_);
    }
  }
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(CudnnHandle);
  cudnnHandle_t get() const { return handle_; }

 private:
  cudnnHandle_t handle_ = nullptr;
};

// Causal with s_q > s_kv and no padding mask selects cuDNN's bottom-right causal mask, which the
// cuDNN frontend rejects in validate() on every backend version and device.
constexpr int kBatchSize = 1;
constexpr int kNumHeadsQ = 8;
constexpr int kNumHeadsKv = 2;
constexpr int kHeadSize = 64;
constexpr int kSeqLenQ = 128;
constexpr int kSeqLenKv = 64;
constexpr float kScale = 0.125f;

bool TryBuildRejectedGraph(cudnnHandle_t handle) {
  return onnxruntime::cudnn_sdpa::try_build_graph(
      kBatchSize, kNumHeadsQ, kNumHeadsKv, kHeadSize, kHeadSize, kSeqLenQ, kSeqLenKv, kScale,
      /*is_causal=*/true, /*is_bf16=*/false, /*has_bias=*/false,
      /*broadcast_attn_bias_dim_0=*/false, /*broadcast_attn_bias_dim_1=*/false,
      /*has_padding_mask=*/false, /*sliding_window=*/0,
      AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH, handle, /*stream=*/nullptr);
}

}  // namespace

TEST(CudnnSdpaBuildProbeTest, RejectedGraphReturnsFalseWithoutThrowing) {
  CudnnHandle handle;
  if (handle.get() == nullptr) {
    GTEST_SKIP() << "No CUDA device or cuDNN handle available.";
  }

  EXPECT_FALSE(TryBuildRejectedGraph(handle.get()));
  // The rejection is cached, so a repeat probe gives the same answer.
  EXPECT_FALSE(TryBuildRejectedGraph(handle.get()));
}

TEST(CudnnSdpaBuildProbeTest, RunThrowsOnRejectedGraph) {
  CudnnHandle handle;
  if (handle.get() == nullptr) {
    GTEST_SKIP() << "No CUDA device or cuDNN handle available.";
  }

  ASSERT_FALSE(TryBuildRejectedGraph(handle.get()));

  // run() fails on the graph lookup, before it touches any buffer, so null pointers are fine here.
  EXPECT_THROW(onnxruntime::cudnn_sdpa::run(
                   /*output=*/nullptr, /*q=*/nullptr, /*k=*/nullptr, /*v=*/nullptr, /*bias=*/nullptr,
                   /*mask_sequence_lengths_q=*/nullptr, /*mask_sequence_lengths_kv=*/nullptr,
                   kBatchSize, kNumHeadsQ, kNumHeadsKv, kHeadSize, kHeadSize, kSeqLenQ, kSeqLenKv, kScale,
                   /*is_causal=*/true, /*is_bf16=*/false,
                   /*broadcast_attn_bias_dim_0=*/false, /*broadcast_attn_bias_dim_1=*/false,
                   /*sliding_window=*/0, AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH, handle.get(),
                   /*stream=*/nullptr, /*allocator=*/nullptr),
               OnnxRuntimeException);
}

}  // namespace test
}  // namespace onnxruntime

#endif  // CUDNN_MAJOR >= 9

#endif  // !defined(USE_CUDA_MINIMAL) && !defined(DISABLE_CONTRIB_OPS) && !defined(BUILD_CUDA_EP_AS_PLUGIN)

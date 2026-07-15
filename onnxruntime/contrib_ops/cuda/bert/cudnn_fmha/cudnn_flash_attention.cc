// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/cudnn_fmha/cudnn_flash_attention.h"
#include <memory>
#include <vector>
#include <unordered_map>
#include <cudnn.h>

#if CUDNN_MAJOR < 9
namespace onnxruntime::cudnn_sdpa {

bool is_stable() {
  return false;
}

bool is_supported(const cudaDeviceProp& /*dprops*/,
                  int /*num_heads_q*/,
                  int /*num_heads_kv*/,
                  int /*head_size_qk*/,
                  int /*head_size_v*/,
                  int /*sequence_length_q*/,
                  int /*sequence_length_kv*/,
                  bool /*is_causal*/) {
  return false;
}

void run(
    void* /*output*/,
    void* /*q*/,
    void* /*k*/,
    void* /*v*/,
    void* /*bias*/,
    int* /*mask_sequence_lengths_q*/,
    int* /*mask_sequence_lengths_kv*/,
    int /*batch_size*/,
    int /*num_heads_q*/,
    int /*num_heads_kv*/,
    int /*head_size_qk*/,
    int /*head_size_v*/,
    int /*sequence_length_q*/,
    int /*sequence_length_kv*/,
    float /*scale*/,
    bool /*is_causal*/,
    bool /*is_bf16*/,
    bool /*broadcast_attn_bias_dim_0*/,
    bool /*broadcast_attn_bias_dim_1*/,
    int /*sliding_window*/,
    AttentionQkvFormat /*qkv_format*/,
    cudnnHandle_t /*handle*/,
    Stream* /*stream*/,
    AllocatorPtr /*allocator*/) {
  ORT_THROW("OnnxRuntime was not compiled with cuDNN Flash Attention.");
}

}  // namespace onnxruntime::cudnn_sdpa

#else  // CUDNN_MAJOR >= 9

#include <cudnn_frontend.h>
#include "core/providers/cuda/shared_inc/cudnn_fe_call.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cuda_stream_handle.h"

namespace onnxruntime::cudnn_sdpa {

bool is_stable() {
  // FP16/BF16 Flash Attention support in CUDNN backend:
  // version 8903 (8.9.3):
  //    Padding mask and causal mask
  //    Additive bias
  //    Multi-query attention (h_kv=1)
  //    Both self attention and cross attention
  //    (padded) variable sequence length
  //    Head dimensions 64 or 128
  // version 8903 (8.9.4):
  //    Alibi mask;
  // version 8907 (8.9.7):
  //    Grouped Query Attention
  // version 90100 (9.1.0):
  //    Head dimensions 256
  // version 90101 (9.1.1)
  //    Sliding window attention
  // version 90300 (9.3.0)
  //    Bug fixes; Variable sequence length supports zero-sequence-length values
  // version 90600 (9.6.0)
  //    Bottom-right causal mask no longer requires sequence lengths to be multiples of 64
  // For more information, please refer to cuDNN release notes, and the following links:
  //    https://docs.nvidia.com/deeplearning/cudnn/latest/developer/graph-api.html#fused-flash-attention-fprop
  //    https://github.com/NVIDIA/cudnn-frontend/blob/v1.24.0/docs/operations/Attention.md

  // For cuDNN version < 9.3, we will disable it by default.
  const size_t version = cudnnGetVersion();

  // cuDNN 9.10.0 and 9.10.1 have known bugs in the FP16/BF16 SDPA forward kernels, so skip them.
  if (version == 91000 || version == 91001) {
    return false;
  }

  return version >= 90300;
}

namespace fe = cudnn_frontend;

bool is_supported(const cudaDeviceProp& dprops,
                  int num_heads_q,
                  int num_heads_kv,
                  int head_size_qk,
                  int head_size_v,
                  int sequence_length_q,
                  int sequence_length_kv,
                  bool is_causal) {
  if (dprops.major < 8 ||
      (head_size_qk % 8 != 0) || (head_size_qk > 256) ||
      (head_size_v % 8 != 0) || (head_size_v > 256) ||
      (num_heads_kv == 0) || (num_heads_q % num_heads_kv != 0)) {
    return false;
  }

  // For a single query token (s_q == 1, e.g. decode) causal masking is a no-op: the token attends to
  // every key up to its own position, which the padding / kv sequence length already bounds. cuDNN is
  // therefore called without a causal mask in that case (see run()), so the causal-specific support
  // restrictions below do not apply.
  if (is_causal && sequence_length_q > 1) {
    // cuDNN expresses causal masking through diagonal alignment (cudnn_frontend >= 1.24):
    //   * s_q == s_kv : top-left aligned mask (standard self-attention causal, e.g. prefill).
    //   * s_q  < s_kv : bottom-right aligned mask (decode / cross attention with past KV).
    //   * s_q  > s_kv : not supported by cuDNN bottom-right causal masking.
    if (sequence_length_q > sequence_length_kv) {
      return false;
    }

    // Bottom-right causal masking requires s_q and s_kv to be multiples of 64 on cuDNN < 9.6.0.
    // Top-left causal masking (s_q == s_kv) has no such restriction.
    if (sequence_length_q != sequence_length_kv &&
        cudnnGetVersion() < 90600 &&
        (sequence_length_q % 64 != 0 || sequence_length_kv % 64 != 0)) {
      return false;
    }
  }

  return true;
}

// A helper function to set stride for q, k, v or output tensor.
// Strides are calculated based on logical tensor layout BNSH (batch_size, num_heads, sequence_length, head_size).
// The physical tensor layout could be either BSNH (is_bsnh=True) or BNSH (is_bsnh=False).
inline void set_stride(std::vector<int64_t>& stride,
                       int64_t num_heads,
                       int64_t sequence_length,
                       int64_t head_size,
                       bool is_bsnh) {
  stride = {num_heads * sequence_length * head_size,              // stride for batch.
            is_bsnh ? head_size : (head_size * sequence_length),  // stride for head.
            is_bsnh ? (num_heads * head_size) : head_size,        // stride for sequence.
            1};                                                   // stride for hidden dim of head, shall always be 1.
}

// It is used as a key for hash table to store cached graphs.
// It contains all parameters used in builing graph. Do not include data pointers that only needed in graph execution.
struct GraphParams {
  int batch_size;
  int num_heads_q;
  int num_heads_kv;
  int head_size_qk;
  int head_size_v;
  int sequence_length_q;
  int sequence_length_kv;
  float scale;
  bool is_causal;
  bool is_bf16;  // True if bfloat16, otherwise float16
  AttentionQkvFormat qkv_format;
  cudnnHandle_t handle;
  bool has_bias;
  bool broadcast_bias_dim_0;
  bool broadcast_bias_dim_1;
  bool has_padding_mask_q;
  bool has_padding_mask_kv;
  int sliding_window;

  bool operator==(const GraphParams& rhs) const {
    return batch_size == rhs.batch_size &&
           num_heads_q == rhs.num_heads_q &&
           num_heads_kv == rhs.num_heads_kv &&
           head_size_qk == rhs.head_size_qk &&
           head_size_v == rhs.head_size_v &&
           sequence_length_q == rhs.sequence_length_q &&
           sequence_length_kv == rhs.sequence_length_kv &&
           scale == rhs.scale &&
           is_causal == rhs.is_causal &&
           is_bf16 == rhs.is_bf16 &&
           qkv_format == rhs.qkv_format &&
           handle == rhs.handle &&
           has_bias == rhs.has_bias &&
           broadcast_bias_dim_0 == rhs.broadcast_bias_dim_0 &&
           broadcast_bias_dim_1 == rhs.broadcast_bias_dim_1 &&
           has_padding_mask_q == rhs.has_padding_mask_q &&
           has_padding_mask_kv == rhs.has_padding_mask_kv &&
           sliding_window == rhs.sliding_window;
  }
};

#define Q_UID 1
#define K_UID 2
#define V_UID 3
#define O_UID 4
#define BIAS_UID 5
#define SEQ_LEN_Q_UID 6
#define SEQ_LEN_KV_UID 7

std::shared_ptr<fe::graph::Graph> build_graph(GraphParams& params) {
  int batch_size = params.batch_size;
  int num_heads_q = params.num_heads_q;
  int num_heads_kv = params.num_heads_kv;
  int head_size_qk = params.head_size_qk;
  int head_size_v = params.head_size_v;
  int sequence_length_q = params.sequence_length_q;
  int sequence_length_kv = params.sequence_length_kv;
  float scale = params.scale;
  bool is_causal = params.is_causal;
  bool is_bf16 = params.is_bf16;
  AttentionQkvFormat qkv_format = params.qkv_format;
  cudnnHandle_t handle = params.handle;

  assert(qkv_format == contrib::AttentionQkvFormat::Q_K_V_BSNH ||
         qkv_format == contrib::AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH ||
         qkv_format == contrib::AttentionQkvFormat::Q_K_V_BNSH);

  auto mha_graph = std::make_shared<fe::graph::Graph>();
  mha_graph->set_io_data_type(is_bf16 ? fe::DataType_t::BFLOAT16 : fe::DataType_t::HALF)
      .set_intermediate_data_type(fe::DataType_t::FLOAT)
      .set_compute_data_type(fe::DataType_t::FLOAT);

  bool is_q_bsnh = (qkv_format == contrib::AttentionQkvFormat::Q_K_V_BSNH ||
                    qkv_format == contrib::AttentionQkvFormat::Q_K_V_BSNH_BNSH_BNSH);
  bool is_kv_bsnh = qkv_format == contrib::AttentionQkvFormat::Q_K_V_BSNH;

  std::vector<int64_t> stride;
  set_stride(stride, num_heads_q, sequence_length_q, head_size_qk, is_q_bsnh);

  auto Q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("Q")
                                 .set_uid(Q_UID)
                                 .set_dim({batch_size, num_heads_q, sequence_length_q, head_size_qk})  // logical layout
                                 .set_stride(stride));

  set_stride(stride, num_heads_kv, sequence_length_kv, head_size_qk, is_kv_bsnh);
  auto K = mha_graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("K")
                                 .set_uid(K_UID)
                                 .set_dim({batch_size, num_heads_kv, sequence_length_kv, head_size_qk})
                                 .set_stride(stride));

  set_stride(stride, num_heads_kv, sequence_length_kv, head_size_v, is_kv_bsnh);
  auto V = mha_graph->tensor(fe::graph::Tensor_attributes()
                                 .set_name("V")
                                 .set_uid(V_UID)
                                 .set_dim({batch_size, num_heads_kv, sequence_length_kv, head_size_v})
                                 .set_stride(stride));

  auto attributes = fe::graph::SDPA_attributes()
                        .set_name("SDPA")
                        .set_generate_stats(false)
                        .set_attn_scale(scale);

  if (is_causal) {
    // Use diagonal-alignment based causal masking (cudnn_frontend >= 1.24). A right bound of 0 keeps
    // only the lower-triangular region. Standard self-attention (s_q == s_kv) is top-left aligned;
    // decode / cross attention (s_q < s_kv) is bottom-right aligned so the query rows line up with the
    // most recent keys.
    attributes.set_diagonal_alignment(sequence_length_q != sequence_length_kv
                                          ? fe::DiagonalAlignment_t::BOTTOM_RIGHT
                                          : fe::DiagonalAlignment_t::TOP_LEFT)
        .set_diagonal_band_right_bound(0);
  }

  if (params.sliding_window > 0) {
    // Sliding window length maps to the left bound of the attention diagonal band.
    attributes.set_diagonal_band_left_bound(params.sliding_window);
  }

  if (params.has_bias) {
    std::vector<int64_t> bias_shape = {params.broadcast_bias_dim_0 ? 1 : batch_size,
                                       params.broadcast_bias_dim_1 ? 1 : num_heads_q,
                                       sequence_length_q,
                                       sequence_length_kv};
    stride = {bias_shape[1] * bias_shape[2] * bias_shape[3], bias_shape[2] * bias_shape[3], bias_shape[3], 1};
    auto bias = mha_graph->tensor(fe::graph::Tensor_attributes()
                                      .set_name("bias")
                                      .set_uid(BIAS_UID)
                                      .set_dim(bias_shape)
                                      .set_stride(stride));
    attributes.set_bias(bias);
  }

  if (params.has_padding_mask_q || params.has_padding_mask_kv) {
    attributes.set_padding_mask(true);

    if (params.has_padding_mask_q) {
      auto seq_q = mha_graph->tensor(fe::graph::Tensor_attributes()
                                         .set_name("seq_q")
                                         .set_uid(SEQ_LEN_Q_UID)
                                         .set_dim({batch_size, 1, 1, 1})
                                         .set_stride({1, 1, 1, 1})
                                         .set_data_type(fe::DataType_t::INT32));
      attributes.set_seq_len_q(seq_q);
    }

    if (params.has_padding_mask_kv) {
      auto seq_kv = mha_graph->tensor(fe::graph::Tensor_attributes()
                                          .set_name("seq_kv")
                                          .set_uid(SEQ_LEN_KV_UID)
                                          .set_dim({batch_size, 1, 1, 1})
                                          .set_stride({1, 1, 1, 1})
                                          .set_data_type(fe::DataType_t::INT32));
      attributes.set_seq_len_kv(seq_kv);
    }
  }

  auto [O, Stats] = mha_graph->sdpa(Q, K, V, attributes);

  constexpr bool is_output_bsnh = true;
  set_stride(stride, num_heads_q, sequence_length_q, head_size_v, is_output_bsnh);

  O->set_output(true)
      .set_dim({batch_size, num_heads_q, sequence_length_q, head_size_v})
      .set_stride(stride)
      .set_uid(O_UID);

  if (!mha_graph->build(handle, {fe::HeurMode_t::A}).is_good()) {
    ORT_THROW("Failed to build cuDNN graph for Flash Attention:", *mha_graph, "cudnn version:", cudnnGetVersion());
  }

  return mha_graph;
}

// Compute hash based on content in memory byte by byte. This can be moved to a common header file if needed.
template <typename T>
struct BytesHash {
  // Verify that Params is good to hash byte by byte.
  static_assert(std::is_standard_layout_v<T>, "Params is not standard layout");

  size_t operator()(const T& params) const {
    auto ptr = reinterpret_cast<const uint8_t*>(&params);
    // Fowler–Noll–Vo hash function
    uint32_t value = 0x811C9DC5;
    constexpr size_t bytes = sizeof(T);
    for (size_t i = 0; i < bytes; ++i) {
      value ^= ptr[i];
      value *= 0x01000193;
    }
    return static_cast<size_t>(value);
  }
};

// Use thread local caches because cuDNN execution plans are not guaranteed to be thread safe.
// TODO(tianleiwu): since the key includes sequence lengths, we may want to limit the cache size.
thread_local std::unordered_map<GraphParams, std::shared_ptr<fe::graph::Graph>, BytesHash<GraphParams> > mha_graph_cache;

// Allocate a device buffer of shape [batch_size] filled with a constant sequence length.
// Used to synthesize a no-op padding mask for one side when cuDNN requires both seq_len_q and
// seq_len_kv to be set (cudnn_frontend validates that both are present when padding mask is on).
// The buffer is filled with a stream-ordered Fill kernel (rather than a synchronous cudaMemcpy)
// so this path is safe to capture into a CUDA graph.
static IAllocatorUniquePtr<int> CreateConstantSeqLenBuffer(AllocatorPtr allocator,
                                                           Stream* stream,
                                                           int batch_size,
                                                           int value) {
  IAllocatorUniquePtr<int> buffer =
      IAllocator::MakeUniquePtr<int>(allocator, static_cast<size_t>(batch_size), false, stream);
  cudaStream_t cuda_stream = stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr;
  onnxruntime::cuda::Fill<int>(cuda_stream, buffer.get(), value, static_cast<int64_t>(batch_size));
  return buffer;
}

void run(
    void* output,
    void* q,
    void* k,
    void* v,
    void* attn_bias,
    int* mask_sequence_lengths_q,
    int* mask_sequence_lengths_kv,
    int batch_size,
    int num_heads_q,
    int num_heads_kv,
    int head_size_qk,
    int head_size_v,
    int sequence_length_q,
    int sequence_length_kv,
    float scale,
    bool is_causal,
    bool is_bf16,
    bool broadcast_attn_bias_dim_0,
    bool broadcast_attn_bias_dim_1,
    int sliding_window,
    AttentionQkvFormat qkv_format,
    cudnnHandle_t handle,
    Stream* stream,
    AllocatorPtr allocator) {
  // cuDNN requires both seq_len_q and seq_len_kv to be present when a padding mask is used. When the
  // caller provides only one side, synthesize the other as the full (unpadded) sequence length so it
  // behaves as a no-op padding mask on that side.
  IAllocatorUniquePtr<int> synthesized_seq_len_q;
  IAllocatorUniquePtr<int> synthesized_seq_len_kv;
  if (mask_sequence_lengths_q != nullptr || mask_sequence_lengths_kv != nullptr) {
    if (mask_sequence_lengths_q == nullptr) {
      synthesized_seq_len_q = CreateConstantSeqLenBuffer(allocator, stream, batch_size, sequence_length_q);
      mask_sequence_lengths_q = synthesized_seq_len_q.get();
    }
    if (mask_sequence_lengths_kv == nullptr) {
      synthesized_seq_len_kv = CreateConstantSeqLenBuffer(allocator, stream, batch_size, sequence_length_kv);
      mask_sequence_lengths_kv = synthesized_seq_len_kv.get();
    }
  }

  GraphParams params;
  params.batch_size = batch_size;
  params.num_heads_q = num_heads_q;
  params.num_heads_kv = num_heads_kv;
  params.head_size_qk = head_size_qk;
  params.head_size_v = head_size_v;
  params.sequence_length_q = sequence_length_q;
  params.sequence_length_kv = sequence_length_kv;
  params.scale = scale;
  // A single query token (s_q == 1, e.g. decode) attends to all keys up to its own position, so causal
  // masking is a no-op and the padding / kv sequence length bounds the valid keys. Dropping the causal
  // mask here also avoids a cuDNN limitation where decode-only graphs (s_q == 1) with a causal
  // right-bound fail to build on cuDNN backend versions <= 9.9.0.
  params.is_causal = is_causal && (sequence_length_q > 1);
  params.is_bf16 = is_bf16;
  params.qkv_format = qkv_format;
  params.handle = handle;
  params.has_bias = attn_bias != nullptr;
  params.broadcast_bias_dim_0 = broadcast_attn_bias_dim_0;
  params.broadcast_bias_dim_1 = broadcast_attn_bias_dim_1;
  params.has_padding_mask_q = (mask_sequence_lengths_q != nullptr);
  params.has_padding_mask_kv = (mask_sequence_lengths_kv != nullptr);
  params.sliding_window = sliding_window;

  std::shared_ptr<fe::graph::Graph> mha_graph;
  auto it = mha_graph_cache.find(params);
  if (it != mha_graph_cache.end()) {
    mha_graph = it->second;
  } else {
    mha_graph = build_graph(params);
    mha_graph_cache[params] = mha_graph;
  }

  std::unordered_map<fe::graph::Tensor_attributes::uid_t, void*> variant_pack = {
      {Q_UID, q},
      {K_UID, k},
      {V_UID, v},
      {O_UID, output},
  };

  if (attn_bias != nullptr) {
    variant_pack[BIAS_UID] = attn_bias;
  }

  if (mask_sequence_lengths_q != nullptr) {
    variant_pack[SEQ_LEN_Q_UID] = mask_sequence_lengths_q;
  }

  if (mask_sequence_lengths_kv != nullptr) {
    variant_pack[SEQ_LEN_KV_UID] = mask_sequence_lengths_kv;
  }

  // Allocate workspace.
  auto bytes = mha_graph->get_workspace_size();

  IAllocatorUniquePtr<void> buffer = IAllocator::MakeUniquePtr<void>(allocator, bytes, false, stream);

  CUDNN_FE_CALL_THROW(mha_graph->execute(handle, variant_pack, buffer.get()));
}

}  // namespace onnxruntime::cudnn_sdpa
#endif

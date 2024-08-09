#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cpu/bert/attention_common.h"

using onnxruntime::Stream;
using onnxruntime::contrib::AttentionQkvFormat;

namespace onnxruntime::cudnn_sdpa {

bool is_stable();

bool is_supported(const cudaDeviceProp& dprops,
                  int num_heads_q,
                  int num_heads_kv,
                  int head_size_qk,
                  int head_size_v,
                  int sequence_length_q,
                  int sequence_length_kv,
                  bool is_causal);

void run(
    void* q,
    void* k,
    void* v,
    void* output,
    int batch_size,
    int num_heads_q,
    int num_heads_kv,
    int head_size_qk,
    int head_size_v,
    int sequence_length_q,
    int sequence_length_kv,
    float scale,
    bool is_causal,
    bool is_bf16,                         // True if bfloat16, otherwise float16
    void* bias,                           // (optional) additive bias before softmax.
    gsl::span<const int64_t> bias_shape,  // Shape of attention_bias. Shall be [b or 1, h_q or 1, s_q, s_kv].
    int* mask_sequence_lengths_q,         // (optional) sequence lengths of q for padding mask. Shape: [b]
    int* mask_sequence_lengths_kv,        // (optional) sequence lengths of k or v for padding mask. Shape: [b]
    int sliding_window,                   // sliding window length. 0 means no sliding window.
    AttentionQkvFormat qkv_format,        // Q_K_V_BNSH, Q_K_V_BSNH, Q_K_V_BSNH_BNSH_BNSH are supported
    cudnnHandle_t handle,
    Stream* stream,
    AllocatorPtr allocator);

}  // namespace onnxruntime::cudnn_sdpa

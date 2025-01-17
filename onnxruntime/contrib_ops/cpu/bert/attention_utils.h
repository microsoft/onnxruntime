#pragma once
#include "core/common/common.h"
#include "core/framework/tensorprotoutils.h"
#include "core/framework/transpose_helper.h"
#include "core/providers/cpu/tensor/reshape_helper.h"
#include "core/providers/cpu/math/element_wise_ops.h"

namespace onnxruntime {
namespace contrib {

// Reshape Q/K/V from BxSxD to BxSxNxH
Status Reshape_BSD_to_BSNH(Tensor* qkv,
                           int batch_size,
                           int sequence_length,
                           int num_heads,
                           int head_size);

// Transpose Q/K/V from BxSxNxH to BxNxSxH
Status Transpose_BSNH_to_BNSH(const Tensor* qkv,
                              OrtValue& qkv_transposed,
                              concurrency::ThreadPool* tp = nullptr);

// Add bias + transpose for each of Q/K/V
template <typename T>
Status AddBiasTranspose(const Tensor* qkv,                   // Input: Q/K/V data - query is BxSxD, key is BxLxD, value is BxLxD_v
                        const T* qkv_bias,                   // Input: QKV bias - bias is (D + D + D_v)
                        OrtValue& qkv_with_bias_transposed,  // Output: Q/K/V data - query is BxNxSxH, key is BxNxLxH, value is BxNxLxH_v
                        int bias_offset,                     // bias offset to enter qkv_bias
                        int batch_size,                      // batch size
                        int sequence_length,                 // sequence_length for Q, kv_sequence_length for K/V
                        int num_heads,                       // num heads
                        int head_size,                       // head_size for Q/K, v_head_size for V
                        int hidden_size,                     // hidden_size for Q/K, v_hidden_size for V
                        OpKernelContext* context);           // OpKernelContext

// Add bias + reshape for each of Q/K/V
// This is used in decoder_with_past when the sequence length is 1
template <typename T>
Status AddBiasReshape(const Tensor* qkv,          // Input: Q/K/V data - query is BxSxD, key is BxLxD, value is BxLxD_v
                      const T* qkv_bias,          // Input: QKV bias - bias is (D + D + D_v)
                      OrtValue& qkv_with_bias,    // Output: Q/K/V data - query is BxNxSxH, key is BxNxLxH, value is BxNxLxH_v
                      int bias_offset,            // bias offset to enter qkv_bias
                      int batch_size,             // batch size
                      int sequence_length,        // sequence_length for Q, kv_sequence_length for K/V
                      int num_heads,              // num heads
                      int head_size,              // head_size for Q/K, v_head_size for V
                      int hidden_size,            // hidden_size for Q/K, v_hidden_size for V
                      OpKernelContext* context);  // OpKernelContext

template <typename T>
Status MaybeTransposeToBNSHAndAddBias(OpKernelContext* context, AllocatorPtr allocator,
                                      int batch_size, int num_heads, int sequence_length, int head_size,
                                      const Tensor* in, const Tensor* bias, int bias_offset, OrtValue& out);

template <typename T>
Status MaybeTransposeToBNSH(AllocatorPtr allocator,
                            int batch_size, int num_heads, int sequence_length, int head_size,
                            const Tensor* in, OrtValue& out);

}  // namespace contrib
}  // namespace onnxruntime

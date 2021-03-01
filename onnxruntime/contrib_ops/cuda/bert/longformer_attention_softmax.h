/*
Copyright (c) NVIDIA Corporation and Microsoft Corporation

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

// This is fast cuda kernels for longformer attention softmax.
// It uses two temporary matrix of BxNxSxS, and consumes more memory when sequence length is large.

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Launch the softmax kernel for non compact memory.
bool launchSoftmaxFastKernel(
    cudaStream_t stream,
    cublasHandle_t cublas,
    void* workspace,              // softmax space
    const void* q,                // transposed Q with shape (B, N, S, H)
    const void* k,                // transposed K with shape (B, N, S, H)
    const void* v,                // transposed V with shape (B, N, S, H)
    const void* attention_mask,   // attention mask with shape (B, S), with value 0.0 not masked, and -10000.0 masked.
    const void* global_q,         // Q for global tokens with shape (B, N, S, H)
    const void* global_k,         // K for global tokens with shape (B, N, S, H)
    const void* global_v,         // V for global tokens with shape (B, N, S, H)
    const int* global_attention,  // global attention with shape (B, S), with value 0 for local attention and 1 for global attention.
    const int* global_index,      // Global index with shape (B, S)
    const int* batch_global_num,  // Number of global tokens per batch with shape (B, 1)
    void* pinned_buffer,          // Pinned memory in CPU. Number of global tokens per batch with shape (B, 1)
    void* output,                 // output with shape (B, N, S, H)
    float scaler,                 // scalar
    int batch_size,               // batch size
    int sequence_length,          // sequence length
    int num_heads,                // number of heads
    int head_size,                // hidden size per head
    int attention_window,         // one sided windows size
    size_t element_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

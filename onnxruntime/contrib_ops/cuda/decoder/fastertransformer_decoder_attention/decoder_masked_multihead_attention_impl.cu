// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "decoder_masked_multihead_attention_impl.h"
#include "decoder_masked_multihead_attention_impl_utils.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {


    template<
        // The type of the inputs. Supported types: float and half.
        typename T,
        // The hidden dimension per head.
        int head_size,
        // The number of threads per key.
        int THREADS_PER_KEY,
        // The number of threads per value.
        int THREADS_PER_VALUE,
        // The number of threads in a threadblock.
        int THREADS_PER_BLOCK>
    __global__ void masked_multihead_attention_kernel(DecoderMaskedMultiheadAttentionParams params)
    {

        // Make sure the hidden dimension per head is a multiple of the number of threads per key.
        static_assert(head_size % THREADS_PER_KEY == 0, "");

        // Make sure the hidden dimension per head is a multiple of the number of threads per value.
        static_assert(head_size % THREADS_PER_VALUE == 0, "");

        // The size of a warp.
        //constexpr int WARP_SIZE = 32;

        // The number of warps in a threadblock.
        //constexpr int WARPS_PER_BLOCK = THREADS_PER_BLOCK / WARP_SIZE;

        // Use smem_size_in_bytes (above) to determine the amount of shared memory.
        //extern __shared__ char smem_[];

        // The shared memory for the Q*K^T values and partial logits in softmax.
        //float* qk_smem = reinterpret_cast<float*>(smem_);

        // The shared memory for the logits. For FP32, that's the same buffer as qk_smem.
        //char* logits_smem_ = smem_;

        // TODO: Understand this
        /*
        if (sizeof(T) != 4) {
            // TODO - change to tlength
            const int max_timesteps = min(params.timestep, params.memory_max_len);
            logits_smem_ +=
                (DO_CROSS_ATTENTION) ? div_up(params.memory_max_len + 1, 4) * 16 : div_up(max_timesteps + 1, 4) * 16;
        }
        */

        //T* logits_smem = reinterpret_cast<T*>(logits_smem_);

        // The shared memory to do the final reduction for the output values. Reuse qk_smem.
        //T* out_smem = reinterpret_cast<T*>(smem_);

        // The shared memory buffers for the block-wide reductions. One for max, one for sum.
        //__shared__ float red_smem[WARPS_PER_BLOCK * 2];

        // A vector of Q or K elements for the current timestep.
        using Qk_vec_k = typename Qk_vec_k_<T, head_size>::Type;  // with kernel-used precision
        using Qk_vec_m = typename Qk_vec_m_<T, head_size>::Type;  // with memory-used precision

        // Use alignment for safely casting the shared buffers as Qk_vec_k.
        // Shared memory to store Q inputs.
        __shared__ __align__(sizeof(Qk_vec_k)) T q_smem[head_size];

        // The number of elements per vector.
        constexpr int QK_VEC_SIZE = sizeof(Qk_vec_m) / sizeof(T);

        // Make sure the hidden size per head is a multiple of the vector size.
        static_assert(head_size % QK_VEC_SIZE == 0, "");

        constexpr int QK_VECS_PER_WARP = head_size / QK_VEC_SIZE;

        // The layout of the cache is [B, H, head_size/x, L, x] with x == 4/8/16 for FP32/FP16/FP8. Since each thread
        // owns x elements, we have to decompose the linear index into chunks of x values and the posi-
        // tion of the thread in that chunk.

        // The number of elements in a chunk of 16B (that's the x in the above formula).
        constexpr int QK_ELTS_IN_16B = 16 / sizeof(T);

        // The number of K vectors in 16B.
        constexpr int QK_VECS_IN_16B = 16 / sizeof(Qk_vec_m);

        // The batch/beam idx
        const int bi = blockIdx.y;

        // The beam idx
        //const int beami = bi % params.beam_width;

        // The "beam-aware" batch idx
        //const int bbi = bi / params.beam_width;

        // The head.
        const int hi = blockIdx.x;
        
        // Combine the batch and the head indices.
        const int bhi = bi * params.num_heads + hi;
        
        // Combine the "beam-aware" batch idx and the head indices.
        //const int bbhi = bbi * params.beam_width * params.num_heads + hi;
        
        // The thread in the block.
        const int tidx = threadIdx.x;

        // While doing the product Q*K^T for the different keys we track the max.
        //float qk_max = -FLT_MAX;

        //float qk = 0.0F;

        int qkv_base_offset = bi * (3 * params.hidden_size) + hi * head_size;

        //const size_t bi_seq_len_offset = bi * params.max_sequence_length;

        // First QK_VECS_PER_WARP load Q and K + the bias values for the current timestep.
        const bool is_masked = tidx >= QK_VECS_PER_WARP;

        // The offset in the Q and K buffer also accounts for the batch.
        int qk_offset = qkv_base_offset + tidx * QK_VEC_SIZE;

        // The offset in the bias buffer.
        int qk_bias_offset = hi * head_size + tidx * QK_VEC_SIZE;

        // Re-interpret cast void* to appropriate types
        T* params_q = &reinterpret_cast<T*>(params.q)[qk_offset];
        T* params_k = &reinterpret_cast<T*>(params.k)[qk_offset];

        T* params_q_bias = &reinterpret_cast<T*>(params.q_bias)[qk_bias_offset];
        T* params_k_bias = &reinterpret_cast<T*>(params.k_bias)[qk_bias_offset];

        // Trigger the loads from the Q and K buffers.
        Qk_vec_k q;
        zero(q);
        
        if (!is_masked) {
            q = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(params_q));
        }

        Qk_vec_k k;
        zero(k);

        if (!is_masked) {
            k = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(params_k));
        }

        // Trigger the loads from the Q and K bias buffers.
        Qk_vec_k q_bias;
        zero(q_bias);

        if (!is_masked) {
            q_bias = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(params_q_bias));
        }

        Qk_vec_k k_bias;
        zero(k_bias);

        if (!is_masked) {
            k_bias = vec_conversion<Qk_vec_k, Qk_vec_m>(*reinterpret_cast<const Qk_vec_m*>(params_k_bias));
        }

        // Computes the Q/K values with bias.
        q = add_vec(q, q_bias);
        k = add_vec(k, k_bias);

        // DEBUG
        if (!is_masked) {
            // Store the Q values to shared memory.
            (void*)(q_smem);
            //*reinterpret_cast<Qk_vec_k*>(&q_smem[tidx * QK_VEC_SIZE]) = q;
 

            *reinterpret_cast<Qk_vec_k*>(params_q) = q;
            *reinterpret_cast<Qk_vec_k*>(params_k) = k;

        }

        T* params_k_cache = reinterpret_cast<T*>(params.k_cache);

        if (!is_masked) {
            // Store the Q values to shared memory.
            //*reinterpret_cast<Qk_vec_k*>(&q_smem[tidx * QK_VEC_SIZE]) = q;

            // Write the K values to the global memory cache.
            //
            // NOTE: The stores are uncoalesced as we have multiple chunks of 16B spread across the memory
            // system. We designed it this way as it allows much better memory loads (and there are many
            // more loads) + the stores are really "write and forget" since we won't need the ack before
            // the end of the kernel. There's plenty of time for the transactions to complete.

            // The 16B chunk written by the thread.
            int co = tidx / QK_VECS_IN_16B;

            // The position of the thread in that 16B chunk.
            int ci = tidx % QK_VECS_IN_16B * QK_VEC_SIZE;

            // Two chunks are separated by L * x elements. A thread write QK_VEC_SIZE elements.
            int offset = bhi * params.max_sequence_length * head_size + co * params.max_sequence_length * QK_ELTS_IN_16B +
                params.past_sequence_length * QK_ELTS_IN_16B + ci;

            // Trigger the stores to global memory.
            *reinterpret_cast<Qk_vec_m*>(&params_k_cache[offset]) = vec_conversion<Qk_vec_m, Qk_vec_k>(k);
        }

            /*
            // Compute \sum_i Q[i] * K^T[i] for the current timestep.
            using Qk_vec_acum = Qk_vec_k;
            qk = dot<Qk_vec_acum, Qk_vec_k>(q, k);

            if (QK_VECS_PER_WARP <= WARP_SIZE) {
#pragma unroll
                for (int mask = QK_VECS_PER_WARP / 2; mask >= 1; mask /= 2) {
                    qk += __shfl_xor_sync(shfl_mask(QK_VECS_PER_WARP), qk, mask);
                }
            }
        }

        if (QK_VECS_PER_WARP > WARP_SIZE) {
            constexpr int WARPS_PER_RED = (QK_VECS_PER_WARP + WARP_SIZE - 1) / WARP_SIZE;
            qk = block_sum<WARPS_PER_RED>(&red_smem[WARPS_PER_RED], qk);
        }

        const float inv_sqrt_dh = 1.f / (sqrtf(static_cast<float>(head_size)));

        // Store that value in shared memory. Keep the Q*K^T value in register for softmax.
        if (tidx == 0) {
            // Normalize qk.
            qk *= inv_sqrt_dh;
            qk_max = qk;
            qk_smem[params.total_sequence_length] = qk;
        }

        // Make sure the data is in shared memory.
        __syncthreads();

        // The type of queries and keys for the math in the Q*K^T product.
        using K_vec_k = typename K_vec_k_<T, THREADS_PER_KEY>::Type;
        using K_vec_m = typename K_vec_m_<T, THREADS_PER_KEY>::Type;

        // The number of elements per vector.
        constexpr int K_VEC_SIZE = sizeof(K_vec_m) / sizeof(T);

        // Make sure the hidden size per head is a multiple of the vector size.
        static_assert(head_size % K_VEC_SIZE == 0, "");

        // The number of elements per thread.
        constexpr int K_ELTS_PER_THREAD = head_size / THREADS_PER_KEY;

        // The number of vectors per thread.
        constexpr int K_VECS_PER_THREAD = K_ELTS_PER_THREAD / K_VEC_SIZE;

        // The position the first key loaded by each thread from the cache buffer (for this B * H).
        int ko = tidx / THREADS_PER_KEY;

        // The position of the thread in the chunk of keys.
        int ki = tidx % THREADS_PER_KEY * K_VEC_SIZE;

        static_assert(head_size == THREADS_PER_KEY * K_VEC_SIZE * K_VECS_PER_THREAD);

        // Load the Q values from shared memory. The values are reused during the loop on K.
        K_vec_k q_vec[K_VECS_PER_THREAD];
#pragma unroll
        for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
            q_vec[ii] = *reinterpret_cast<const K_vec_k*>(&q_smem[ki + ii * THREADS_PER_KEY * K_VEC_SIZE]);
        }

        // The number of timesteps loaded per iteration.
        constexpr int K_PER_ITER = THREADS_PER_BLOCK / THREADS_PER_KEY;

        // The number of keys per warp.
        constexpr int K_PER_WARP = WARP_SIZE / THREADS_PER_KEY;

        // Base pointer for the beam's batch, before offsetting with indirection buffer
        T* k_cache_batch = &params_k_cache[bbhi * params.max_sequence_length * head_size + ki];

        // Pick a number of keys to make sure all the threads of a warp enter (due to shfl_sync).
        int ti_end = ((params.total_sequence_length + K_PER_WARP - 1) / K_PER_WARP) * K_PER_WARP;

        // Iterate over the keys/timesteps to compute the various (Q*K^T)_{ti} values.
        bool has_beams = params.cache_indir != nullptr;

        const int* beam_indices = has_beams ? &params.cache_indir[bi_seq_len_offset] : nullptr;

        for (int ti = ko; ti < ti_end; ti += K_PER_ITER) {
            bool      is_masked = (params.mask != nullptr) && params.mask[bi_seq_len_offset + ti];

            // The keys loaded from the key cache.
            K_vec_k k[K_VECS_PER_THREAD];
#pragma unroll
            for (int ii = 0; ii < K_VECS_PER_THREAD; ++ii) {
                int jj = ii * params.max_sequence_length + ti;

                if (ti < params.total_sequence_length) {
                    if (has_beams) {
                        const int beam_offset = beam_indices[ti] * params.num_heads * params.max_sequence_length * head_size;
                        k[ii] = vec_conversion<K_vec_k, K_vec_m>(
                            (*reinterpret_cast<const K_vec_m*>(&k_cache_batch[beam_offset + jj * QK_ELTS_IN_16B])));
                    }
                    else {
                        k[ii] = vec_conversion<K_vec_k, K_vec_m>(
                            (*reinterpret_cast<const K_vec_m*>(&k_cache_batch[jj * QK_ELTS_IN_16B])));
                    }
                }
            }

            // Perform the dot product and normalize qk.
            // WARNING: ALL THE THREADS OF A WARP MUST ENTER!!!
            float qk = Qk_dot<T, THREADS_PER_KEY>::dot(q_vec, k) * inv_sqrt_dh;

            // Store the product to shared memory. There's one qk value per timestep. Update the max.
            // if( ti < params.timestep && tidx % THREADS_PER_KEY == 0 ) {
            if (ti < params.total_sequence_length && tidx % THREADS_PER_KEY == 0) {
                qk_max = is_masked ? qk_max : fmaxf(qk_max, qk);
                qk_smem[ti] = qk;
            }
        }

        // Perform the final reduction to compute the max inside each warp.
        //
        // NOTE: In a group of THREADS_PER_KEY threads, the leader already has the max value for the
        // group so it's not needed to run the reduction inside the group (again).
#pragma unroll
        for (int mask = WARP_SIZE / 2; mask >= THREADS_PER_KEY; mask /= 2) {
            qk_max = fmaxf(qk_max, __shfl_xor_sync(uint32_t(-1), qk_max, mask));
        }
        */
    }

    template void __global__ masked_multihead_attention_kernel<float, 64, 4, 16, 64>(DecoderMaskedMultiheadAttentionParams params);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

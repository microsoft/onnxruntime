#include "mlas_flashattn.h"
#include <numeric>
#include "mlasi.h"

void
FlashAttentionThreaded(
    std::ptrdiff_t thread_id,
    const FlashAttentionThreadedArgs* args
)
{
    ptrdiff_t q_block_size = static_cast<ptrdiff_t>(args->q_block_size);
    ptrdiff_t kv_block_size = static_cast<ptrdiff_t>(args->kv_block_size);
    ptrdiff_t batch_size = static_cast<ptrdiff_t>(args->batch_size);
    ptrdiff_t num_heads = static_cast<ptrdiff_t>(args->num_heads);
    ptrdiff_t q_sequence_length = static_cast<ptrdiff_t>(args->q_sequence_length);
    ptrdiff_t kv_sequence_length = static_cast<ptrdiff_t>(args->kv_sequence_length);
    ptrdiff_t qk_head_size = static_cast<ptrdiff_t>(args->qk_head_size);
    ptrdiff_t v_head_size = static_cast<ptrdiff_t>(args->v_head_size);
    float* buffer = args->buffer;
    ptrdiff_t buffer_size_per_thread = static_cast<ptrdiff_t>(args->buffer_size_per_thread);
    ptrdiff_t thread_count = static_cast<ptrdiff_t>(args->thread_count);
    const float* query = args->query;
    const float* key = args->key;
    const float* value = args->value;
    float* output = args->output;

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
    auto&& mlas_platform = GetMlasPlatform();
#endif

    ptrdiff_t q_block_count = (q_sequence_length + (q_block_size - 1)) / q_block_size;

    ptrdiff_t task_start = 0;
    ptrdiff_t task_end = 0;
    ptrdiff_t total_task_count = batch_size * num_heads * q_block_count;
    ptrdiff_t quotient = total_task_count / thread_count;
    ptrdiff_t remainder = total_task_count % thread_count;
    if (thread_id < remainder) {
        task_start = (quotient + 1) * thread_id;
        task_end = task_start + quotient + 1;
    } else {
        task_start = quotient * thread_id + remainder;
        task_end = task_start + quotient;
    }

    for (ptrdiff_t task_index = task_start; task_index < task_end; ++task_index) {
        ptrdiff_t ib = task_index;
        ptrdiff_t il = (ib % q_block_count) * q_block_size;
        ib /= q_block_count;
        ptrdiff_t ih = ib % num_heads;
        ib /= num_heads;

        float* buffer_current_thread = buffer + thread_id * buffer_size_per_thread;
        float* l = buffer_current_thread;

        memset(l, 0, q_block_size * sizeof(float));
        float* m = l + q_block_size;
        for (ptrdiff_t t = 0; t < q_block_size; ++t) {
            m[t] = std::numeric_limits<float>::lowest();
        }
        float* intermediate = m + q_block_size;
        float* temp_output = intermediate + q_block_size * kv_block_size;
        float negmax = 0;

        for (ptrdiff_t ir = 0; ir < kv_sequence_length; ir += kv_block_size) {
            /*
                S = Q[ib, ih, il:il+q_block_size, :] * (K[ib, ih, ir:ir+kv_block_size, :]).T
                old_m = m
                m = max(m, rowmax(S))
                diff = old_m - m
                S = exp(S - m)
                l = exp(diff) * l + rowsum(S)
                O = diag(exp(diff)) * O + S * V[ib, ih, ir:ir+kv_block_size, :]
            */
            // TODO: Need to concat if past_k is present
            ptrdiff_t h = ib * num_heads + ih;
            const float* inputQ = query + (h * q_sequence_length + il) * qk_head_size;
            const float* inputK = key + (h * kv_sequence_length + ir) * qk_head_size;
            const float* inputV = value + (h * kv_sequence_length + ir) * v_head_size;

            size_t q_block_size_capped = static_cast<size_t>(std::min(q_block_size, q_sequence_length - il));
            size_t kv_block_size_capped = static_cast<size_t>(std::min(kv_block_size, kv_sequence_length - ir));

            MlasGemm(CBLAS_TRANSPOSE::CblasNoTrans,
                     CBLAS_TRANSPOSE::CblasTrans,
                     q_block_size_capped,
                     kv_block_size_capped,
                     static_cast<size_t>(qk_head_size),
                     args->scale,
                     inputQ,
                     static_cast<size_t>(qk_head_size),
                     inputK,
                     static_cast<size_t>(qk_head_size),
                     0.0f,
                     intermediate,
                     kv_block_size_capped,
                     nullptr);

            for (ptrdiff_t irow = 0; irow < static_cast<ptrdiff_t>(q_block_size_capped); ++irow) {
                float* p = intermediate + irow * kv_block_size_capped;

#if defined(MLAS_TARGET_AMD64) || defined(MLAS_TARGET_LARCH64)
                float rowmax = mlas_platform.ReduceMaximumF32Kernel(p, kv_block_size_capped);
#else
                float rowmax = MlasReduceMaximumF32Kernel(p, kv_block_size_capped);
#endif
                float m_diff = m[irow];
                m[irow] = std::max(m[irow], rowmax);  // new m
                negmax = -m[irow];
                m_diff -= m[irow];  // old - new (less than 0)

#if defined(MLAS_TARGET_AMD64)
                float rowsum = mlas_platform.ComputeSumExpF32Kernel(p, p, kv_block_size_capped, &negmax);
#else
                float rowsum = MlasComputeSumExpF32Kernel(p, p, kv_block_size_capped, &negmax);
#endif

                // Note: for ir == 0, there is actually no need to calculate exp_diff
                if (ir != 0) {
                    float exp_diff = std::exp(m_diff);
                    l[irow] = exp_diff * l[irow] + rowsum;

                    for (ptrdiff_t icol = 0; icol < v_head_size; ++icol) {
                        temp_output[irow * v_head_size + icol] = exp_diff * temp_output[irow * v_head_size + icol];
                    }
                } else {
                    l[irow] = rowsum;
                    // When ir == 0, there is no need to scale the old result because it is zero.
                }
            }
            MlasGemm(CBLAS_TRANSPOSE::CblasNoTrans,
                     CBLAS_TRANSPOSE::CblasNoTrans,
                     q_block_size_capped,
                     static_cast<size_t>(v_head_size),
                     kv_block_size_capped,
                     1.0f,
                     intermediate,
                     kv_block_size_capped,
                     inputV,
                     static_cast<size_t>(v_head_size),
                     ir == 0 ? 0.0f : 1.0f,
                     temp_output,
                     static_cast<size_t>(v_head_size),
                     nullptr);
        }

        float* output_row = output + ((ib * q_sequence_length + il) * num_heads + ih) * v_head_size;
        ptrdiff_t q_block_size_valid = std::min(q_block_size, q_sequence_length - il);
        // TODO: leverage advanced instruction sets
        for (ptrdiff_t irow = 0; irow < q_block_size_valid; ++irow) {
            for (ptrdiff_t icol = 0; icol < v_head_size; ++icol) {
                output_row[icol] = temp_output[irow * v_head_size + icol] / l[irow];
            }
            output_row += num_heads * v_head_size;
        }
    }
}

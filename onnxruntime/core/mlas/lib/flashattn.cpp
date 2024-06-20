#include "mlasi.h"
#include "mlas_flashattn.h"

#include <numeric>

void FlashAttentionThreaded(
    std::ptrdiff_t thread_id,
    struct FlashAttentionThreadedArgs* args
    )
{
    int row_size_q = args->row_size_q;
    int row_size_kv = args->row_size_kv;
    int batch_size = args->batch_size;
    int num_heads = args->num_heads;
    int q_sequence_length = args->q_sequence_length;
    int kv_sequence_length = args->kv_sequence_length;
    int qk_head_size = args->qk_head_size;
    int v_head_size = args->v_head_size;
    float* buffer = args->buffer;
    size_t buffer_size_per_thread = args->buffer_size_per_thread;
    int thread_count = args->thread_count;
    const float* query = args->query;
    const float* key = args->key;
    const float* value = args->value;
    float* output = args->output;
    const float scale = args->scale;

    auto&& mlas_platform = GetMlasPlatform();

    int q_chunk_count = (q_sequence_length + (row_size_q - 1))/ row_size_q;

    int task_start = 0;
    int task_end = 0;
    int total_task_count = batch_size * num_heads * q_chunk_count;
    int quotient = total_task_count / thread_count;
    int remainder = total_task_count % thread_count;
    if(thread_id < remainder){
        task_start = (quotient + 1) * static_cast<int>(thread_id);
        task_end = task_start + quotient + 1;
    }
    else{
        task_start = quotient * static_cast<int>(thread_id) + remainder;
        task_end = task_start + quotient;
    }

    for(auto task_index = task_start; task_index < task_end; ++task_index){
        int ib = static_cast<int>(task_index);
        int il = (ib % q_chunk_count) * row_size_q;
        ib /= q_chunk_count;
        int ih = ib % num_heads;
        ib /= num_heads;

        float* buffer_current_thread = reinterpret_cast<float*>(reinterpret_cast<char*>(buffer) + thread_id * buffer_size_per_thread);

        float* l = buffer_current_thread;
        memset(l, 0, row_size_q * sizeof(float));
        float* m = l + row_size_q;
        for (int t = 0; t < row_size_q; ++t) {
            m[t] = std::numeric_limits<float>::lowest();
        }
        float* intermediate = m + row_size_q;
        float* temp_output = intermediate + row_size_q * row_size_kv;
        float negmax = 0;

        for(int ir = 0; ir < kv_sequence_length; ir += row_size_kv) {
            /*
                S = Q[ib, ih, il:il+row_size_q, :] * (K[ib, ih, ir:ir+row_size_kv, :]).T
                old_m = m
                m = max(m, rowmax(S))
                diff = old_m - m
                S = exp(S - m)
                l = exp(diff) * l + rowsum(S)
                O = diag(exp(diff)) * O + S * V[ib, ih, ir:ir+row_size_kv, :]
            */
            // TODO: Need to concat if past_k is present
            const float* inputQ = query + ((ib * num_heads + ih) * q_sequence_length + il) * qk_head_size;
            const float* inputK = key + ((ib * num_heads + ih) * kv_sequence_length + ir) * qk_head_size;
            const float* inputV = value + ((ib * num_heads + ih) * kv_sequence_length + ir) * v_head_size;

            auto row_size_q_capped = std::min(row_size_q, q_sequence_length - il);
            auto row_size_kv_capped = std::min(row_size_kv, kv_sequence_length - ir);
            MlasGemm(CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasTrans, row_size_q_capped, row_size_kv_capped, qk_head_size, scale, inputQ, qk_head_size, inputK, qk_head_size, 0.0f, intermediate, row_size_kv_capped, nullptr);

            for(int irow = 0; irow < row_size_q_capped; ++irow) {
                float rowmax = mlas_platform.ReduceMaximumF32Kernel(intermediate + irow * row_size_kv_capped, row_size_kv_capped);
                float m_diff = m[irow];
                m[irow] = std::max(m[irow], rowmax);  // new m
                negmax = -m[irow];
                m_diff -= m[irow]; // old - new (less than 0)

                float rowsum = mlas_platform.ComputeSumExpF32Kernel(intermediate + irow * row_size_kv_capped, intermediate + irow * row_size_kv_capped, row_size_kv_capped, &negmax);

                // Note: for ir == 0, there is actually no need to calculate exp_diff
                if (ir != 0) {
                    float exp_diff = std::exp(m_diff);
                    l[irow] = exp_diff * l[irow] + rowsum;

                    for (int icol = 0; icol < v_head_size; ++icol) {
                        temp_output[irow * v_head_size + icol] = exp_diff * temp_output[irow * v_head_size + icol];
                    }
                }
                else {
                    l[irow] = rowsum;
                    // When ir == 0, there is no need to scale the old result because it is zero.
                }
            }
            MlasGemm(CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans, row_size_q_capped, v_head_size, row_size_kv_capped, 1.0f, intermediate, row_size_kv_capped, inputV, v_head_size, ir == 0 ? 0.0f : 1.0f, temp_output, v_head_size, nullptr);
        }

        float* output_row = output + ((ib * q_sequence_length + il) * num_heads + ih) * v_head_size;
        auto row_size_q_valid = std::min(row_size_q, q_sequence_length - il);
        // TODO: leverage advanced instruction sets
        for (int irow = 0; irow < row_size_q_valid; ++irow) {
            for (int icol = 0; icol < v_head_size; ++icol) {
                output_row[icol] = temp_output[irow * v_head_size + icol] / l[irow];
            }
            output_row += num_heads * v_head_size;
        }
    }
}

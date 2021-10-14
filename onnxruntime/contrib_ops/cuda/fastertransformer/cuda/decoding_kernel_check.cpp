/*
Copyright (c) 2020-2021, NVIDIA CORPORATION.  All rights reserved.

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

#include "decoding_kernel_check.h"

namespace fastertransformer
{

void init_kernel_check(bool *d_finished, int *d_sequence_length, int *d_word_ids, float *d_cum_log_probs, const int sentence_id, const int batch_size,
                       const int beam_width, cudaStream_t stream)
{

    printf("[INFO] decoding init check. \n");

    bool *h_finished = new bool[batch_size * beam_width];
    int *h_sequence_length = new int[batch_size * beam_width];
    int *h_word_ids = new int[batch_size * beam_width];
    float *h_cum_log_probs = new float[batch_size * beam_width];

    init_kernelLauncher(d_finished, d_sequence_length, d_word_ids, d_cum_log_probs,
         sentence_id, batch_size, beam_width, stream);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    check_cuda_error(cudaMemcpy(h_finished, d_finished, sizeof(bool) * batch_size * beam_width, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_sequence_length, d_sequence_length, sizeof(int) * batch_size * beam_width, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_word_ids, d_word_ids, sizeof(int) * batch_size * beam_width, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_cum_log_probs, d_cum_log_probs, sizeof(float) * batch_size * beam_width, cudaMemcpyDeviceToHost));

    bool *h_finished_cpu = new bool[batch_size * beam_width];
    int *h_sequence_length_cpu = new int[batch_size * beam_width];
    int *h_word_ids_cpu = new int[batch_size * beam_width];
    float *h_cum_log_probs_cpu = new float[batch_size * beam_width];

    for (int i = 0; i < batch_size * beam_width; i++)
    {
        h_finished_cpu[i] = false;
        h_sequence_length_cpu[i] = 0;
        h_word_ids_cpu[i] = sentence_id;
        if (i % beam_width == 0)
            h_cum_log_probs_cpu[i] = 0.0f;
        else
            h_cum_log_probs_cpu[i] = -1e20f;
    }

    for (int i = 0; i < batch_size * beam_width; i++)
    {
        if (h_finished[i] != h_finished_cpu[i])
        {
            printf("[ERROR] finished initialize fail. \n");
            exit(-1);
        }
        if (h_sequence_length[i] != h_sequence_length_cpu[i])
        {
            printf("[ERROR] sequence length initialize fail. \n");
            exit(-1);
        }
        if (h_word_ids[i] != h_word_ids_cpu[i])
        {
            printf("[ERROR] %d kernel word is: %d, cpu word is: %d \n", i, h_word_ids[i], h_word_ids_cpu[i]);
            printf("[ERROR] word ids initialize fail. \n");
            exit(-1);
        }
        if (h_cum_log_probs[i] != h_cum_log_probs_cpu[i])
        {
            printf("[ERROR] cum log probs initialize fail. \n");
            exit(-1);
        }
    }

    delete[] h_cum_log_probs_cpu;
    delete[] h_word_ids_cpu;
    delete[] h_sequence_length_cpu;
    delete[] h_finished_cpu;

    delete[] h_cum_log_probs;
    delete[] h_word_ids;
    delete[] h_sequence_length;
    delete[] h_finished;
    printf("[INFO] decoding init check Finish. \n");
}

void update_logits_kernel_check(float *logits, const float *tmp_logits, const float *bias, const int end_id, const bool *finished, const int m, const int n, cudaStream_t stream)
{
    // m: batch_size * beam_width
    // n: vocab size

    printf("[INFO] decoding update logits check. \n");

    float *h_logits = new float[m * n];
    float *h_logits_after_update = new float[m * n];
    float *h_logits_after_update_cpu = new float[m * n];
    float *h_bias = new float[n];
    bool *h_finished = new bool[m];

    check_cuda_error(cudaMemcpy(h_logits, logits, sizeof(float) * m * n, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_bias, bias, sizeof(float) * n, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_finished, finished, sizeof(bool) * m, cudaMemcpyDeviceToHost));
    update_logits(logits, tmp_logits, bias, end_id, finished, m, n, stream);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    check_cuda_error(cudaMemcpy(h_logits_after_update, logits, sizeof(float) * m * n, cudaMemcpyDeviceToHost));

    // update logits in cpu
    // add bias
    for (int i = 0; i < m; i++)
    {
        if (h_finished[i] == false)
        {
            for (int j = 0; j < n; j++)
            {
                h_logits_after_update_cpu[i * n + j] = h_logits[i * n + j] + h_bias[j];
            }
        }
        else
        {
            for (int j = 0; j < n; j++)
            {
                h_logits_after_update_cpu[i * n + j] = ((j == end_id) ? FLT_MAX : -1 * FLT_MAX);
            }
        }
    }

    for (int i = 0; i < m; i++)
    {
        for (int j = 0; j < n; j++)
        {
            h_logits_after_update_cpu[i * n + j] = h_logits[i * n + j] + h_bias[j];
        }
    }

    // compute log_softmax
    for (int i = 0; i < m; i++)
    {
        //
        // reduce max
        float max = -1 * FLT_MAX;
        for (int j = 0; j < n; j++)
        {
            float val = h_logits_after_update_cpu[i * n + j];
            if (val > max)
                max = val;
        }

        // minus the max value to prevent overflow, and compute the exponential
        float sum = 0.0f;
        for (int j = 0; j < n; j++)
        {
            h_logits_after_update_cpu[i * n + j] = expf((float)h_logits_after_update_cpu[i * n + j] - max);
            sum = sum + (float)h_logits_after_update_cpu[i * n + j];
        }

        for (int j = 0; j < n; j++)
        {
            h_logits_after_update_cpu[i * n + j] = logf((float)h_logits_after_update_cpu[i * n + j] / sum);
        }
    }

    // check the logits
    for (int i = 0; i < m * n; i++)
    {
        float diff = (float)(h_logits_after_update_cpu[i] - h_logits_after_update[i]);
        if (diff < 0)
            diff = diff * -1.0;
        if (diff > 2e-5)
        {
            printf("[ERROR] update logits fail on %d with | %f - %f | = %f. \n", i, (float)h_logits_after_update_cpu[i], (float)h_logits_after_update[i], diff);
            exit(-1);
        }
    }

    delete[] h_logits;
    delete[] h_logits_after_update;
    delete[] h_logits_after_update_cpu;
    delete[] h_bias;
    delete[] h_finished;
    printf("[INFO] decoding update logits check finish. \n");
}

void broadcast_kernel_check(float *log_probs, float *cum_log_probs, const int batch_size, const int beam_width,
                            const int vocab_size, cudaStream_t stream)
{

    printf("[INFO] decoding broacast check. \n");
    float *h_log_probs = new float[batch_size * beam_width * vocab_size];
    float *h_cum_log_probs = new float[batch_size * beam_width];
    float *h_log_probs_after_update = new float[batch_size * beam_width * vocab_size];
    float *h_log_probs_after_update_cpu = new float[batch_size * beam_width * vocab_size];

    check_cuda_error(cudaMemcpy(h_log_probs, log_probs, sizeof(float) * batch_size * beam_width * vocab_size, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_cum_log_probs, cum_log_probs, sizeof(float) * batch_size * beam_width, cudaMemcpyDeviceToHost));

    broadcast_kernelLauncher(log_probs, cum_log_probs, batch_size, beam_width, vocab_size, stream);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    check_cuda_error(cudaMemcpy(h_log_probs_after_update, log_probs, sizeof(float) * batch_size * beam_width * vocab_size, cudaMemcpyDeviceToHost));

    for (int i = 0; i < batch_size * beam_width; i++)
    {
        for (int j = 0; j < vocab_size; j++)
        {
            h_log_probs_after_update_cpu[i * vocab_size + j] = h_log_probs[i * vocab_size + j] + h_cum_log_probs[i];
        }
    }

    // check the logits
    for (int i = 0; i < batch_size * beam_width * vocab_size; i++)
    {
        float diff = (float)(h_log_probs_after_update_cpu[i] - h_log_probs_after_update[i]);
        if (diff < 0)
            diff = diff * -1;
        if (diff > 1e-5)
        {
            printf("[ERROR] broadcast fail on %d with | %f - %f | = %f. \n",
                    i, (float)h_log_probs_after_update_cpu[i], (float)h_log_probs_after_update[i], diff);
            exit(-1);
        }
    }

    delete[] h_log_probs;
    delete[] h_cum_log_probs;
    delete[] h_log_probs_after_update;
    delete[] h_log_probs_after_update_cpu;
    printf("[INFO] decoding broacast check finish. \n");
}

void topK_kernel_check(const float *log_probs, int *ids, const int batch_size, const int beam_width, const int vocab_size,
                       cudaStream_t stream)
{

    printf("[INFO] decoding topK check. \n");
    float *h_log_probs = new float[batch_size * beam_width * vocab_size];
    int *h_ids_after_update = new int[batch_size * beam_width];
    int *h_ids_after_update_cpu = new int[batch_size * beam_width];

    topK(log_probs, ids, batch_size, beam_width, vocab_size, stream);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());
    check_cuda_error(cudaMemcpy(h_log_probs, log_probs, sizeof(float) * batch_size * beam_width * vocab_size, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_ids_after_update, ids, sizeof(int) * batch_size * beam_width, cudaMemcpyDeviceToHost));

    for (int i = 0; i < batch_size; i++)
    {
        for (int j = 0; j < beam_width; j++)
        {
            float max_val = -1 * FLT_MAX;
            int max_id = -1;
            for (int k = 0; k < beam_width * vocab_size; k++)
            {
                if (h_log_probs[i * beam_width * vocab_size + k] > max_val)
                {
                    max_id = i * beam_width * vocab_size + k;
                    max_val = h_log_probs[max_id];
                }
            }

            h_ids_after_update_cpu[i * beam_width + j] = max_id;
            h_log_probs[max_id] = -FLT_MAX;
        }
    }

    check_cuda_error(cudaMemcpy(h_log_probs, log_probs, sizeof(float) * batch_size * beam_width * vocab_size, cudaMemcpyDeviceToHost));

    // check the topK
    for (int i = 0; i < batch_size * beam_width; i++)
    {
        if (h_ids_after_update[i] != h_ids_after_update_cpu[i])
        {
            for (int i = 0; i < batch_size * beam_width; i++)
            {
                printf("[INFO] cpu result: %d %d %f \n", i, h_ids_after_update_cpu[i], (float)(h_log_probs[h_ids_after_update_cpu[i]]));
            }
            for (int i = 0; i < batch_size * beam_width; i++)
            {
                printf("[INFO] gpu result: %d %d %f \n", i, h_ids_after_update[i], (float)(h_log_probs[h_ids_after_update[i]]));
            }
            printf("[WARNING] topK fail on %d with %d (%f) %d (%f). \n", i,
                   h_ids_after_update_cpu[i], (float)h_log_probs[h_ids_after_update_cpu[i]],
                   h_ids_after_update[i], (float)h_log_probs[h_ids_after_update[i]]);
            if (h_log_probs[h_ids_after_update_cpu[i]] != h_log_probs[h_ids_after_update[i]])
            {
                printf("[ERROR] topK fail on %d with %d (%f) %d (%f). \n", i,
                       h_ids_after_update_cpu[i], (float)h_log_probs[h_ids_after_update_cpu[i]],
                       h_ids_after_update[i], (float)h_log_probs[h_ids_after_update[i]]);

                exit(-1);
            }
        }
    }

    delete[] h_log_probs;
    delete[] h_ids_after_update;
    delete[] h_ids_after_update_cpu;
    printf("[INFO] decoding topK check finish. \n");
}

void update_kernel_check(float *log_probs, float *cum_log_probs, int *ids, bool *finished, int *parent_ids, int *sequence_length,
                         int *word_ids, int *output_ids,
                         const int batch_size, const int beam_width,
                         const int vocab_size, cudaStream_t stream,
                         const int end_id, int* finished_count)
{

    printf("[INFO] decoding update check. \n");
    // CPU inputs
    float *h_log_probs = new float[batch_size * beam_width * vocab_size];
    int *h_ids = new int[batch_size * beam_width];
    bool *h_finished = new bool[batch_size * beam_width];
    int *h_parent_ids = new int[batch_size * beam_width];
    int *h_sequence_length = new int[batch_size * beam_width];
    int *h_output_ids = new int[batch_size * beam_width];

    // CPU output
    float *h_cum_log_probs_after_update_cpu = new float[batch_size * beam_width];
    bool *h_finished_after_update_cpu = new bool[batch_size * beam_width];
    int *h_parent_ids_after_update_cpu = new int[batch_size * beam_width];
    int *h_sequence_length_after_update_cpu = new int[batch_size * beam_width];
    int *h_output_ids_after_update_cpu = new int[batch_size * beam_width];

    // GPU output
    float *h_cum_log_probs_after_update = new float[batch_size * beam_width];
    bool *h_finished_after_update = new bool[batch_size * beam_width];
    int *h_parent_ids_after_update = new int[batch_size * beam_width];
    int *h_sequence_length_after_update = new int[batch_size * beam_width];
    int *h_output_ids_after_update = new int[batch_size * beam_width];

    // copy to CPU input
    check_cuda_error(cudaMemcpy(h_log_probs, log_probs, sizeof(float) * batch_size * beam_width * vocab_size, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_ids, ids, sizeof(int) * batch_size * beam_width, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_finished, finished, sizeof(bool) * batch_size * beam_width, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_parent_ids, parent_ids, sizeof(int) * batch_size * beam_width, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_sequence_length, sequence_length, sizeof(int) * batch_size * beam_width, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_output_ids, output_ids, sizeof(int) * batch_size * beam_width, cudaMemcpyDeviceToHost));

    // compute on GPU and copy to GPU output
    update_kernelLauncher(log_probs, cum_log_probs, finished, parent_ids, sequence_length, word_ids, output_ids,
           batch_size, beam_width, vocab_size, stream, end_id, finished_count);
    cudaDeviceSynchronize();
    check_cuda_error(cudaGetLastError());

    check_cuda_error(cudaMemcpy(h_cum_log_probs_after_update, cum_log_probs, sizeof(float) * batch_size * beam_width, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_finished_after_update, finished, sizeof(bool) * batch_size * beam_width, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_parent_ids_after_update, parent_ids, sizeof(int) * batch_size * beam_width, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_sequence_length_after_update, sequence_length, sizeof(int) * batch_size * beam_width, cudaMemcpyDeviceToHost));
    check_cuda_error(cudaMemcpy(h_output_ids_after_update, output_ids, sizeof(int) * batch_size * beam_width, cudaMemcpyDeviceToHost));

    // compute on CPU
    for (int i = 0; i < batch_size * beam_width; i++)
    {
        if (h_finished[i] == false)
            h_sequence_length[i] = h_sequence_length[i] + 1;
        else
            h_sequence_length[i] = h_sequence_length[i];
    }

    for (int i = 0; i < batch_size * beam_width; i++)
    {
        int sample_id = h_ids[i];
        int word_id = h_ids[i] % vocab_size;
        int beam_indices = h_ids[i] / vocab_size;
        h_cum_log_probs_after_update_cpu[i] = h_log_probs[sample_id];
        h_finished_after_update_cpu[i] = h_finished[beam_indices];
        h_sequence_length_after_update_cpu[i] = h_sequence_length[beam_indices];
        h_parent_ids_after_update_cpu[i] = beam_indices;
        h_output_ids_after_update_cpu[i] = word_id;

        printf("[INFO] sample id %d, word id %d, beam id %d, with log prob: %f \n", sample_id, word_id, beam_indices, (float)h_log_probs[sample_id]);
    }

    for (int i = 0; i < batch_size * beam_width; i++)
    {
        if (h_parent_ids_after_update[i] != h_parent_ids_after_update_cpu[i])
        {
            printf("[ERROR] update %d parent_ids fails: %d %d. \n", i, h_parent_ids_after_update_cpu[i], h_parent_ids_after_update[i]);
            exit(0);
        }

        if (h_output_ids_after_update[i] != h_output_ids_after_update_cpu[i])
        {
            printf("[ERROR] update %d output_ids fails: %d %d. \n", i, h_output_ids_after_update_cpu[i], h_output_ids_after_update[i]);
            exit(0);
        }

        if (h_cum_log_probs_after_update[i] != h_cum_log_probs_after_update_cpu[i])
        {
            printf("[ERROR] update %d cum log probs fails: %f %f. \n", i, (float)h_cum_log_probs_after_update_cpu[i], (float)h_cum_log_probs_after_update[i]);
            exit(0);
        }

        if (h_finished_after_update[i] != h_finished_after_update_cpu[i])
        {
            printf("[ERROR] update %d finished fails: %d %d. \n", i, h_finished_after_update_cpu[i], h_finished_after_update[i]);
            exit(0);
        }

        if (h_sequence_length_after_update[i] != h_sequence_length_after_update_cpu[i])
        {
            printf("[ERROR] update %d sequence length fails: %d %d. \n", i, h_sequence_length_after_update_cpu[i], h_sequence_length_after_update[i]);
            exit(0);
        }
    }

    delete[] h_log_probs;
    delete[] h_ids;
    delete[] h_finished;
    delete[] h_parent_ids;
    delete[] h_sequence_length;
    delete[] h_output_ids;

    delete[] h_cum_log_probs_after_update_cpu;
    delete[] h_finished_after_update_cpu;
    delete[] h_parent_ids_after_update_cpu;
    delete[] h_sequence_length_after_update_cpu;
    delete[] h_output_ids_after_update_cpu;

    delete[] h_cum_log_probs_after_update;
    delete[] h_finished_after_update;
    delete[] h_parent_ids_after_update;
    delete[] h_sequence_length_after_update;
    delete[] h_output_ids_after_update;
    printf("[INFO] decoding update check finish. \n");
}

} // end of namespace fastertransformer
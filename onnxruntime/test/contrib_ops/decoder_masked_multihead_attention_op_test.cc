// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/platform/env_var_utils.h"
#include "gtest/gtest.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "test/contrib_ops/attention_op_test_helper.h"

namespace onnxruntime {
using contrib::AttentionMaskType;

namespace test {

    template <typename T>
    static std::vector<T> CreateOnes(int size) {
        std::vector<T> f;
        f.reserve(size);

        for (int i = 0; i < size; ++i) {
            f.push_back(T(1));
        }

        return f;
    }

    template <typename T>
    static std::vector<T> CreateValues(int size, int val) {
        std::vector<T> f;
        f.reserve(size);

        for (int i = 0; i < size; ++i) {
            f.push_back(T(val));
        }

        return f;
    }

    template <typename T>
        static std::vector<T> CreateRandom(int size) {
            std::vector<T> f;
            f.reserve(size);

            for (int i = 0; i < size; ++i) {
                f.push_back(T(i));
            }

            return f;
    }

    template<typename T>
    static void QKV(std::vector<T>& input, std::vector<T>& weights, std::vector<T>& bias, 
                                 int batch_size, int sequence_length, int hidden_size, /*out*/std::vector<T>& qkv) {
        qkv.resize(batch_size * sequence_length * 3 * hidden_size, 0);

        for (int b = 0; b < batch_size; ++b) {
            for (int i = 0; i < sequence_length; ++i) {
                for (int j = 0; j < 3 * hidden_size; ++j) {

                    T sum = 0;

                    for (int k = 0; k < hidden_size; ++k) {
                        sum += input[b * sequence_length * hidden_size + i * hidden_size + k] * weights[k * 3 * hidden_size + j];
                    }

                    qkv[b * sequence_length * 3 * hidden_size + i * 3 * hidden_size + j] = sum + bias[j];
                }
            }

        }

    }

    // Reorder from [B, N, S, H] to [B, N, H/x, S, x]
    // where x = (sizeof(T) / 16);
    template<typename T>
    static std::vector<T> ReorderKCache(std::vector<T>& unordered_k_cache,
        int batch_size, int num_heads, int sequence_length,
        int head_size, int max_sequence_length) {

        std::vector<T> ordered(unordered_k_cache.size(), 0);

        int num_inner_elements = 16 / sizeof(T);
        int num_iter = head_size / num_inner_elements;

        for (int b = 0; b < batch_size; ++b) {
            for (int h = 0; h < num_heads; ++h) {                
                for (int i = 0; i < num_iter; ++i) {
                    for (int s = 0; s < sequence_length; ++s) {

                        int base_offset = (b * num_heads * max_sequence_length * head_size) +
                            (h * max_sequence_length * head_size);

                        int input_base_offset = base_offset + (s * head_size) + (i * num_inner_elements);
                        int output_base_offset = base_offset + (i * max_sequence_length * num_inner_elements) + (s * num_inner_elements);

                        for (int e = 0; e < num_inner_elements; ++e) {
                            ordered[output_base_offset + e] = unordered_k_cache[input_base_offset + e];
                        }
                    }

                }

            }
        }

        return ordered;
    }

    // Merge [B, N, H/x, max_sequence_length (S), x] with [B, N, H/x, 1, x]
    // and create [B, N, H/x, max_sequence_length(S+1), x]
    template <typename T>
    static std::vector<T> MergeReorderedKCacheWithK(std::vector<T>& ordered_k_cache,
        float* k,
        int batch_size, int num_heads, 
        int past_sequence_length, int max_sequence_length,
        int head_size) {

        std::vector<T> merged = ordered_k_cache;

        int num_inner_elements = 16 / sizeof(T);
        int num_iter = head_size / num_inner_elements;

        int num_elements = batch_size * num_heads * 1 * head_size;
        int iter = past_sequence_length * num_inner_elements;

        while (num_elements > 0) {

            for (int i = 0; i < num_inner_elements; ++i) {
                merged[iter + i] = k[i];
            }

            num_elements -= num_inner_elements;
            k += num_inner_elements;
            iter += max_sequence_length * num_inner_elements;
        }

        return merged;
    }

    template<typename T>
    static std::vector<float> MergePastKWithPresentKAndTranspose(float* past_k, float* present_k, 
                                                   int num_batch, int num_heads, 
                                                   int past_sequence_length, int max_sequence_length, 
                                                   int head_size) {


        int total_seq_length = (past_sequence_length + 1);
        std::vector<T> merged_k(num_batch * num_heads * total_seq_length * head_size, 0);
        std::vector<T> transposed_merged_k(num_batch * num_heads * total_seq_length * head_size, 0);

        for (int b = 0; b < num_batch; ++b) {
            for (int n = 0; n < num_heads; ++n) {
                for (int s = 0; s < total_seq_length; ++s) {
                    for (int h = 0; h < head_size; ++h) {
                        float input_value = 0.f;

                        if (s < past_sequence_length) {
                            int input_offset = b * num_heads * max_sequence_length * head_size
                                + (n * max_sequence_length * head_size)
                                + (s * head_size) 
                                + h;
                            input_value = past_k[input_offset];
                        }
                        else {
                            int input_offset = b * num_heads * 1 * head_size
                                + (n * 1 * head_size)
                                + h;
                            input_value = present_k[input_offset];
                        }

                        int output_offset = b * num_heads * total_seq_length * head_size
                            + (n * total_seq_length * head_size)
                            + (s * head_size)
                            + h;

                        merged_k[output_offset] = input_value;

                    }
                }
            }
        }


        for (int b = 0; b < num_batch; ++b) {
            for (int n = 0; n < num_heads; ++n) {
                int base_offset = (b * num_heads * total_seq_length * head_size) +
                    (n * total_seq_length * head_size);

                for (int s = 0; s < total_seq_length; ++s) {
                    for (int h = 0; h < head_size; ++h) {
                        int input_offset = base_offset + (s * head_size) + h;
                        int output_offset = base_offset + (h * total_seq_length) + s;
                            transposed_merged_k[output_offset] = merged_k[input_offset];

                    }
                }
            }
        }

        return transposed_merged_k;
    }

    template<typename T>
    std::vector<T> QK_Transpose(float* q_matrix, float* k_transpose_matrix, 
        int batch_size, int num_heads,
        int sequence_length, int total_sequence_length, int head_size) {
        
        if (sequence_length != 1) {
            throw std::exception("Not supported");
        }

        std::vector<T> qk_transpose;
        qk_transpose.resize(batch_size * num_heads * sequence_length * total_sequence_length, 0);

        for (int b = 0; b < batch_size; ++b) {
            for (int n = 0; n < num_heads; ++n) {
                int input_1_base_offset = (b * num_heads * sequence_length * head_size) +
                                        (n * sequence_length * head_size);

                int input_2_base_offset = (b * num_heads * total_sequence_length * head_size) +
                    (n * total_sequence_length * head_size);

                int output_base_offset = (b * num_heads * sequence_length * total_sequence_length) +
                    (n * sequence_length * total_sequence_length);


                for (int i = 0; i < sequence_length; ++i) {
                    for (int j = 0; j < total_sequence_length; ++j) {

                        T sum = 0;
                        for (int k = 0; k < head_size; ++k) {

                            sum += (q_matrix[input_1_base_offset + i * head_size + k] *
                                k_transpose_matrix[input_2_base_offset + k * total_sequence_length + j]);
                        }

                        float scale = 1 / sqrt(static_cast<float>(head_size));
                        qk_transpose[output_base_offset + i * total_sequence_length + j] = scale * sum;
                    }
                }
            }

        }

        return qk_transpose;
    }
    TEST(AttentionTest, Test) {
        int batch_size = 1;
        int sequence_length = 1;
        int hidden_size = 128;
        int number_of_heads = 2;
        int head_size = (hidden_size / number_of_heads);
        int past_sequence_length = 2;
        int total_sequence_length = sequence_length + past_sequence_length;
        int max_sequence_length = 4;

        /*
        int batch_size = 1;
        int sequence_length = 1;
        int hidden_size = 768;
        int number_of_heads = 12;
        int head_size = (hidden_size / number_of_heads);
        int past_sequence_length = 3;
        int total_sequence_length = sequence_length + past_sequence_length;
        int max_sequence_length = 10;
        */

        OpTester tester("DecoderMaskedSelfAttention", 1, onnxruntime::kMSDomain);
        tester.AddAttribute<int64_t>("num_heads", static_cast<int64_t>(number_of_heads));
        tester.AddAttribute<int64_t>("past_present_share_buffer", static_cast<int64_t>(1));

        std::vector<int64_t> input_dims = { batch_size, sequence_length, hidden_size };
        std::vector<int64_t> weights_dims = { hidden_size, 3 * hidden_size };
        std::vector<int64_t> bias_dims = { 3 * hidden_size };
        std::vector<int64_t> output_dims = { batch_size, sequence_length, hidden_size };

        auto input = CreateOnes<float>(batch_size * sequence_length * hidden_size);
        tester.AddInput<float>("input", input_dims, input);

        auto weight = CreateOnes<float>(hidden_size * 3 * hidden_size);
        tester.AddInput<float>("weight", weights_dims, weight);

        auto bias = CreateOnes<float>(3 * hidden_size);
        tester.AddInput<float>("bias", bias_dims, bias);

        // Mask
        tester.AddOptionalInputEdge<int32_t>();

        // Past
        std::vector<int64_t> past_dims = { 2, batch_size, number_of_heads, max_sequence_length, head_size };
        int past_present_size = 2 * batch_size * number_of_heads * max_sequence_length * head_size;

        auto kv_cache = CreateRandom<float>(past_present_size);

        auto reordered_kv_cache = ReorderKCache(kv_cache, batch_size,
            number_of_heads, past_sequence_length, head_size, max_sequence_length);
        
        tester.AddInput<float>("past", past_dims, reordered_kv_cache);


        // DEBUG
        //auto random_past_present = ReorderKCache(CreateRandom<float>(1 * 2 * 4 * 8), 1, 2, 2, 8, 4); 
        //auto merge_k = CreateValues<float>(1 * 2 * 1 * 8, -1);

        //auto result = MergeReorderedKCacheWithK(random_past_present, merge_k.data(), 1, 2, 2, 8, 4);

        // Rel 
        tester.AddOptionalInputEdge<float>();

        // Past sequence length
        std::vector<int32_t> arr_past_sequence_len(1, past_sequence_length);
        tester.AddInput<int32_t>("past_sequence_length", {1}, arr_past_sequence_len);

        // QKV MatMul
        std::vector<float> qkv;
        QKV(input, weight, bias, batch_size, sequence_length, hidden_size, qkv);
        auto k_transpose = MergePastKWithPresentKAndTranspose<float>(kv_cache.data(), qkv.data() + hidden_size, batch_size, number_of_heads, past_sequence_length, max_sequence_length, head_size);
        auto qk_transpose = QK_Transpose<float>(qkv.data(), k_transpose.data(), batch_size, number_of_heads, sequence_length, total_sequence_length, head_size);


        // Output(s)
        tester.AddOutput<float>("output", {batch_size, number_of_heads, sequence_length, total_sequence_length}, qk_transpose);

        auto present = MergeReorderedKCacheWithK(reordered_kv_cache, qkv.data() + hidden_size, batch_size, number_of_heads, past_sequence_length, max_sequence_length, head_size);
        tester.AddOutput<float>("present", past_dims, present);
        
        // Run
        std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
        execution_providers.push_back(DefaultCudaExecutionProvider());
        tester.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
    }

    }
}
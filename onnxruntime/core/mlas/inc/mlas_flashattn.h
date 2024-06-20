/*++

Copyright (c) Microsoft Corporation. All rights reserved.

Licensed under the MIT License.

Module Name:

    mlas_flashattn.h

Abstract:

    Utilities for FlashAttention on CPU. Used internally
    by MLAS on platforms without half precision support.  Provided here as
    convenience for tests or other client libraries/apps.

--*/

#pragma once

struct FlashAttentionThreadedArgs {
    int batch_size;
    int num_heads;
    int q_sequence_length;
    int kv_sequence_length;
    int qk_head_size;
    int v_head_size;
    int row_size_q;
    int row_size_kv;
    float scale;
    int thread_count;
    float* buffer;
    size_t buffer_size_per_thread;
    const float* query;
    const float* key;
    const float* value;
    float* output;
};

void FlashAttentionThreaded(
    std::ptrdiff_t thread_id,
    struct FlashAttentionThreadedArgs* args
);

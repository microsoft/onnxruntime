// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

namespace onnxruntime {
namespace contrib {

struct AttentionParameters {
  int batch_size;
  int sequence_length;
  int kv_sequence_length;     // input sequence length of K or V
  int past_sequence_length;   // sequence length in past state of K or V
  int total_sequence_length;  // total sequence length of K or V
  int max_sequence_length;
  int input_hidden_size;
  int hidden_size;    // hidden size of Q or K
  int head_size;      // hidden size per head of Q or K
  int v_hidden_size;  // hidden size of V
  int v_head_size;    // hidden size per head of V
  int num_heads;
  bool is_unidirectional;
};

}  // namespace contrib
}  // namespace onnxruntime

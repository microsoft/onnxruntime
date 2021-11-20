// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {

struct BeamSearchParameters {
  // from node attributes
  int eos_token_id;
  int pad_token_id;
  int no_repeat_ngram_size;
  bool early_stopping;

  // from inputs
  int min_length;
  int max_length;
  int num_beams;
  int num_return_sequences;
  float temperature;
  float length_penalty;
  float repetition_penalty;
  int batch_size;           // deduce from first dimension of input_ids
  int sequence_length;      // deduce from second dimension of input_ids
  
  gsl::span<const int32_t> vocab_mask;

  // from outputs
  bool output_scores;       // whether scores existed in output

  // deduce from subgraph
  int num_heads;
  int head_size;
  int vocab_size;
  int num_layers;

  Status Validate();

  void ParseFromAttributes(const OpKernelInfo& info);
  void ParseFromInputs(OpKernelContext* context);
  void SetSubgraphParameters(int num_heads, int head_size, int vocab_size, int num_layers);
};

}  // namespace contrib
}  // namespace onnxruntime

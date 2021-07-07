// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <stdint.h>
#include <vector>

namespace onnxruntime {
namespace test {
namespace embedlayernorm {

constexpr float kEpsilon = 1e-12f;

// EmbedLayerNorm and QEmbedLayerNorm contain many inputs and outputs. This
// utility class helps readability of Op unit tests by wrapping Op data.
class OpData {
 public:
  explicit OpData(
      int batch_size,
      int sequence_size,
      int hidden_size,
      const std::vector<int32_t>& input_ids_data,
      const std::vector<int32_t>& segment_ids_data,
      const std::vector<int32_t>& mask_data,
      const std::vector<float>& word_embedding_data,
      const std::vector<float>& position_embedding_data,
      const std::vector<float>& segment_embedding_data,
      const std::vector<float>& gamma_data,
      const std::vector<float>& beta_data,
      const std::vector<float>& output_data,
      const std::vector<int32_t>& mask_index_data,
      float epsilon = kEpsilon,
      bool has_mask = true,
      bool has_segment = true)
      : batch_size(batch_size)
      , sequence_size(sequence_size)
      , hidden_size(hidden_size)
      , input_ids_data(input_ids_data)
      , segment_ids_data(segment_ids_data)
      , mask_data(mask_data)
      , word_embedding_data(word_embedding_data)
      , position_embedding_data(position_embedding_data)
      , segment_embedding_data(segment_embedding_data)
      , gamma_data(gamma_data)
      , beta_data(beta_data)
      , output_data(output_data)
      , mask_index_data(mask_index_data)
      , epsilon(epsilon)
      , has_mask(has_mask)
      , has_segment(has_segment)
  {}

  const int batch_size;
  const int sequence_size;
  const int hidden_size;
  const std::vector<int32_t> input_ids_data;
  const std::vector<int32_t> segment_ids_data;
  const std::vector<int32_t> mask_data;
  const std::vector<float> word_embedding_data;
  const std::vector<float> position_embedding_data;
  const std::vector<float> segment_embedding_data;
  const std::vector<float> gamma_data;
  const std::vector<float> beta_data;
  const std::vector<float> output_data;
  const std::vector<int32_t> mask_index_data;
  const float epsilon;
  const bool has_mask = true;
  const bool has_segment = true;
};

inline OpData EmbedLayerNormBatch1() {
  int batch_size = 1;
  int sequence_size = 2;
  int hidden_size = 4;

  std::vector<int32_t> input_ids_data = {
      1, 3};

  std::vector<int32_t> segment_ids_data = {
      0, 1};

  std::vector<int32_t> mask_data = {
      1, 1};

  std::vector<float> word_embedding_data = {
      0.2f, 0.1f, 0.4f, -0.6f,
      0.3f, 0.2f, 0.5f, 0.6f,
      0.6f, 0.7f, 0.0f, -0.1f,
      0.8f, 0.6f, 0.9f, 1.2f,
      0.1f, 0.3f, 0.5f, 0.9f,
      1.0f, -2.0f, 1.1f, 0.8f};

  std::vector<float> position_embedding_data = {
      0.1f, 0.1f, 0.4f, 0.6f,
      0.6f, 0.0f, 0.8f, 0.6f,
      0.3f, 0.9f, -2.0f, 0.8f};

  std::vector<float> segment_embedding_data = {
      0.3f, 0.4f, 0.9f, 0.1f,
      0.7f, 0.3f, 0.5f, 0.2f};

  std::vector<float> gamma_data = {
      0.25f, 0.15f, 0.45f, -0.66f};

  std::vector<float> beta_data = {
      0.6f, 0.2f, 0.5f, -0.6f};

  std::vector<float> output_data = {
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337};

  std::vector<int32_t> mask_index_data = {
      2};

  return OpData(batch_size, sequence_size, hidden_size, input_ids_data, segment_ids_data,
                mask_data, word_embedding_data, position_embedding_data, segment_embedding_data,
                gamma_data, beta_data, output_data, mask_index_data);
}

inline OpData EmbedLayerNormBatch2(bool has_mask=true) {
  int batch_size = 3;
  int sequence_size = 2;
  int hidden_size = 4;

  std::vector<int32_t> input_ids_data = {
      1, 3,
      1, 3,
      2, 0};

  std::vector<int32_t> segment_ids_data = {
      0, 1,
      0, 1,
      0, 0};

  std::vector<int32_t> mask_data = {};
  if (has_mask) {
    mask_data = {
        1, 1,
        1, 1,
        1, 0};
  }

  std::vector<float> word_embedding_data = {
      0.2f, 0.1f, 0.4f, -0.6f,
      0.3f, 0.2f, 0.5f, 0.6f,
      0.6f, 0.7f, 0.0f, -0.1f,
      0.8f, 0.6f, 0.9f, 1.2f,
      0.1f, 0.3f, 0.5f, 0.9f,
      1.0f, -2.0f, 1.1f, 0.8f};

  std::vector<float> position_embedding_data = {
      0.1f, 0.1f, 0.4f, 0.6f,
      0.6f, 0.0f, 0.8f, 0.6f,
      0.3f, 0.9f, -2.0f, 0.8f};

  std::vector<float> segment_embedding_data = {
      0.3f, 0.4f, 0.9f, 0.1f,
      0.7f, 0.3f, 0.5f, 0.2f};

  std::vector<float> gamma_data = {
      0.25f, 0.15f, 0.45f, -0.66f};

  std::vector<float> beta_data = {
      0.6f, 0.2f, 0.5f, -0.6f};

  std::vector<float> output_data = {
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337,
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337,
      0.57668739557266235, 0.2979130744934082, 0.96158987283706665, 0.44627034664154053,
      0.64977931976318359, 0.11039737612009048, 1.1869535446166992, 0.14469735324382782};

  std::vector<int32_t> mask_index_data;
  if (has_mask) {
    mask_index_data = {2, 2, 1};
  } else {
    mask_index_data = {0, 0, 0};
  }

  return OpData(batch_size, sequence_size, hidden_size, input_ids_data, segment_ids_data,
                mask_data, word_embedding_data, position_embedding_data, segment_embedding_data,
                gamma_data, beta_data, output_data, mask_index_data, kEpsilon, has_mask);
}

inline OpData EmbedLayerNormLargeBatchSmallHiddenSize() {
  int batch_size = 5;
  int sequence_size = 2;
  int hidden_size = 4;

  std::vector<int32_t> input_ids_data = {
      1, 3,
      1, 3,
      2, 0,
      1, 3,
      2, 0};

  std::vector<int32_t> segment_ids_data = {
      0, 1,
      0, 1,
      0, 0,
      0, 1,
      0, 0};

  std::vector<int32_t> mask_data = {
      1, 1,
      1, 1,
      1, 0,
      1, 1,
      1, 0};

  std::vector<float> word_embedding_data = {
      0.2f, 0.1f, 0.4f, -0.6f,
      0.3f, 0.2f, 0.5f, 0.6f,
      0.6f, 0.7f, 0.0f, -0.1f,
      0.8f, 0.6f, 0.9f, 1.2f,
      0.1f, 0.3f, 0.5f, 0.9f,
      1.0f, -2.0f, 1.1f, 0.8f};

  std::vector<float> position_embedding_data = {
      0.1f, 0.1f, 0.4f, 0.6f,
      0.6f, 0.0f, 0.8f, 0.6f,
      0.3f, 0.9f, -2.0f, 0.8f};

  std::vector<float> segment_embedding_data = {
      0.3f, 0.4f, 0.9f, 0.1f,
      0.7f, 0.3f, 0.5f, 0.2f};

  std::vector<float> gamma_data = {
      0.25f, 0.15f, 0.45f, -0.66f};

  std::vector<float> beta_data = {
      0.6f, 0.2f, 0.5f, -0.6f};

  std::vector<float> output_data = {
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337,
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337,
      0.57668739557266235, 0.2979130744934082, 0.96158987283706665, 0.44627034664154053,
      0.64977931976318359, 0.11039737612009048, 1.1869535446166992, 0.14469735324382782,
      0.36917170882225037, 0.061503000557422638, 1.1598974466323853, -0.85092413425445557,
      0.74301940202713013, -0.057434864342212677, 0.84324657917022705, -0.85171419382095337,
      0.57668739557266235, 0.2979130744934082, 0.96158987283706665, 0.44627034664154053,
      0.64977931976318359, 0.11039737612009048, 1.1869535446166992, 0.14469735324382782};

  std::vector<int32_t> mask_index_data = {
      2, 2, 1, 2, 1};

  return OpData(batch_size, sequence_size, hidden_size, input_ids_data, segment_ids_data,
                mask_data, word_embedding_data, position_embedding_data, segment_embedding_data,
                gamma_data, beta_data, output_data, mask_index_data);
}

inline OpData EmbedLayerNormBatch_Distill() {
  int batch_size = 3;
  int sequence_size = 2;
  int hidden_size = 4;

  std::vector<int32_t> input_ids_data = {
      1, 3,
      1, 3,
      2, 0};

  std::vector<int32_t> segment_ids_data = {};

  std::vector<int32_t> mask_data = {
      1, 1,
      1, 1,
      1, 0};

  std::vector<float> word_embedding_data = {
      0.2f, 0.1f, 0.4f, -0.6f,
      0.3f, 0.2f, 0.5f, 0.6f,
      0.6f, 0.7f, 0.0f, -0.1f,
      0.8f, 0.6f, 0.9f, 1.2f,
      0.1f, 0.3f, 0.5f, 0.9f,
      1.0f, -2.0f, 1.1f, 0.8f};

  std::vector<float> position_embedding_data = {
      0.1f, 0.1f, 0.4f, 0.6f,
      0.6f, 0.0f, 0.8f, 0.6f,
      0.3f, 0.9f, -2.0f, 0.8f};

  std::vector<float> segment_embedding_data = {};

  std::vector<float> gamma_data = {
      0.25f, 0.15f, 0.45f, -0.66f};

  std::vector<float> beta_data = {
    0.6f, 0.2f, 0.5f, -0.6f};

  std::vector<float> output_data = {
      0.39587587118148804, 0.03670068085193634, 0.7449488639831543, -1.4981462955474854,
      0.61326867341995239, -0.046796366572380066, 0.81048583984375, -1.1954958438873291,
      0.39587587118148804, 0.03670068085193634, 0.7449488639831543, -1.4981462955474854,
      0.61326867341995239, -0.046796366572380066, 0.81048583984375, -1.1954958438873291,
      0.75811392068862915, 0.38973665237426758, -0.069209933280944824, -0.18257927894592285,
      0.73836749792098999, 0.071695566177368164, 1.111332893371582, 0.097372293472290039};

  std::vector<int32_t> mask_index_data = {
      2, 2, 1};

  return OpData(batch_size, sequence_size, hidden_size, input_ids_data, segment_ids_data,
                mask_data, word_embedding_data, position_embedding_data, segment_embedding_data,
                gamma_data, beta_data, output_data, mask_index_data, kEpsilon,
                /*has_mask=*/true,
                /*has_segment=*/false);
}

}  // namespace embedlayernorm
}  // namespace test
}  // namespace onnxruntime

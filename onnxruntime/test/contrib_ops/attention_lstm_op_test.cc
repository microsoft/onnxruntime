// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include <iterator>
#include <vector>
#include <string>
#include <algorithm>

namespace onnxruntime {
namespace test {

// copy the contents of the container to the end so the original values are duplicated
template <typename T>
T DuplicateContainer(const T& container) {
  T doubled;
  doubled.reserve(container.size() * 2);  // need to avoid reallocation when inserting
  std::copy(container.cbegin(), container.cend(), std::back_inserter(doubled));
  std::copy(container.cbegin(), container.cend(), std::back_inserter(doubled));

  return doubled;
}

// Test data is generated using attention_lstm_data_gen.py, which covers the all options
// that need to be specified to generate different cases here. Also from the file, it is
// easy to understand how the operate works in Tensorflow.
static void RunAttnLstmTest(
    const std::vector<float>& X_data,
    const std::vector<float>& W_data,
    const std::vector<float>& R_data,
    const std::vector<float>& Y_data,
    const std::vector<float>& Y_h_data,
    const std::vector<float>& Y_c_data,

    const std::vector<float>& MW_data,                // memory layer weight: [num_directions, memory_depth,  am_attn_depth]
    const std::vector<float>& QW_data,                // query layer weight: [num_directions, query_layer_depth=cell_hidden_size,  am_attn_depth]
    const std::vector<float>& attn_v_data,            // [num_directions, am_attn_depth]
    const std::vector<float>& M_data,                 // memory sequence: [batch_size, maxMemoryTimeStep, memory_depth]
    const std::vector<int>* memory_sequence_lengths,  // [batch_size]
    const std::vector<float>* attn_layer_weights,     // [num_directions, cell_hidden_size + memory_depth, aw_hidden_depth]

    int64_t x_depth,
    int batch_size,
    int64_t hidden_size,
    int64_t seq_length,

    int64_t memory_max_time,
    int64_t memory_depth,
    int64_t am_attn_size,
    int64_t aw_attn_size,

    const std::vector<float>* B_data = nullptr,
    const std::vector<float>* P_data = nullptr,
    const std::vector<float>* initial_h_data = nullptr,
    const std::vector<float>* initial_c_data = nullptr,
    const std::vector<int>* sequence_lengths = nullptr,
    const std::string& direction = "forward",
    float clip = -9999.f,
    bool output_sequence = true,
    bool input_forget = false,
    // copy the following vectors as we may modify them
    std::vector<std::string> activations = {},
    std::vector<float> activation_alphas = {},
    std::vector<float> activation_betas = {}) {
  const int64_t input_size = x_depth + aw_attn_size;

  OpTester test("AttnLSTM", 1, onnxruntime::kMSDomain);

  int num_directions = (direction == "bidirectional") ? 2 : 1;

  if (activations.empty()) {
    activations = {"sigmoid", "tanh", "tanh"};
  }

  if (num_directions == 2 && activations.size() == 3) {
    activations = DuplicateContainer(activations);
  }

  test.AddAttribute<std::vector<std::string>>("activations", activations);
  if (!activation_alphas.empty())
    test.AddAttribute<std::vector<float>>("activation_alpha", activation_alphas);
  if (!activation_betas.empty())
    test.AddAttribute<std::vector<float>>("activation_beta", activation_betas);

  test.AddAttribute("direction", direction);
  test.AddAttribute("hidden_size", hidden_size);
  // test.AddAttribute<int64_t>("output_sequence", output_sequence);
  test.AddAttribute<int64_t>("input_forget", input_forget);
  if (clip < 0.0f) clip = std::numeric_limits<float>::max();
  test.AddAttribute<float>("clip", clip);

  std::vector<int64_t> X_dims = {seq_length, batch_size, x_depth};
  std::vector<int64_t> W_dims = {num_directions, 4 * hidden_size, input_size};
  std::vector<int64_t> R_dims = {num_directions, 4 * hidden_size, hidden_size};

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("W", W_dims, W_data);
  test.AddInput<float>("R", R_dims, R_data);

  if (B_data) {
    std::vector<int64_t> B_dims = {num_directions, 8 * hidden_size};
    test.AddInput<float>("B", B_dims, *B_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  if (sequence_lengths) {
    std::vector<int64_t> sequence_lens_dims{batch_size};
    test.AddInput<int>("sequence_lens", sequence_lens_dims, *sequence_lengths);
  } else {
    test.AddMissingOptionalInput<int>();
  }

  if (initial_h_data && !initial_h_data->empty()) {
    std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
    test.AddInput<float>("initial_h", initial_h_dims, *initial_h_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  if (initial_c_data && !initial_c_data->empty()) {
    std::vector<int64_t> initial_c_dims = {num_directions, batch_size, hidden_size};
    test.AddInput<float>("initial_c", initial_c_dims, *initial_c_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  if (P_data && !P_data->empty()) {
    std::vector<int64_t> P_dims = {num_directions, 3 * hidden_size};
    test.AddInput<float>("P", P_dims, *P_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  std::vector<int64_t> QW_dims{num_directions, hidden_size, am_attn_size};
  test.AddInput<float>("QW", QW_dims, QW_data);

  std::vector<int64_t> MW_dims{num_directions, memory_depth, am_attn_size};
  test.AddInput<float>("MW", MW_dims, MW_data);

  std::vector<int64_t> attn_v_dims{num_directions, am_attn_size};
  test.AddInput<float>("V", attn_v_dims, attn_v_data);

  std::vector<int64_t> M_dims{batch_size, memory_max_time, memory_depth};
  test.AddInput<float>("M", M_dims, M_data);

  if (memory_sequence_lengths) {
    std::vector<int64_t> M_seq_dims{batch_size};
    test.AddInput<int>("memory_seq_lens", M_seq_dims, *memory_sequence_lengths);
  } else {
    test.AddMissingOptionalInput<int>();
  }

  if (attn_layer_weights) {
    std::vector<int64_t> attn_layer_weight_dims{num_directions, memory_depth + hidden_size, aw_attn_size};
    test.AddInput<float>("AW", attn_layer_weight_dims, *attn_layer_weights);
  } else {
    test.AddMissingOptionalInput<int>();
  }

  if (output_sequence != 0 && !Y_data.empty()) {
    std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y", Y_dims, Y_data);
  } else {
    // add placeholder so node counts match as Y_h will always be the second Y_data,
    // so Y must exist as the first Y_data
    test.AddMissingOptionalOutput<float>();
  }

  if (!Y_h_data.empty()) {
    std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);
  } else {
    test.AddMissingOptionalOutput<float>();
  }

  if (!Y_c_data.empty()) {
    std::vector<int64_t> Y_c_dims{num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y_c", Y_c_dims, Y_c_data);
  } else {
    test.AddMissingOptionalOutput<float>();
  }

  test.Run();
}

template <typename T>
static std::vector<T> ConcatDup(const std::vector<T>& src) {
  std::vector<T> dst(2 * src.size());
  std::copy(src.cbegin(), src.cend(), dst.begin());
  std::copy(src.cbegin(), src.cend(), dst.begin() + src.size());
  return dst;
}

template <typename T>
static std::vector<T> ConcatLastDim(const std::vector<T>& a, const std::vector<T>& b, int depth) {
  std::vector<T> dst(2 * a.size());
  for (size_t s = 0; s < a.size(); s += depth) {
    std::copy(a.cbegin() + s, a.cbegin() + s + depth, dst.begin() + 2 * s);
    std::copy(b.cbegin() + s, b.cbegin() + s + depth, dst.begin() + 2 * s + depth);
  }
  return dst;
}

template <typename T>
static std::vector<T> Transpose2D(const std::vector<T>& src, int num_rows, int num_cols) {
  std::vector<T> dst(src.size());
  int pos = 0;
  for (int x = 0; x < num_rows; ++x) {
    for (int y = 0; y < num_cols; ++y) {
      dst[y * num_rows + x] = src[pos++];
    }
  }
  return dst;
}

template <typename T>
static std::vector<T> ConvertBatchSeqToSeqBatch(
    const std::vector<T>& bs, int batch_size, int max_step, int depth) {
  std::vector<T> sb(bs.size());
  int pos = 0;
  for (int b = 0; b < batch_size; ++b) {
    for (int t = 0; t < max_step; ++t) {
      for (int i = 0; i < depth; ++i) {
        int tp = t * batch_size * depth + b * depth + i;
        sb[tp] = bs[pos++];
      }
    }
  }
  return sb;
}

// This  convert seq in [*, 4*cell_hidden_size] from: I, J(C), F, O into: I O, F, C(J)
// for weights from semantic TF to Onnx semantic.
template <typename T>
static std::vector<T> ConvertIcfoToIofc(const std::vector<T>& icfo, int cell_hidden_size) {
  std::vector<T> iofc(icfo.size());
  for (size_t i = 0; i < icfo.size(); i += 4 * cell_hidden_size) {
    auto src = icfo.cbegin() + i;
    auto dst = iofc.begin() + i;

    std::copy(src + 0 * cell_hidden_size, src + 1 * cell_hidden_size, dst + 0 * cell_hidden_size);
    std::copy(src + 3 * cell_hidden_size, src + 4 * cell_hidden_size, dst + 1 * cell_hidden_size);
    std::copy(src + 2 * cell_hidden_size, src + 3 * cell_hidden_size, dst + 2 * cell_hidden_size);
    std::copy(src + 1 * cell_hidden_size, src + 2 * cell_hidden_size, dst + 3 * cell_hidden_size);
  }
  return iofc;
}

//Settings for this group of test data
static const int batch_size = 1;
static const int memory_max_step = 3;
static const int memory_depth = 3;
static const int input_max_step = 3;
static const int input_only_depth = 3;
static const int am_attn_size = 2;
static const int cell_hidden_size = 3;
static const int aw_attn_size = 2;
static const int input_size = input_only_depth + aw_attn_size;

// [batch_size=1, memory_max_step=3, memory_depth=3]
static std::vector<float> s_M_data{0.1f, -0.25f, 1.0f, 1.0f, -1.0f, -1.5f, 1.0f, 0.25f, -0.125f};
static const std::vector<float> s_M_2batch{0.1f, -0.25f, 1.0f, 1.0f, -1.0f, -1.5f, 1.0f, 0.25f, -0.125f,
                                           0.1f, -0.25f, 0.5f, -0.25f, -1.25f, 0.25f, -1.0f, 1.5f, -1.25f};

//real seq lens for memory
static std::vector<int> s_mem_seq_lenghts{3};
static const std::vector<int> s_mem_seq_lenghts_2batch{3, 2};

// [batch_size=1, input_max_step=3, input_only_depth=3]
static std::vector<float> s_X_T_data{
    0.25f,
    -1.5f,
    1.0f,
    0.25f,
    -0.5f,
    -1.5f,
    0.1f,
    1.5f,
    0.25f,
};

//real seq lens for X
static std::vector<int> s_seq_lengths{3};

// [num_directions, memory_depth=3, am_attn_size=2]
static std::vector<float> s_memory_layer_weight{4.0f, 2.0f, 0.5f, -8.0f, -2.0f, -2.0f};

// [num_directions, query_depth(cell_hidden_size)=3, am_attn_size=2]
static std::vector<float> s_query_layer_weight{-0.125f, -0.25f, 0.1f, -0.125f, -0.5f, 1.5f};

// [num_directions, memory_depth+cell_hidden_size=3+3=6, aw_attn_size=2]
static std::vector<float> s_attn_layer_weight{1.5f, 1.0f, 0.1f, -0.25f, 0.1f, 1.0f, -0.25f, -0.125f, -1.5f, -1.5f, -0.25f, 1.5f};

//[2]
static std::vector<float> s_attn_v{-0.25f, 0.1f};

//lstm kernel weights, [8, 12]  8 = x_depth + aw_attn_size + cell_hidden_size, 12 = 4 * cell_hidden_size (ijfo)
static std::vector<float> s_WR_T_data_ICFO{
    //  ---- x_depth lines of attention input weight
    -1.0f, -1.5f, -0.5f, -1.5f, 0.1f, -0.5f, 0.5f, -1.5f, -0.25f, 1.0f, -0.125f, -0.25f,
    -1.0f, -0.5f, 0.25f, -0.125f, -0.25f, -1.0f, 1.5f, 1.0f, -1.5f, 0.25f, 0.5f, 0.5f,
    1.5f, -0.5f, -1.0f, -0.5f, 0.1f, 1.0f, 0.1f, -0.5f, -0.125f, -1.5f, 0.1f, 1.5f,
    //  ---- aw_attn_size lines of attention input weight
    1.0f, -0.5f, -0.5f, -1.5f, -0.125f, -0.125f, 0.25f, -0.25f, -0.25f, 0.1f, -0.5f, -0.25f,
    0.25f, -0.5f, 0.1f, -0.5f, -0.25f, 0.25f, 0.1f, 0.5f, -1.5f, -0.125f, 1.5f, 0.5f,
    //  ---- cell_hidden_size lines of attention input weight
    -1.5f, 1.0f, 0.1f, -0.5f, -1.5f, 0.5f, -1.0f, 0.25f, -0.25f, 1.0f, 0.25f, 0.5f,
    -0.125f, 0.1f, -1.0f, -1.0f, 0.1f, 1.5f, -1.5f, 0.1f, 1.5f, 0.5f, 0.25f, 1.0f,
    1.0f, -1.5f, -0.25f, 0.5f, -0.25f, 1.0f, -1.0f, 0.25f, -0.5f, 0.5f, -1.5f, 0.5f};

//lstm_cell_bias, [12] = 4 * 3, append extra zero for onnix
std::vector<float> s_lstm_cell_bias_ICFO{
    0.25f, -0.25f, 0.1f, 1.0f, 1.5f, -1.5f, 1.5f, -1.0f, -0.25f, 1.0f, -0.25f, 1.0f,
    0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

TEST(AttnLSTMTest, ForwardLstmWithBahdanauAMZeroAttention) {
  std::vector<float> X_data = ConvertBatchSeqToSeqBatch(s_X_T_data, batch_size, input_max_step, input_only_depth);

  //clear attention layer weight, so that no attention value will be in effective
  std::vector<float> zero_attn_layer_weight(s_attn_layer_weight.size(), 0.0f);

  std::vector<float> WR_T_data = ConvertIcfoToIofc(s_WR_T_data_ICFO, cell_hidden_size);

  const size_t W_data_size = 5 * 12;
  std::vector<float> W_T_data(&(WR_T_data[0]), &(WR_T_data[0]) + W_data_size);
  // Fake zero for attention input weight now
  std::fill(W_T_data.begin() + 3 * 12, W_T_data.begin() + W_data_size, 0.0f);

  std::vector<float> R_T_data(&(WR_T_data[0]) + W_data_size, &(WR_T_data[0]) + WR_T_data.size());

  // transpose W and R for onnx sematic
  std::vector<float> W_data = Transpose2D(W_T_data, input_size, 4 * cell_hidden_size);
  std::vector<float> R_data = Transpose2D(R_T_data, cell_hidden_size, 4 * cell_hidden_size);

  std::vector<float> B_data = ConvertIcfoToIofc(s_lstm_cell_bias_ICFO, cell_hidden_size);

  // [1, 3, 3]
  std::vector<float> Y_T_data{
      0.0978363082f, 0.105625421f, 0.116753615f,
      0.277829766f, 0.166462898f, -0.119725525f,
      0.244476393f, 0.344343185f, -0.470122069f};

  // convert to onnx output semantic, should be same in this case.
  std::vector<float> Y_data = ConvertBatchSeqToSeqBatch(
      Y_T_data, batch_size, input_max_step, cell_hidden_size);

  const std::vector<float> Y_h_data{};
  const std::vector<float> Y_c_data{};

  RunAttnLstmTest(
      X_data, W_data, R_data, Y_data, Y_h_data, Y_c_data,
      s_memory_layer_weight, s_query_layer_weight, s_attn_v, s_M_data, &s_mem_seq_lenghts, &zero_attn_layer_weight,
      input_only_depth, batch_size, cell_hidden_size, input_max_step,
      memory_max_step, memory_depth, am_attn_size, aw_attn_size,
      &B_data, nullptr, nullptr, nullptr, &s_seq_lengths,
      "forward", -9999.f, true, false);
}

TEST(AttnLSTMTest, ForwardLstmWithBahdanauAM) {
  std::vector<float> X_data = ConvertBatchSeqToSeqBatch(s_X_T_data, batch_size, input_max_step, input_only_depth);

  std::vector<float> WR_T_data = ConvertIcfoToIofc(s_WR_T_data_ICFO, cell_hidden_size);

  const size_t W_data_size = 5 * 12;
  std::vector<float> W_T_data(&(WR_T_data[0]), &(WR_T_data[0]) + W_data_size);
  std::vector<float> R_T_data(&(WR_T_data[0]) + W_data_size, &(WR_T_data[0]) + WR_T_data.size());

  // transpose W and R for onnx sematic
  std::vector<float> W_data = Transpose2D(W_T_data, input_size, 4 * cell_hidden_size);
  std::vector<float> R_data = Transpose2D(R_T_data, cell_hidden_size, 4 * cell_hidden_size);

  std::vector<float> B_data = ConvertIcfoToIofc(s_lstm_cell_bias_ICFO, cell_hidden_size);

  // [1, 3, 3]
  std::vector<float> Y_T_data{
      0.0978363082f, 0.105625421f, 0.116753615f,
      0.236107856f, 0.195716992f, -0.133973882f,
      -0.029754376f, 0.274325848f, -0.387993187f};

  // convert to onnx output semantic, should be same in this case.
  std::vector<float> Y_data = ConvertBatchSeqToSeqBatch(
      Y_T_data, batch_size, input_max_step, cell_hidden_size);

  const std::vector<float> Y_h_data{};
  const std::vector<float> Y_c_data{};

  RunAttnLstmTest(
      X_data, W_data, R_data, Y_data, Y_h_data, Y_c_data,
      s_memory_layer_weight, s_query_layer_weight, s_attn_v, s_M_data, &s_mem_seq_lenghts, &s_attn_layer_weight,
      input_only_depth, batch_size, cell_hidden_size, input_max_step,
      memory_max_step, memory_depth, am_attn_size, aw_attn_size,
      &B_data, nullptr, nullptr, nullptr, &s_seq_lengths,
      "forward", -9999.f, true, false);
}

TEST(AttnLSTMTest, ForwardLstmWithBahdanauAMShortenSeqLength) {
  std::vector<float> X_data = ConvertBatchSeqToSeqBatch(s_X_T_data, batch_size, input_max_step, input_only_depth);

  std::vector<float> WR_T_data = ConvertIcfoToIofc(s_WR_T_data_ICFO, cell_hidden_size);

  const size_t W_data_size = 5 * 12;
  std::vector<float> W_T_data(&(WR_T_data[0]), &(WR_T_data[0]) + W_data_size);
  std::vector<float> R_T_data(&(WR_T_data[0]) + W_data_size, &(WR_T_data[0]) + WR_T_data.size());

  // transpose W and R for onnx sematic
  std::vector<float> W_data = Transpose2D(W_T_data, input_size, 4 * cell_hidden_size);
  std::vector<float> R_data = Transpose2D(R_T_data, cell_hidden_size, 4 * cell_hidden_size);

  std::vector<float> B_data = ConvertIcfoToIofc(s_lstm_cell_bias_ICFO, cell_hidden_size);

  // [1, 3, 3]
  std::vector<float> Y_T_data{
      0.0978363082f, 0.105625421f, 0.116753615f,
      0.236107856f, 0.195716992f, -0.133973882f,
      0.0f, 0.0f, 0.0f};

  // convert to onnx output semantic, should be same in this case.
  std::vector<float> Y_data = ConvertBatchSeqToSeqBatch(
      Y_T_data, batch_size, input_max_step, cell_hidden_size);

  const std::vector<float> Y_h_data{};
  const std::vector<float> Y_c_data{};

  std::vector<int> shortenSeqLen{2};

  RunAttnLstmTest(
      X_data, W_data, R_data, Y_data, Y_h_data, Y_c_data,
      s_memory_layer_weight, s_query_layer_weight, s_attn_v, s_M_data, &s_mem_seq_lenghts, &s_attn_layer_weight,
      input_only_depth, batch_size, cell_hidden_size, input_max_step,
      memory_max_step, memory_depth, am_attn_size, aw_attn_size,
      &B_data, nullptr, nullptr, nullptr, &shortenSeqLen,
      "forward", -9999.f, true, false);
}

TEST(AttnLSTMTest, ReverseLstmWithBahdanauAMShortenSeqLength) {
  std::vector<float> X_data = ConvertBatchSeqToSeqBatch(s_X_T_data, batch_size, input_max_step, input_only_depth);

  std::vector<float> WR_T_data = ConvertIcfoToIofc(s_WR_T_data_ICFO, cell_hidden_size);

  const size_t W_data_size = 5 * 12;
  std::vector<float> W_T_data(&(WR_T_data[0]), &(WR_T_data[0]) + W_data_size);
  std::vector<float> R_T_data(&(WR_T_data[0]) + W_data_size, &(WR_T_data[0]) + WR_T_data.size());

  // transpose W and R for onnx sematic
  std::vector<float> W_data = Transpose2D(W_T_data, input_size, 4 * cell_hidden_size);
  std::vector<float> R_data = Transpose2D(R_T_data, cell_hidden_size, 4 * cell_hidden_size);

  std::vector<float> B_data = ConvertIcfoToIofc(s_lstm_cell_bias_ICFO, cell_hidden_size);

  // [1, 3, 3]
  std::vector<float> Y_T_data{
      -0.229537353f, 0.136488736f, -0.414591223f,
      0.127119571f, 0.164731115f, -0.1136849f,
      0.0f, 0.0f, 0.0f};

  // convert to onnx output semantic, should be same in this case.
  std::vector<float> Y_data = ConvertBatchSeqToSeqBatch(
      Y_T_data, batch_size, input_max_step, cell_hidden_size);

  const std::vector<float> Y_h_data{};
  const std::vector<float> Y_c_data{};

  std::vector<int> shortenSeqLen{2};

  RunAttnLstmTest(
      X_data, W_data, R_data, Y_data, Y_h_data, Y_c_data,
      s_memory_layer_weight, s_query_layer_weight, s_attn_v, s_M_data, &s_mem_seq_lenghts, &s_attn_layer_weight,
      input_only_depth, batch_size, cell_hidden_size, input_max_step,
      memory_max_step, memory_depth, am_attn_size, aw_attn_size,
      &B_data, nullptr, nullptr, nullptr, &shortenSeqLen,
      "reverse", -9999.f, true, false);
}

TEST(AttnLSTMTest, BidirectionLstmWithBahdanauAMShortenSeqLength) {
  std::vector<float> X_data = ConvertBatchSeqToSeqBatch(s_X_T_data, batch_size, input_max_step, input_only_depth);

  std::vector<float> WR_T_data = ConvertIcfoToIofc(s_WR_T_data_ICFO, cell_hidden_size);

  const size_t W_data_size = 5 * 12;
  std::vector<float> W_T_data(&(WR_T_data[0]), &(WR_T_data[0]) + W_data_size);
  std::vector<float> R_T_data(&(WR_T_data[0]) + W_data_size, &(WR_T_data[0]) + WR_T_data.size());

  // transpose W and R for onnx sematic
  std::vector<float> W_data = Transpose2D(W_T_data, input_size, 4 * cell_hidden_size);
  std::vector<float> R_data = Transpose2D(R_T_data, cell_hidden_size, 4 * cell_hidden_size);

  std::vector<float> B_data = ConvertIcfoToIofc(s_lstm_cell_bias_ICFO, cell_hidden_size);

  // concat two result sequence from tf
  std::vector<float> Y_data = ConcatLastDim(
      ConvertBatchSeqToSeqBatch(
          std::vector<float>{
              0.0978363082f, 0.105625421f, 0.116753615f,
              0.236107856f, 0.195716992f, -0.133973882f,
              0.0f, 0.0f, 0.0f},
          batch_size, input_max_step, cell_hidden_size),
      ConvertBatchSeqToSeqBatch(
          std::vector<float>{
              -0.229537353f, 0.136488736f, -0.414591223f,
              0.127119571f, 0.164731115f, -0.1136849f,
              0.0f, 0.0f, 0.0f},
          batch_size, input_max_step, cell_hidden_size),
      cell_hidden_size * batch_size);

  const std::vector<float> Y_h_data{};
  const std::vector<float> Y_c_data{};

  std::vector<int> shortenSeqLen{2};

  auto d_W_data = ConcatDup(W_data);
  auto d_R_data = ConcatDup(R_data);
  auto d_B_data = ConcatDup(B_data);

  auto d_memory_layer_weight = ConcatDup(s_memory_layer_weight);
  auto d_query_layer_weight = ConcatDup(s_query_layer_weight);
  auto d_attn_v = ConcatDup(s_attn_v);
  auto d_attn_layer_weight = ConcatDup(s_attn_layer_weight);

  RunAttnLstmTest(
      X_data, d_W_data, d_R_data, Y_data, Y_h_data, Y_c_data,
      d_memory_layer_weight, d_query_layer_weight, d_attn_v, s_M_data, &s_mem_seq_lenghts, &d_attn_layer_weight,
      input_only_depth, batch_size, cell_hidden_size, input_max_step,
      memory_max_step, memory_depth, am_attn_size, aw_attn_size,
      &d_B_data, nullptr, nullptr, nullptr, &shortenSeqLen,
      "bidirectional", -9999.f, true, false);
}

TEST(AttnLSTMTest, BidirectionLstmWithBahdanauAM2BatchShortenSeqLen) {
  const int batch2Size = 2;
  const int inputMaxStep4 = 4;

  static const std::vector<float> s_X_T_2batch{0.25f, -1.5f, 1.0f, 0.25f, -0.5f, -1.5f, 0.1f, 1.5f, 0.25f, 0.0f, 0.0f, 0.0f,
                                               0.1f, -0.125f, 0.25f, -0.5f, 0.25f, 0.1f, 1.0f, 0.5f, -1.5f, 0.0f, 0.0f, 0.0f};
  static const std::vector<int> s_seq_lengths_2batch{3, 2};

  std::vector<float> X_data = ConvertBatchSeqToSeqBatch(s_X_T_2batch, batch2Size, inputMaxStep4, input_only_depth);

  std::vector<float> WR_T_data = ConvertIcfoToIofc(s_WR_T_data_ICFO, cell_hidden_size);

  const size_t W_data_size = 5 * 12;
  std::vector<float> W_T_data(&(WR_T_data[0]), &(WR_T_data[0]) + W_data_size);
  std::vector<float> R_T_data(&(WR_T_data[0]) + W_data_size, &(WR_T_data[0]) + WR_T_data.size());

  // transpose W and R for onnx sematic
  std::vector<float> W_data = Transpose2D(W_T_data, input_size, 4 * cell_hidden_size);
  std::vector<float> R_data = Transpose2D(R_T_data, cell_hidden_size, 4 * cell_hidden_size);

  std::vector<float> B_data = ConvertIcfoToIofc(s_lstm_cell_bias_ICFO, cell_hidden_size);

  // concat two result sequence from tf
  std::vector<float> Y_data = ConcatLastDim(
      ConvertBatchSeqToSeqBatch(
          std::vector<float>{
              0.0978363082f, 0.105625421f, 0.116753615f, 0.236107856f, 0.195716992f, -0.133973882f, -0.029754376f, 0.274325848f, -0.387993187f, 0.0f, 0.0f, 0.0f,
              0.261070877f, 0.144692719f, -0.274273455f, -0.272313654f, 0.324584424f, -0.298215479f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
          batch2Size, inputMaxStep4, cell_hidden_size),
      ConvertBatchSeqToSeqBatch(
          std::vector<float>{
              -0.248873845f, 0.139064044f, -0.596312642f, 0.134674609f, 0.255465984f, -0.119320348f, 0.10030812f, 0.110956885f, -0.438956916f, 0.0f, 0.0f, 0.0f,
              -0.230028018f, 0.230880141f, -0.193421111f, 0.328211069f, 0.230195627f, -0.3777861f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
          batch2Size, inputMaxStep4, cell_hidden_size),
      cell_hidden_size * batch2Size);

  const std::vector<float> Y_h_data{};
  const std::vector<float> Y_c_data{};

  auto d_W_data = ConcatDup(W_data);
  auto d_R_data = ConcatDup(R_data);
  auto d_B_data = ConcatDup(B_data);

  auto d_memory_layer_weight = ConcatDup(s_memory_layer_weight);
  auto d_query_layer_weight = ConcatDup(s_query_layer_weight);
  auto d_attn_v = ConcatDup(s_attn_v);
  auto d_attn_layer_weight = ConcatDup(s_attn_layer_weight);

  RunAttnLstmTest(
      X_data, d_W_data, d_R_data, Y_data, Y_h_data, Y_c_data,
      d_memory_layer_weight, d_query_layer_weight, d_attn_v, s_M_2batch, &s_mem_seq_lenghts_2batch, &d_attn_layer_weight,
      input_only_depth, batch2Size, cell_hidden_size, inputMaxStep4,
      memory_max_step, memory_depth, am_attn_size, aw_attn_size,
      &d_B_data, nullptr, nullptr, nullptr, &s_seq_lengths_2batch,
      "bidirectional", -9999.f, true, false);
}

}  // namespace test
}  // namespace onnxruntime

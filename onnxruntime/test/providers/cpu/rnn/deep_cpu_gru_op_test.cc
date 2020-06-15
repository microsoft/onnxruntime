// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <iterator>
#include <vector>

#include "core/providers/cpu/rnn/deep_cpu_gru.h"
#include "test/providers/provider_test_utils.h"
using namespace std;
namespace onnxruntime {
namespace test {

static const std::vector<string> default_activations = {"sigmoid", "tanh"};

static void RunGruTest(const std::vector<float>& X_data,
                       const std::vector<float>& W_data,
                       const std::vector<float>& R_data,
                       const std::vector<float>& Y_data,
                       const std::vector<float>& Y_h_data,
                       int64_t input_size,
                       int batch_size,
                       int64_t hidden_size,
                       int64_t seq_length,
                       const std::vector<float>* B_data = nullptr,
                       const std::vector<float>* initial_h_data = nullptr,
                       const std::vector<int>* sequence_lengths = nullptr,
                       const std::string& direction = "forward",
                       float clip = 9999.0,
                       bool output_sequence = true,
                       bool linear_before_reset = false,
                       // copy the following vectors as we may modify them
                       std::vector<string> activations = default_activations,
                       std::vector<float> activation_alphas = {},
                       std::vector<float> activation_betas = {}) {
  OpTester test("GRU");

  test.AddShapeToTensorData();

  int num_directions = (direction == "bidirectional") ? 2 : 1;

  if (num_directions == 2 && activations.size() == 2) {
    activations.reserve(4);  // need to avoid reallocation when inserting
    // default to copying the activations so the same are used for forward and backwards
    std::copy(activations.cbegin(), activations.cend(), std::back_inserter(activations));
  }

  test.AddAttribute<std::vector<string>>("activations", activations);
  if (!activation_alphas.empty())
    test.AddAttribute<std::vector<float>>("activation_alpha", activation_alphas);
  if (!activation_betas.empty())
    test.AddAttribute<std::vector<float>>("activation_beta", activation_betas);

  test.AddAttribute("direction", direction);
  test.AddAttribute("hidden_size", hidden_size);
  // test.AddAttribute<int64_t>("output_sequence", output_sequence);
  test.AddAttribute<int64_t>("linear_before_reset", linear_before_reset);
  // if clip is a very big number (usually it is default value), don't set the clip
  if (clip < 999.f)
    test.AddAttribute<float>("clip", clip);

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<int64_t> W_dims = {num_directions, 3 * hidden_size, input_size};
  std::vector<int64_t> R_dims = {num_directions, 3 * hidden_size, hidden_size};

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("W", W_dims, W_data, true);
  test.AddInput<float>("R", R_dims, R_data, true);

  if (B_data) {
    std::vector<int64_t> B_dims = {num_directions, 6 * hidden_size};
    test.AddInput<float>("B", B_dims, *B_data, true);
  }

  if (sequence_lengths) {
    std::vector<int64_t> sequence_lens_dims{batch_size};
    test.AddInput<int>("sequence_lens", sequence_lens_dims, *sequence_lengths);
  }

  if (initial_h_data) {
    std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
    test.AddInput<float>("initial_h", initial_h_dims, *initial_h_data);
  }

  if (output_sequence != 0) {
    std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y", Y_dims, Y_data);
  } else {
    test.AddMissingOptionalOutput<float>();
  }

  if (!Y_h_data.empty()) {
    std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);
  } else {
    test.AddMissingOptionalOutput<float>();
  }

  // TensorRT failed on GRU tests
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

void DefaultActivationsSimpleWeightsNoBias(std::string direction,
                                           const std::vector<float>& Y_data,
                                           const std::vector<float>& Y_h_data,
                                           bool linear_before_reset = false) {
  int64_t seq_length = 2;
  int batch_size = linear_before_reset ? 3 : 2;  // extra row to validate usage of linear_output_
  int64_t input_size = 1;
  int64_t hidden_size = 3;

  int num_directions = direction == "bidirectional" ? 2 : 1;

  std::vector<float> X_data;

  if (linear_before_reset) {
    X_data = {1.f, 2.f, 3.f,
              10.f, 11.f, 12.f};
  } else {
    X_data = {1.f, 2.f,
              10.f, 11.f};
  }

  std::vector<float> W_data{0.1f, 0.2f, 0.3f,   // wz
                            1.f, 2.f, 3.f,      // wr
                            10.f, 11.f, 12.f};  // wh

  // duplicate for bidirectional
  if (num_directions == 2) {
    W_data.reserve(W_data.size() * 2);  // need to avoid reallocation when inserting
    std::copy(W_data.cbegin(), W_data.cend(), std::back_inserter(W_data));
  }

  std::vector<float> R_data(num_directions * 3 * hidden_size * hidden_size, 0.1f);

  RunGruTest(X_data, W_data, R_data, Y_data, Y_h_data, input_size, batch_size, hidden_size, seq_length,
             nullptr, nullptr, nullptr, direction, 9999.0, true, linear_before_reset);

  // if Y_h_data is empty that tests Y_h not being returned. we need to have at least one output or
  // the node will get removed, so only test with output_sequence == false (no Y as output) if Y_h is not optional
  if (!Y_h_data.empty())
    RunGruTest(X_data, W_data, R_data, Y_data, Y_h_data, input_size, batch_size, hidden_size, seq_length,
               nullptr, nullptr, nullptr, direction, 9999.0, /* output_sequence*/ false, linear_before_reset);
}

TEST(GRUTest, ForwardDefaultActivationsSimpleWeightsNoBiasTwoRows) {
  std::vector<float> Y_data{
      0.4750208f, 0.450166f, 0.4255575f,
      0.45016602f, 0.40131235f, 0.35434368f,

      0.6027093f, 0.5083023f, 0.44950223f,
      0.5754369f, 0.45485455f, 0.3747841f};

  std::vector<float> Y_h_data{
      0.6027093f, 0.5083023f, 0.44950223f,
      0.5754369f, 0.45485455f, 0.3747841f};

  DefaultActivationsSimpleWeightsNoBias("forward", Y_data, Y_h_data);

  // test Y_h not being returned
  DefaultActivationsSimpleWeightsNoBias("forward", Y_data, {});
}

TEST(GRUTest, ReverseDefaultActivationsSimpleWeightsNoBiasTwoRows) {
  std::vector<float> Y_data{
      0.6082785f, 0.50623393f, 0.4426924f,
      0.5803454f, 0.4527356f, 0.36886263f,

      0.26894143f, 0.11920292f, 0.04742587f,
      0.24973989f, 0.09975048f, 0.03557118f};

  std::vector<float> Y_h_data{
      0.6082785f, 0.50623393f, 0.4426924f,
      0.5803454f, 0.4527356f, 0.36886263f};

  DefaultActivationsSimpleWeightsNoBias("reverse", Y_data, Y_h_data);
}

TEST(GRUTest, BidirectionalDefaultActivationsSimpleWeightsNoBias) {
  std::vector<float> Y_data{
      // forward output for input sequence 0
      0.4750208f, 0.450166f, 0.4255575f,
      0.45016602f, 0.40131235f, 0.35434368f,

      // reverse output for input sequence 0 [sequence 1 in reversed input]
      0.6082785f, 0.50623393f, 0.4426924f,
      0.5803454f, 0.4527356f, 0.36886263f,

      // forward output for input sequence 1
      0.6027093f, 0.5083023f, 0.44950223f,
      0.5754369f, 0.45485455f, 0.3747841f,

      // reverse output for input sequence 1 [sequence 0 in reversed input]
      0.26894143f, 0.11920292f, 0.04742587f,
      0.24973989f, 0.09975048f, 0.03557118f};

  std::vector<float> Y_h_data{
      // we did the forward processing of input[1] last
      0.6027093f, 0.5083023f, 0.44950223f,
      0.5754369f, 0.45485455f, 0.3747841f,

      // and the reverse processing of input[0] last as the input order was reversed
      0.6082785f, 0.50623393f, 0.4426924f,
      0.5803454f, 0.4527356f, 0.36886263f};

  DefaultActivationsSimpleWeightsNoBias("bidirectional", Y_data, Y_h_data);
}

TEST(GRUTest, BidirectionalDefaultActivationsSimpleWeightsNoBiasLinearBeforeReset) {
  std::vector<float> Y_data{
      // forward output for input sequence 0
      0.4750208f, 0.450166f, 0.4255575f,
      0.45016602f, 0.40131235f, 0.35434368f,
      0.42555748f, 0.35434369f, 0.28905049f,

      // reverse output for input sequence 0 [sequence 1 in reversed input]
      0.6082785f, 0.50623393f, 0.4426924f,
      0.5803454f, 0.4527356f, 0.36886263f,
      0.5521325f, 0.40092295f, 0.30118297f,

      // forward output for input sequence 1
      0.6027093f, 0.5083023f, 0.44950223f,
      0.5754369f, 0.45485455f, 0.3747841f,
      0.54791767f, 0.40301081f, 0.30608854f,

      // reverse output for input sequence 1 [sequence 0 in reversed input]
      0.26894143f, 0.11920292f, 0.04742587f,
      0.24973989f, 0.09975048f, 0.03557118f,
      0.23147521f, 0.08317269f, 0.02659699f};

  std::vector<float> Y_h_data{
      // we did the forward processing of input[1] last
      0.6027093f, 0.5083023f, 0.44950223f,
      0.5754369f, 0.45485455f, 0.3747841f,
      0.54791767f, 0.40301081f, 0.30608854f,

      // and the reverse processing of input[0] last as the input order was reversed
      0.6082785f, 0.50623393f, 0.4426924f,
      0.5803454f, 0.4527356f, 0.36886263f,
      0.5521325f, 0.40092295f, 0.30118297f};

  DefaultActivationsSimpleWeightsNoBias("bidirectional", Y_data, Y_h_data, true);
}

void DefaultActivationsSimpleWeightsWithBias(std::string direction,
                                             const std::vector<float>& Y_data,
                                             bool linear_before_reset = false,
                                             bool one_row = false) {
  int64_t seq_length = 2;
  int batch_size = one_row ? 1 : 2;  // if 2 take batch_parallel_ path. if 1, don't.
  int64_t input_size = 1;
  int64_t hidden_size = 3;

  int num_directions = direction == "bidirectional" ? 2 : 1;

  std::vector<float> X_data;

  if (batch_size == 2)
    X_data = {-0.1f, 0.2f, -0.3f, 0.4f};
  else
    X_data = {-0.1f, -0.3f};

  std::vector<float> W_data{0.1f, 0.2f, 0.3f,   // wz
                            0.2f, 0.3f, 0.1f,   // wr
                            0.3f, 0.1f, 0.2f};  // wh

  std::vector<float> B_data{
      -0.01f, 0.1f, 0.01f,  // Wb[zrh]
      -0.2f, -0.02f, 0.02f,
      0.3f, -0.3f, -0.3f,

      -0.03f, 0.5f, -0.7f,  // Rb[zrh]
      0.05f, -0.7f, 0.3f,
      0.07f, -0.03f, 0.5f};

  // duplicate for bidirectional
  auto duplicate_data = [](std::vector<float>& data) {
    data.reserve(data.size() * 2);  // need to avoid reallocation when inserting
    std::copy(data.cbegin(), data.cend(), std::back_inserter(data));
  };

  if (num_directions == 2) {
    duplicate_data(W_data);
    duplicate_data(B_data);
  }

  std::vector<float> R_data(num_directions * 3 * hidden_size * hidden_size, 0.1f);

  RunGruTest(X_data, W_data, R_data, Y_data, {}, input_size, batch_size, hidden_size, seq_length,
             &B_data, nullptr, nullptr, direction, 999.f, /* output_sequence*/ true, linear_before_reset);
}

TEST(GRUTest, ForwardDefaultActivationsSimpleWeightsWithBiasBatchParallel) {
  std::vector<float> Y_data{
      0.16783132f, -0.11754231f, 0.11977843f,
      0.2046872f, -0.10372487f, 0.15365849f,

      0.22688604f, -0.19698407f, 0.14017843f,
      0.33386092f, -0.15799662f, 0.2381169f};

  DefaultActivationsSimpleWeightsWithBias("forward", Y_data);
}

TEST(GRUTest, ForwardDefaultActivationsSimpleWeightsWithBiasBatchParallelLinearBeforeReset) {
  std::vector<float> Y_data{
      0.15024948f, -0.11097029f, -0.02121867f,
      0.18887489f, -0.09747667f, 0.02093463f,

      0.19538902f, -0.19016478f, -0.05644283f,
      0.30856851f, -0.15190377f, 0.05999807f};

  const bool linear_before_reset = true;
  DefaultActivationsSimpleWeightsWithBias("forward", Y_data, linear_before_reset);
}

TEST(GRUTest, ReverseDefaultActivationsSimpleWeightsWithBiasBatchParallelLinearBeforeReset) {
  std::vector<float> Y_data{
      0.20910699f, -0.18880953f, -0.04005555f,
      0.29700265f, -0.15308119f, 0.04537245f,

      0.12252139f, -0.12032216f, -0.05064924f,
      0.21249877f, -0.08884402f, 0.04751285f};

  const bool linear_before_reset = true;
  DefaultActivationsSimpleWeightsWithBias("reverse", Y_data, linear_before_reset);
}

// test forward !batch_parallel_ path with linear_before_reset
TEST(GRUTest, ForwardDefaultActivationsSimpleWeightsWithBiasLinearBeforeReset) {
  std::vector<float> Y_data{
      0.15024948f, -0.11097029f, -0.02121867f,
      0.19538902f, -0.19016478f, -0.05644283f};

  const bool linear_before_reset = true;
  const bool one_row = true;
  DefaultActivationsSimpleWeightsWithBias("forward", Y_data, linear_before_reset, one_row);
}

// test reverse !batch_parallel_ path with linear_before_reset
TEST(GRUTest, ReverseDefaultActivationsSimpleWeightsWithBiasLinearBeforeReset) {
  std::vector<float> Y_data{
      0.20910699f, -0.18880953f, -0.04005555f,
      0.12252139f, -0.12032216f, -0.05064924f};

  const bool linear_before_reset = true;
  const bool one_row = true;
  DefaultActivationsSimpleWeightsWithBias("reverse", Y_data, linear_before_reset, one_row);
}

/*******************
* Legacy tests from LotusRT
*/
class DeepCpuGruOpTestContext {
 public:
  DeepCpuGruOpTestContext(std::string direction,
                          const std::vector<std::string>& activations,
                          const bool use_bias = true,
                          const std::vector<float>& alpha = {},
                          const std::vector<float>& beta = {},
                          bool large_hidden = false,
                          int input_size = 2);

  ~DeepCpuGruOpTestContext() = default;

  void RunTest(const std::vector<float>& X,
               const int batch,
               const int seq_length,
               const std::vector<int>& sequence_length,
               const std::vector<float>* initial_h,
               const std::vector<float>& expected_Y,
               const std::vector<float>& expected_Y_h,
               const bool linear_before_reset = false);

 private:
  const int input_size_;
  const int hidden_dim_;
  const bool use_bias_;
  const std::string direction_;
  int num_directions_;
  const std::vector<std::string> activation_func_names_;
  const std::vector<float> alphas_;
  const std::vector<float> betas_;
  std::vector<float> gru_input_weights_;
  std::vector<float> gru_recurrent_weights_;
  std::vector<float> gru_bias_;
};

DeepCpuGruOpTestContext::DeepCpuGruOpTestContext(const std::string direction,
                                                 const std::vector<std::string>& activations,
                                                 const bool use_bias,
                                                 const std::vector<float>& alpha,
                                                 const std::vector<float>& beta,
                                                 bool large_hidden,
                                                 int input_size)
    : input_size_(input_size),
      hidden_dim_(large_hidden ? 32 : 2),
      use_bias_(use_bias),
      direction_(direction),
      activation_func_names_(activations),
      alphas_(alpha),
      betas_(beta) {
  if (direction == "bidirectional")
    num_directions_ = 2;
  else
    num_directions_ = 1;

  if (large_hidden) {
    const int input_weight_block_size = input_size_ * hidden_dim_;
    const int hidden_weight_block_size = hidden_dim_ * hidden_dim_;
    std::vector<float> gru_input_weights(3 * input_weight_block_size);
    std::vector<float> gru_hidden_weights(3 * hidden_weight_block_size);
    std::vector<float> gru_bias(6 * hidden_dim_);

    // Construction of input weights.
    for (int i = 0; i < input_weight_block_size; i++) {
      gru_input_weights[i] = 0.1f;
      gru_input_weights[i + input_weight_block_size] = 0.2f;
      gru_input_weights[i + 2 * input_weight_block_size] = 0.3f;
    }

    for (int i = 0; i < hidden_dim_; i++) {
      int diag_index = i * hidden_dim_ + i;
      gru_hidden_weights[diag_index] = 0.1f + 0.1f * (i % 2);
      gru_hidden_weights[diag_index + hidden_weight_block_size] = 0.1f + 0.1f * (i % 3);
      gru_hidden_weights[diag_index + 2 * hidden_weight_block_size] = 0.1f + 0.1f * (i % 5);
    }

    for (int i = 0; i < hidden_dim_; i++) {
      gru_bias[i] = 0.1f;
      gru_bias[i + hidden_dim_] = 0.2f;
      gru_bias[i + 2 * hidden_dim_] = 0.3f;
    }

    for (int i = 0; i < 3 * hidden_dim_; i++) {
      gru_bias[i + 3 * hidden_dim_] = 0.0f;
    }

    for (int i = 0; i < num_directions_; i++) {
      gru_input_weights_.insert(gru_input_weights_.end(), gru_input_weights.begin(), gru_input_weights.end());
      gru_recurrent_weights_.insert(gru_recurrent_weights_.end(), gru_hidden_weights.begin(), gru_hidden_weights.end());
      gru_bias_.insert(gru_bias_.end(), gru_bias.begin(), gru_bias.end());
    }

  } else {
    if (num_directions_ == 2) {
      // kBidirectional weights.
      gru_input_weights_ = {
          -0.494659f, 0.0453352f, -0.487793f, 0.417264f,    // Wz
          -0.0091708f, -0.255364f, -0.106952f, -0.266717f,  // Wr
          -0.0888852f, -0.428709f, -0.283349f, 0.208792f,   // Wh
          -0.494659f, 0.0453352f, -0.487793f, 0.417264f,    // WBz
          -0.0091708f, -0.255364f, -0.106952f, -0.266717f,  // WBr
          -0.0888852f, -0.428709f, -0.283349f, 0.208792f};  // WBh

      gru_recurrent_weights_ = {
          0.146626f, -0.0620289f, -0.0815302f, 0.100482f,  // Rz
          -0.228172f, 0.405972f, 0.31576f, 0.281487f,      // Rr
          -0.394864f, 0.42111f, -0.386624f, -0.390225f,    // Rh
          0.146626f, -0.0620289f, -0.0815302f, 0.100482f,  // RBz
          -0.228172f, 0.405972f, 0.31576f, 0.281487f,      // RBr
          -0.394864f, 0.42111f, -0.386624f, -0.390225f};   // RBh

      gru_bias_ = {
          0.381619f, 0.0323954f,   // Wbz
          -0.258721f, 0.45056f,    // Wbr
          -0.250755f, 0.0967895f,  // Wbh
          0.0f, 0.0f,              // Rbz
          -0.0f, 0.0f,             // Rbr
          -0.0f, 0.0f,             // Rbh
          0.381619f, 0.0323954f,   // WBbz
          -0.258721f, 0.45056f,    // WBbr
          -0.250755f, 0.0967895f,  // WBbh
          0.0f, 0.0f,              // RBbz
          -0.0f, 0.0f,             // RBbr
          -0.0f, 0.0f};            // RBbh
    } else {
      // Unidirectional weights.
      gru_input_weights_ = {
          -0.494659f, 0.0453352f, -0.487793f, 0.417264f,    // Wz
          -0.0091708f, -0.255364f, -0.106952f, -0.266717f,  // Wr
          -0.0888852f, -0.428709f, -0.283349f, 0.208792f    // Wh
      };

      gru_recurrent_weights_ = {
          0.146626f, -0.0620289f, -0.0815302f, 0.100482f,  // Rz
          -0.228172f, 0.405972f, 0.31576f, 0.281487f,      // Rr
          -0.394864f, 0.42111f, -0.386624f, -0.390225f};   // Rh

      gru_bias_ = {
          0.381619f, 0.0323954f,   // Wbz
          -0.258721f, 0.45056f,    // Wbr
          -0.250755f, 0.0967895f,  // Wbh
          0.0f, 0.0f,              // Rbz
          -0.0f, 0.0f,             // Rbr
          -0.0f, 0.0f              // Rbh
      };
    }
  }
}

void DeepCpuGruOpTestContext::RunTest(const std::vector<float>& X,
                                      const int batch_size,
                                      const int seq_length,
                                      const std::vector<int>& sequence_lens,
                                      const std::vector<float>* initial_h,
                                      const std::vector<float>& expected_Y,
                                      const std::vector<float>& expected_Y_h,
                                      const bool linear_before_reset) {
  //run with and without output_sequence
  RunGruTest(X, gru_input_weights_, gru_recurrent_weights_,
             expected_Y, expected_Y_h,
             input_size_, batch_size, hidden_dim_, seq_length,
             use_bias_ ? &gru_bias_ : nullptr,
             initial_h,
             &sequence_lens,
             direction_,
             9999999999.f,
             /*output_sequence*/ true,
             linear_before_reset,
             activation_func_names_,
             alphas_,
             betas_);

  RunGruTest(X, gru_input_weights_, gru_recurrent_weights_,
             expected_Y, expected_Y_h,
             input_size_, batch_size, hidden_dim_, seq_length,
             use_bias_ ? &gru_bias_ : nullptr,
             initial_h,
             &sequence_lens,
             direction_,
             9999999999.f,
             /*output_sequence*/ false,
             linear_before_reset,
             activation_func_names_,
             alphas_,
             betas_);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpForwardBasic) {
  const std::string direction = "forward";
  const std::vector<std::string> activations = {"sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch = 1;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          -0.185934f, -0.269585f};
  std::vector<int> sequence_length = {2};
  std::vector<float> initial_h = {0.0f, 0.0f};
  std::vector<float> expected_Y = {-0.03255286f, 0.0774838f, -0.05556786f, 0.0785508f};
  std::vector<float> expected_Y_h = {-0.05556786f, 0.0785508f};

  ctx.RunTest(X, batch, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpBackwardBasic) {
  const std::string direction = "reverse";
  const std::vector<std::string> activations = {"sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 1;
  const int seq_length = 2;
  std::vector<float> X = {-0.185934f, -0.269585f,
                          -0.455351f, -0.276391f};
  std::vector<int> sequence_length = {2};
  std::vector<float> initial_h = {0.0f, 0.0f};
  std::vector<float> expected_Y = {-0.05556786f, 0.0785508f,
                                   -0.03255286f, 0.0774838f};
  std::vector<float> expected_Y_h = {-0.05556786f, 0.0785508f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpBidirectionalBasic) {
  const std::string direction = "bidirectional";
  const std::vector<std::string> activations = {"sigmoid", "tanh", "sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 1;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          -0.185934f, -0.269585f};
  std::vector<int> sequence_length = {2};
  std::vector<float> initial_h = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> expected_Y = {-0.03255286f, 0.0774838f,
                                   -0.05469977f, 0.1004222f,

                                   -0.05556786f, 0.0785508f,
                                   -0.04566499f, 0.04621252f};
  std::vector<float> expected_Y_h = {-0.05556786f, 0.0785508f,
                                     -0.05469977f, 0.1004222f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpForwardActivation) {
  const std::string direction = "forward";
  const std::vector<std::string> activations = {"tanh", "sigmoid"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 1;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          -0.185934f, -0.269585f};
  std::vector<int> sequence_length = {2};
  std::vector<float> initial_h = {0.0f, 0.0f};
  std::vector<float> expected_Y = {0.222789f, 0.4669829f,
                                   0.3810334f, 0.4944591f};
  std::vector<float> expected_Y_h = {0.3810334f, 0.4944591f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpForwardInitialHiddenState) {
  const std::string direction = "forward";
  const std::vector<std::string> activations = {"sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 1;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          -0.185934f, -0.269585f};
  std::vector<int> sequence_length = {2};
  std::vector<float> initial_h = {0.5f, -0.5f};
  std::vector<float> expected_Y = {0.2366661f, -0.1500429f,
                                   0.07378622f, -0.02782359f};
  std::vector<float> expected_Y_h = {0.07378622f, -0.02782359f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpForwardBatch) {
  const std::string direction = "forward";
  const std::vector<std::string> activations = {"sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 2;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          -0.455351f, -0.276391f,

                          -0.185934f, -0.269585f,
                          -0.185934f, -0.269585f};
  std::vector<int> sequence_length = {2, 2};
  std::vector<float> initial_h = {0.5f, -0.5f, 0.0f, 0.0f};
  std::vector<float> expected_Y = {0.2366661f, -0.1500429f,
                                   -0.03255286f, 0.0774838f,

                                   0.07378622f, -0.02782359f,
                                   -0.05556786f, 0.0785508f};

  std::vector<float> expected_Y_h = {0.07378622f, -0.02782359f,
                                     -0.05556786f, 0.0785508f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpForwardBatchLinearBeforeReset) {
  const std::string direction = "forward";
  const std::vector<std::string> activations = {"sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 2;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          -0.455351f, -0.276391f,

                          -0.185934f, -0.269585f,
                          -0.185934f, -0.269585f};
  std::vector<int> sequence_length = {2, 2};
  std::vector<float> initial_h = {0.5f, -0.5f, 0.0f, 0.0f};
  std::vector<float> expected_Y = {0.253942400f, -0.174207777f,
                                   -0.0325528607f, 0.0774837881f,

                                   0.0874997079f, -0.0485242009f,
                                   -0.0577347837f, 0.0796165839f};

  std::vector<float> expected_Y_h = {0.0874997079f, -0.0485242009f,
                                     -0.0577347837f, 0.0796165839f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h, true);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpGrowBatchSequenceLength) {
  const std::string direction = "forward";
  const std::vector<std::string> activations = {"sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 1;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          -0.185934f, -0.269585f};
  std::vector<int> sequence_length = {2};
  std::vector<float> initial_h = {0.0f, 0.0f};
  std::vector<float> expected_Y = {-0.03255286f, 0.0774838f,
                                   -0.05556786f, 0.0785508f};
  std::vector<float> expected_Y_h = {-0.05556786f, 0.0785508f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);

  const int batch2 = 2;
  const int seq_length2 = 2;
  std::vector<float> X2 = {-0.455351f, -0.276391f,
                           -0.455351f, -0.276391f,

                           -0.185934f, -0.269585f,
                           0.0f, 0.0f};
  std::vector<int> sequence_length2 = {2, 1};
  std::vector<float> initial_h2 = {0.5f, -0.5f,
                                   0.0f, 0.0f};
  std::vector<float> expected_Y2 = {0.2366661f, -0.1500429f,
                                    -0.03255286f, 0.0774838f,

                                    0.07378622f, -0.02782359f,
                                    0.0f, 0.0f};

  std::vector<float> expected_Y_h2 = {0.07378622f, -0.02782359f,
                                      -0.03255286f, 0.0774838f};

  ctx.RunTest(X2, batch2, seq_length2, sequence_length2, &initial_h2, expected_Y2, expected_Y_h2);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpGrowBatchSequenceLengthLinearBeforeReset) {
  const std::string direction = "forward";
  const std::vector<std::string> activations = {"sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 1;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          -0.185934f, -0.269585f};
  std::vector<int> sequence_length = {2};
  std::vector<float> initial_h = {0.0f, 0.0f};
  std::vector<float> expected_Y = {-0.0325528607f, 0.0774837881f,
                                   -0.0577347837f, 0.0796165839f};
  std::vector<float> expected_Y_h = {-0.0577347837f, 0.0796165839f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h, true);

  const int batch2 = 2;
  const int seq_length2 = 2;
  std::vector<float> X2 = {-0.455351f, -0.276391f,
                           -0.455351f, -0.276391f,

                           -0.185934f, -0.269585f,
                           0.0f, 0.0f};
  std::vector<int> sequence_length2 = {2, 1};
  std::vector<float> initial_h2 = {0.5f, -0.5f,
                                   0.0f, 0.0f};
  std::vector<float> expected_Y2 = {0.253942400f, -0.174207777f,
                                    -0.0325528607f, 0.0774837881f,

                                    0.0874997079f, -0.0485242009f,
                                    0.0f, 0.0f};

  std::vector<float> expected_Y_h2 = {0.0874997079f, -0.0485242009f,
                                      -0.0325528607f, 0.0774837881f};

  ctx.RunTest(X2, batch2, seq_length2, sequence_length2, &initial_h2, expected_Y2, expected_Y_h2, true);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpSequenceLengthWithBidirectionalLinearBeforeResetB1) {
  const std::string direction = "bidirectional";
  const std::vector<std::string> activations = {"sigmoid", "tanh", "sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 1;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          -0.185934f, -0.269585f};
  std::vector<int> sequence_length = {2};
  std::vector<float> initial_h = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> expected_Y = {-0.0325528607f, 0.0774837881f, -0.0559310019f, 0.101836264f,
                                   -0.0577347837f, 0.0796165839f, -0.0456649922f, 0.0462125242f};

  std::vector<float> expected_Y_h = {-0.0577347837f, 0.0796165839f,
                                     -0.0559310019f, 0.101836264f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h, true);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpSequenceLengthWithBidirectionalLinearBeforeResetB2) {
  const std::string direction = "bidirectional";
  const std::vector<std::string> activations = {"sigmoid", "tanh", "sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 1;
  const int seq_length = 2;
  std::vector<float> X = {0.855351f, 0.676391f,
                          0.585934f, 0.669585f};
  std::vector<int> sequence_length = {2};
  std::vector<float> initial_h = {0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> expected_Y = {-0.275918573f, -0.0022855850f, -0.385578573f, 0.0370728001f,
                                   -0.382134795f, 0.0607641526f, -0.248751760f, 0.0347689129f};
  std::vector<float> expected_Y_h = {-0.382134795f, 0.0607641526f,
                                     -0.385578573f, 0.0370728001f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h, true);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpSequenceLengthWithBidirectionalLinearBeforeReset) {
  const std::string direction = "bidirectional";
  const std::vector<std::string> activations = {"sigmoid", "tanh", "sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 2;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          0.855351f, 0.676391f,
                          -0.185934f, -0.269585f,
                          0.585934f, 0.669585f};
  std::vector<int> sequence_length = {2, 1};
  std::vector<float> initial_h = {0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> expected_Y = {-0.0325528607f, 0.0774837881f, -0.275918573f, -0.00228558504f,
                                   -0.0559310019f, 0.101836264f, -0.275918573f, -0.00228558504f,

                                   -0.0577347837f, 0.0796165839f, 0.0f, 0.0f,
                                   -0.0456649922f, 0.0462125242f, 0.0f, 0.0f};
  std::vector<float> expected_Y_h = {-0.0577347837f, 0.0796165839f,
                                     -0.275918573f, -0.00228558504f,
                                     -0.0559310019f, 0.101836264f,
                                     -0.275918573f, -0.00228558504f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h, true);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpShorterSeqInMiddle) {
  const std::string direction = "bidirectional";
  const std::vector<std::string> activations = {"sigmoid", "tanh", "sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 3;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          0.855351f, 0.676391f,
                          -0.185934f, -0.269585f,
                          -0.585934f, 0.669585f,
                          -0.351455f, -0.391276f,
                          0.670351f, 0.894676f};
  std::vector<int> sequence_length = {2, 1, 2};
  std::vector<float> initial_h = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> expected_Y = {-0.0325528607f, 0.0774837881f, -0.275918573f, -0.00228558504f, -0.0456649921f, 0.0462125241f,
                                   -0.108452908f, 0.15118938684f, -0.2759185731f, -0.0022855850f, -0.1950065642f, 0.0961040258f,

                                   -0.1671274304f, 0.1817691028f, 0.0f, 0.0f, -0.3073617219f, 0.0686715841f,
                                   -0.1494070887f, 0.1356348693f, 0.0f, 0.0f, -0.2866500020f, 0.0448506586f};
  std::vector<float> expected_Y_h = {-0.1671274304f, 0.18176910281f,
                                     -0.2759185731f, -0.00228558504f,
                                     -0.3073617219f, 0.0686715841f,

                                     -0.1084529086f, 0.15118938684f,
                                     -0.2759185731f, -0.00228558504f,
                                     -0.1950065642f, 0.0961040258f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h, true);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpZeroSeqInMiddle) {
  const std::string direction = "bidirectional";
  const std::vector<std::string> activations = {"sigmoid", "tanh", "sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 4;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          0.855351f, 0.676391f,
                          -0.185934f, -0.269585f,
                          -0.585934f, 0.669585f,
                          -0.351455f, -0.391276f,
                          0.670351f, 0.894676f,
                          0.987653f, 1.876567f,
                          -1.234357f, -0.775668f};
  std::vector<int> sequence_length = {2, 0, 2, 2};
  std::vector<float> initial_h = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

  std::vector<float> expected_Y = {-0.0325528607f, 0.0774837881f, 0.0f, 0.0f, -0.0456649921f, 0.0462125241f, -0.1494070887f, 0.1356348693f,
                                   -0.0398676469f, 0.1030099019f, 0.0f, 0.0f, -0.2552363872f, 0.1258624643f, -0.1111927852f, 0.1987708956f,

                                   -0.0317345410f, 0.0898682102f, 0.0f, 0.0f, -0.4344840049f, 0.1124109625f, -0.0373909101f, 0.1958667039f,
                                   -0.0190722197f, 0.0559314489f, 0.0f, 0.0f, -0.4121740460f, 0.0858790874f, 0.0524947792f, 0.1172080263f};

  std::vector<float> expected_Y_h = {-0.0317345410f, 0.0898682102f, 0.0f, 0.0f, -0.4344840049f, 0.1124109625f, -0.0373909101f, 0.1958667039f,

                                     -0.0398676469f, 0.1030099019f, 0.0f, 0.0f, -0.2552363872f, 0.1258624643f, -0.1111927852f, 0.1987708956f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h, true);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpSequenceLengthWithPartialZero) {
  const std::string direction = "bidirectional";
  const std::vector<std::string> activations = {"sigmoid", "tanh", "sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch_size = 2;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          0.455351f, 0.276391f,
                          -0.185934f, -0.269585f,
                          0.185934f, 0.269585f};
  std::vector<int> sequence_length = {2, 0};
  std::vector<float> initial_h = {0.0f, 0.0f, 0.0f, 0.0f,
                                  0.0f, 0.0f, 0.0f, 0.0f};
  std::vector<float> expected_Y = {-0.03255286f, 0.0774838f, 0.0f, 0.0f,
                                   -0.05469977f, 0.1004222f, 0.0f, 0.0f,

                                   -0.05556786f, 0.0785508f, 0.0f, 0.0f,
                                   -0.04566499f, 0.04621252f, 0.0f, 0.0f};
  std::vector<float> expected_Y_h = {-0.05556786f, 0.0785508f,
                                     0.0f, 0.0f,
                                     -0.05469977f, 0.1004222f,
                                     0.0f, 0.0f};

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpSequenceLengthShorterThanInputSequenceLength) {
  const std::string direction = "bidirectional";
  const std::vector<std::string> activations = {"sigmoid", "tanh", "sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch = 1;
  const int seq_length = 2;

  std::vector<float> X = {-0.455351f, -0.276391f,
                          -0.185934f, -0.269585f};

  std::vector<int> sequence_lengths = {1};

  std::vector<float> initial_h = {0.0f, 0.0f,
                                  -0.04566499f, 0.04621252f};

  std::vector<float> expected_Y = {-0.03255286f, 0.0774838f,
                                   -0.05469977f, 0.1004222f,

                                   0.0f, 0.0f,
                                   0.0f, 0.0f};

  std::vector<float> expected_Y_h = {-0.03255286f, 0.0774838f,
                                     -0.05469977f, 0.1004222f};

  ctx.RunTest(X, batch, seq_length, sequence_lengths, &initial_h, expected_Y, expected_Y_h);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpSequenceLengthAllZeros) {
  const std::string direction = "forward";
  const std::vector<std::string> activations = {"sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations);

  const int batch = 2;
  const int seq_length = 2;
  std::vector<float> X = {-0.455351f, -0.276391f,
                          -0.455351f, -0.276391f,

                          -0.185934f, -0.269585f,
                          0.0f, 0.0f};
  std::vector<int> sequence_length = {0, 0};
  std::vector<float> initial_h = {0.0f, 0.0f,
                                  0.0f, 0.0f};
  std::vector<float> expected_Y = {0.0f, 0.0f,
                                   0.0f, 0.0f,

                                   0.0f, 0.0f,
                                   0.0f, 0.0f};

  std::vector<float> expected_Y_h = {0.0f, 0.0f,
                                     0.0f, 0.0f};

  ctx.RunTest(X, batch, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);
}

TEST(GRUTest, ONNXRuntime_TestGRUOpSingleBatchMultipleHiddenThreads) {
  const std::string direction = "forward";
  const std::vector<std::string> activations = {"sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations, true, {}, {}, /*large_hidden*/ true);

  const int batch_size = 1;
  const int seq_length = 1;
  std::vector<float> X = {0.1f, -0.2f};
  std::vector<int> sequence_length = {1};
  std::vector<float> initial_h(32);
  for (int i = 0; i < 32; i++) {
    initial_h[i] = 0.5f;
  }

  std::vector<float> expected_Y =
      {
          0.40203814648622f, 0.416614999456787f, 0.426893838272102f, 0.438425099258723f,
          0.449074949310697f, 0.405161353080481f, 0.41381287883561f, 0.428113160854675f,
          0.438710576166608f, 0.449253502147958f, 0.402300128581669f, 0.417112500336769f,
          0.425382986540999f, 0.439390099095881f, 0.450276939756071f, 0.404653232879823f,
          0.414327989766397f, 0.428845403314675f, 0.436736277602997f, 0.45043439079097f,
          0.402560202956173f, 0.416113639384501f, 0.426141512516655f, 0.44034669402871f,
          0.447861672303443f, 0.40490822137737f, 0.414839135658386f, 0.42737488368901f,
          0.437727744598091f, 0.451604294166264f, 0.40203814648622f, 0.416614999456787f};
  std::vector<float> expected_Y_h(expected_Y);

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);
}

TEST(GRUTest, ONNXRuntime_TestGRUPositiveActivationClipping) {
  const std::string direction = "forward";
  const std::vector<std::string> activations = {"sigmoid", "tanh"};

  DeepCpuGruOpTestContext ctx(direction, activations, true, {}, {}, /*large_hidden*/ true);

  const int batch_size = 2;
  const int seq_length = 1;
  std::vector<float> X = {1000.0f, 2000.0f, -1e+20f, -1e+10f};
  std::vector<int> sequence_length = {1, 1};
  std::vector<float> initial_h(64);
  for (int i = 0; i < 64; i++) {
    initial_h[i] = 0.25f;
  }

  std::vector<float> expected_Y(64);
  for (int i = 0; i < 32; i++) {
    expected_Y[i] = 0.25f;
  }

  for (int i = 32; i < 64; i++) {
    expected_Y[i] = -1.0f;
  }

  std::vector<float> expected_Y_h(expected_Y);

  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);
}

TEST(GRUTest, ONNXRuntime_TestGRUPositiveActivationAlphaBeta) {
  const std::string direction = "bidirectional";
  const std::vector<std::string> activations = {"LeakyRelu", "Tanh", "Sigmoid", "ScaledTanh"};
  const std::vector<float> alpha = {0.5f, 2.0f};
  const std::vector<float> beta = {2.0f};

  const int input_size = 2;  //  4;
  const int batch_size = 1;
  const int seq_length = 1;
  std::vector<float> X = {1.0f, 2.0f};  //  , -3.0f, -4.0f};

  std::vector<int> sequence_length = {1};
  std::vector<float> initial_h(64);
  for (int i = 0; i < 64; i++) {
    initial_h[i] = 0.25f;
  }

  std::vector<float> expected_Y =
      {
          0.589157104133446f, 0.57789190635582f, 0.596499289838044f, 0.583932266186645f,
          0.602660999282266f, 0.574613517108484f, 0.592591742002174f, 0.581152059107645f,
          0.599870466365438f, 0.58688901119379f, 0.589262946094602f, 0.578086950802852f,
          0.595908617320326f, 0.584296265507284f, 0.603117381311498f, 0.574411143084165f,
          0.592796083917448f, 0.581434103323347f, 0.599111005558765f, 0.58732791235695f,
          0.589368676977052f, 0.577696448871644f, 0.59620442543072f, 0.584658706958245f,
          0.602202148066235f, 0.574512383220923f, 0.592999994021163f, 0.580869112219442f,
          0.59949155030307f, 0.587764451689259f, 0.589157104133446f, 0.57789190635582f,
          0.92976461079108f, 0.920318863429123f, 0.931350963541001f, 0.921736050322775f,
          0.932696544850119f, 0.91952832256688f, 0.930568024045208f, 0.921062381128771f,
          0.93205941179011f, 0.922380156527733f, 0.929770910118088f, 0.920330320081968f,
          0.931318215990949f, 0.921756237467303f, 0.932720157970373f, 0.919515978936628f,
          0.930579785457139f, 0.921078414664293f, 0.932018665973048f, 0.922403708998701f,
          0.929777143371747f, 0.920307279340899f, 0.931334684571751f, 0.921776181825893f,
          0.932672631849138f, 0.919522183290344f, 0.930591417482971f, 0.921046160840421f,
          0.932039162132774f, 0.922426966110166f, 0.92976461079108f, 0.920318863429123f};

  std::vector<float> expected_Y_h(expected_Y);

  DeepCpuGruOpTestContext ctx(direction, activations, true, alpha, beta, /*large_hidden*/ true, input_size);
  ctx.RunTest(X, batch_size, seq_length, sequence_length, &initial_h, expected_Y, expected_Y_h);
}

}  // namespace test
}  // namespace onnxruntime

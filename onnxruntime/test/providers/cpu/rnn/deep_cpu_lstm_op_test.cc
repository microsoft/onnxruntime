// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <iterator>
#include <vector>

#include "core/providers/cpu/rnn/deep_cpu_lstm.h"
#include "test/providers/provider_test_utils.h"
#include "default_providers.h"

using namespace std;
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

static void RunLstmTest(const std::vector<float>& X_data,
                        const std::vector<float>& W_data,
                        bool is_initializer_W,
                        const std::vector<float>& R_data,
                        bool is_initializer_R,
                        const std::vector<float>& Y_data,
                        const std::vector<float>& Y_h_data,
                        const std::vector<float>& Y_c_data,
                        int64_t input_size,
                        int64_t batch_size,
                        int64_t hidden_size,
                        int64_t seq_length,
                        const std::vector<float>* B_data = nullptr,
                        const std::vector<float>* P_data = nullptr,
                        const std::vector<float>* initial_h_data = nullptr,
                        const std::vector<float>* initial_c_data = nullptr,
                        const std::vector<int>* sequence_lengths = nullptr,
                        const std::string& direction = "forward",
                        float clip = 9999.f,
                        bool output_sequence = true,
                        bool input_forget = false,
                        // copy the following vectors as we may modify them
                        std::vector<string> activations = {},
                        std::vector<float> activation_alphas = {},
                        std::vector<float> activation_betas = {},
                        bool hasClip = true) {
  OpTester test("LSTM");

  int num_directions = (direction == "bidirectional") ? 2 : 1;

  if (activations.empty()) {
    activations = {"sigmoid", "tanh", "tanh"};
  }

  if (num_directions == 2 && activations.size() == 3) {
    activations = DuplicateContainer(activations);
  }

  test.AddAttribute<std::vector<string>>("activations", activations);
  if (!activation_alphas.empty())
    test.AddAttribute<std::vector<float>>("activation_alpha", activation_alphas);
  if (!activation_betas.empty())
    test.AddAttribute<std::vector<float>>("activation_beta", activation_betas);

  test.AddAttribute("direction", direction);
  test.AddAttribute("hidden_size", hidden_size);
  // test.AddAttribute<int64_t>("output_sequence", output_sequence);
  test.AddAttribute<int64_t>("input_forget", input_forget);
  if (hasClip) {
    test.AddAttribute<float>("clip", clip);
  }

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<int64_t> W_dims = {num_directions, 4 * hidden_size, input_size};
  std::vector<int64_t> R_dims = {num_directions, 4 * hidden_size, hidden_size};

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("W", W_dims, W_data, is_initializer_W);
  test.AddInput<float>("R", R_dims, R_data, is_initializer_R);

  if (B_data) {
    std::vector<int64_t> B_dims = {num_directions, 8 * hidden_size};
    test.AddInput<float>("B", B_dims, *B_data);
  } else {
    test.AddOptionalInputEdge<float>();
  }

  if (sequence_lengths) {
    std::vector<int64_t> sequence_lens_dims{batch_size};
    test.AddInput<int>("sequence_lens", sequence_lens_dims, *sequence_lengths);
  } else {
    test.AddOptionalInputEdge<int>();
  }

  if (initial_h_data && !initial_h_data->empty()) {
    std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
    test.AddInput<float>("initial_h", initial_h_dims, *initial_h_data);
  } else {
    test.AddOptionalInputEdge<float>();
  }

  if (initial_c_data && !initial_c_data->empty()) {
    std::vector<int64_t> initial_c_dims = {num_directions, batch_size, hidden_size};
    test.AddInput<float>("initial_c", initial_c_dims, *initial_c_data);
  } else {
    test.AddOptionalInputEdge<float>();
  }

  if (P_data && !P_data->empty()) {
    std::vector<int64_t> P_dims = {num_directions, 3 * hidden_size};
    test.AddInput<float>("P", P_dims, *P_data);
  } else {
    test.AddOptionalInputEdge<float>();
  }

  if (output_sequence != 0 && !Y_data.empty()) {
    std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y", Y_dims, Y_data);
  } else {
    // add placeholder so node counts match as Y_h will always be the second Y_data,
    // so Y must exist as the first Y_data
    test.AddOptionalOutputEdge<float>();
  }

  if (!Y_h_data.empty()) {
    std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);
  } else {
    test.AddOptionalOutputEdge<float>();
  }

  if (!Y_c_data.empty()) {
    std::vector<int64_t> Y_c_dims{num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y_c", Y_c_dims, Y_c_data);
  } else {
    test.AddOptionalOutputEdge<float>();
  }

  test.SetOutputTolerance(0.0001f);

  // TensorRT failed on LSTM tests
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

void SimpleWeightsNoBiasTwoRows(std::string direction,
                                const std::vector<float>& Y_data,
                                const std::vector<float>& Y_h_data,
                                const std::vector<float>& Y_c_data,
                                const std::vector<int>* seq_lengths = nullptr) {
  int64_t seq_length = 2;
  int batch_size = 2;
  int64_t input_size = 1;
  int64_t hidden_size = 3;

  int num_directions = direction == "bidirectional" ? 2 : 1;

  std::vector<float> X_data{1.f, 2.f, 10.f, 11.f};

  std::vector<float> W_data{
      0.1f, 0.2f, 0.3f, 0.4f,
      1.f, 2.f, 3.f, 4.f,
      10.f, 11.f, 12.f, 13.f};

  std::vector<float> R_data(num_directions * 4 * hidden_size * hidden_size, 0.1f);

  // duplicate for bidirectional
  if (num_directions == 2) {
    W_data = DuplicateContainer(W_data);
  }

  RunLstmTest(X_data, W_data, false, R_data, false, Y_data, Y_h_data, Y_c_data,
              input_size, batch_size, hidden_size, seq_length,
              nullptr, nullptr, nullptr, nullptr, seq_lengths, direction);

  // need at least one output, so we need Y_h or Y_c to be requested (non-empty output to compare against) in order
  // to test Y not being returned (output_sequence == false)
  if (!Y_h_data.empty() || !Y_c_data.empty())
    RunLstmTest(X_data, W_data, false, R_data, false, Y_data, Y_h_data, Y_c_data,
                input_size, batch_size, hidden_size, seq_length,
                nullptr, nullptr, nullptr, nullptr, seq_lengths, direction, 999.f, /* output_sequence*/ false);
}

TEST(LSTMTest, ForwardSimpleWeightsNoBiasTwoRows) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  std::vector<float> Y_data{
      0.28828835f, 0.36581863f, 0.45679406f,
      0.34526032f, 0.47220859f, 0.55850911f,

      0.84196719f, 0.89402526f, 0.91073048f,
      0.85882828f, 0.90703777f, 0.92382453f};

  std::vector<float> Y_h_data{
      0.84196719f, 0.89402526f, 0.91073048f,
      0.85882828f, 0.90703777f, 0.92382453f};

  std::vector<float> Y_c_data{
      1.27731147f, 1.44181041f, 1.53179041f,
      1.3249796f, 1.51063104f, 1.61451544f};

  SimpleWeightsNoBiasTwoRows("forward", Y_data, Y_h_data, Y_c_data);

  // test Y_h and Y_c being optional
  SimpleWeightsNoBiasTwoRows("forward", Y_data, {}, {});
}

TEST(LSTMTest, ReverseSimpleWeightsNoBiasTwoRows) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  std::vector<float> Y_data{
      0.55391603f, 0.69201493f, 0.82696019f,
      0.64046413f, 0.82303363f, 0.91610711f,

      0.61249432f, 0.70678632f, 0.74094619f,
      0.62759886f, 0.71640738f, 0.74624585f};

  std::vector<float> Y_h_data{
      0.55391603f, 0.69201493f, 0.82696019f,
      0.64046413f, 0.82303363f, 0.91610711f};

  std::vector<float> Y_c_data{
      1.27850552f, 1.46799496f, 1.57641257f,
      1.34960834f, 1.54772296f, 1.65633056f};

  SimpleWeightsNoBiasTwoRows("reverse", Y_data, Y_h_data, Y_c_data);
}

TEST(LSTMTest, BidirectionalSimpleWeightsNoBiasTwoRows) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  std::vector<float> Y_data{
      0.28828835f, 0.36581863f, 0.45679406f,
      0.34526032f, 0.47220859f, 0.55850911f,

      0.55391603f, 0.69201493f, 0.82696019f,
      0.64046413f, 0.82303363f, 0.91610711f,

      0.84196719f, 0.89402526f, 0.91073048f,
      0.85882828f, 0.90703777f, 0.92382453f,

      0.61249432f, 0.70678632f, 0.74094619f,
      0.62759886f, 0.71640738f, 0.74624585f};

  std::vector<float> Y_h_data{
      // we did the forward processing of X_data[1] last
      0.84196719f, 0.89402526f, 0.91073048f,
      0.85882828f, 0.90703777f, 0.92382453f,

      // and the reverse processing of X_data[0] last as the X_data order was reversed
      0.55391603f, 0.69201493f, 0.82696019f,
      0.64046413f, 0.82303363f, 0.91610711f};

  std::vector<float> Y_c_data{
      1.27731147f, 1.44181041f, 1.53179041f,
      1.3249796f, 1.51063104f, 1.61451544f,

      1.27850552f, 1.46799496f, 1.57641257f,
      1.34960834f, 1.54772296f, 1.65633056f};

  // cudnn don't support customized activation
  SimpleWeightsNoBiasTwoRows("bidirectional", Y_data, Y_h_data, Y_c_data);
}

TEST(LSTMTest, MixedSequenceLengths) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  // we don't have numpy output for this, but by testing twice and swapping which batch is smaller
  // we can largely verify the behaviour by comparing to ForwardSimpleWeightsNoBiasTwoRows output.
  std::vector<int> seq_lengths{1, 2};

  std::vector<float> Y_data{
      0.28828835f, 0.36581863f, 0.45679406f,
      0.34526032f, 0.47220859f, 0.55850911f,

      0.f, 0.f, 0.f,
      0.85882828f, 0.90703777f, 0.92382453f};

  std::vector<float> Y_h_data{
      0.28828835f, 0.36581863f, 0.45679406f,
      0.85882828f, 0.90703777f, 0.92382453f};

  std::vector<float> Y_c_data{
      0.52497941f, 0.54983425f, 0.5744428f,  // see intermediate output from ForwardSimpleWeightsNoBiasTwoRows
      1.3249796f, 1.51063104f, 1.61451544f};

  // Not able to mask on Y_c for CUDA using cudnn lib
  SimpleWeightsNoBiasTwoRows("forward", Y_data, Y_h_data, Y_c_data, &seq_lengths);

  // swap which one is short
  seq_lengths = {2, 1};

  Y_data = {
      0.28828835f, 0.36581863f, 0.45679406f,
      0.34526032f, 0.47220859f, 0.55850911f,

      0.84196719f, 0.89402526f, 0.91073048f,
      0.f, 0.f, 0.f};

  Y_h_data = {
      0.84196719f, 0.89402526f, 0.91073048f,
      0.34526032f, 0.47220859f, 0.55850911f};

  Y_c_data = {
      1.27731147f, 1.44181041f, 1.53179041f,
      0.54983425f, 0.59868795f, 0.64565659f};

  SimpleWeightsNoBiasTwoRows("forward", Y_data, Y_h_data, Y_c_data, &seq_lengths);
}

TEST(LSTMTest, MixedSequenceLengthsReverse) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  // we don't have numpy output for this, but by testing twice and swapping which batch is smaller
  // we can largely verify the behaviour by comparing to ReverseSimpleWeightsNoBiasTwoRows output.
  std::vector<int> seq_lengths{1, 2};

  std::vector<float> Y_data{
      0.28828844f, 0.36581877f, 0.45679423f,
      0.64046413f, 0.82303363f, 0.91610711f,

      0.f, 0.f, 0.f,
      0.62759886f, 0.71640738f, 0.74624585f};

  std::vector<float> Y_h_data{
      0.28828844f, 0.36581877f, 0.45679423f,
      0.64046413f, 0.82303363f, 0.91610711f};

  std::vector<float> Y_c_data{
      0.52497941f, 0.54983425f, 0.5744428f,
      1.34960834f, 1.54772296f, 1.65633056f};

  SimpleWeightsNoBiasTwoRows("reverse", Y_data, Y_h_data, Y_c_data, &seq_lengths);

  // swap which one is short
  seq_lengths = {2, 1};

  Y_data = {
      0.55391603f, 0.69201493f, 0.82696019f,
      0.34526044f, 0.47220877f, 0.55850935f,

      0.61249432f, 0.70678632f, 0.74094619f,
      0.f, 0.f, 0.f};

  Y_h_data = {
      0.55391603f, 0.69201493f, 0.82696019f,
      0.34526044f, 0.47220877f, 0.55850935f};

  Y_c_data = {
      1.27850552f, 1.46799496f, 1.57641257f,
      0.54983425f, 0.59868795f, 0.64565659f};

  SimpleWeightsNoBiasTwoRows("reverse", Y_data, Y_h_data, Y_c_data, &seq_lengths);
}

// test path in LSTM model where batch_parallel_ is false and there are multiple steps (seq_length > 1)
TEST(LSTMTest, BatchParallelFalseSeqLengthGreaterThanOne) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  int64_t seq_length = 2;
  int batch_size = 1;
  int64_t input_size = 1;
  int64_t hidden_size = 2;

  int num_directions = 1;

  std::vector<float> X_data{1.f, 2.f};

  std::vector<float> W_data{
      0.1f, 0.2f, 0.3f, 0.4f,
      1.f, 2.f, 3.f, 4.f};

  std::vector<float> R_data(num_directions * 4 * hidden_size * hidden_size, 0.1f);

  std::vector<float> Y_data{
      0.27546653f, 0.29941525f,
      0.50903179f, 0.57476457f};

  std::vector<float> Y_c_data{
      1.02721067f, 1.15254318f};

  for (bool is_initializer_W : std::initializer_list<bool>{false, true}) {
    for (bool is_initializer_R : std::initializer_list<bool>{false, true}) {
      RunLstmTest(X_data, W_data, is_initializer_W, R_data, is_initializer_R,
                  Y_data, {}, Y_c_data, input_size, batch_size, hidden_size, seq_length);
    }
  }
}

// make sure GateComputations works correctly if batch_parallel_ is true due to large batch size
static void LargeBatchWithClip(const std::vector<float>& Y_h_data, float clip = 9999.0) {
  int64_t seq_length = 2;
  int batch_size = 32;
  int64_t input_size = 1;
  int64_t hidden_size = 3;

  const std::string direction = "forward";
  int num_directions = 1;

  std::vector<float> X_data;

  // generate input of 64 values
  float i = 0.f, increment = 1.f;
  std::generate_n(std::back_inserter(X_data), batch_size * seq_length, [&]() { return i += increment; });

  std::vector<float> W_data{0.1f, 0.2f, 0.3f, 0.4f,
                            1.f, 2.f, 3.f, 4.f,
                            10.f, 11.f, 12.f, 13.f};

  std::vector<float> R_data(num_directions * 4 * hidden_size * hidden_size, 0.1f);

  for (bool is_initializer_W : std::initializer_list<bool>{false, true}) {
    for (bool is_initializer_R : std::initializer_list<bool>{false, true}) {
      RunLstmTest(X_data, W_data, is_initializer_W, R_data, is_initializer_R, {},
                  Y_h_data, {}, input_size, batch_size, hidden_size, seq_length,
                  nullptr, nullptr, nullptr, nullptr, nullptr, direction, clip);
    }
  }
}

TEST(LSTMTest, LargeBatchNoClipping) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  std::vector<float> Y_h_data = {
      0.90387899f, 0.9135572f, 0.91772245f,
      0.90897038f, 0.92132433f, 0.92825467f,
      0.91365823f, 0.92815113f, 0.93676105f,
      0.91799162f, 0.93406357f, 0.94344562f,
      0.92199681f, 0.93912057f, 0.94859476f,
      0.92569357f, 0.94340185f, 0.95250664f,
      0.92909964f, 0.94699686f, 0.95545127f,
      0.93223207f, 0.94999634f, 0.95765468f,
      0.93510761f, 0.9524867f, 0.95929726f,
      0.93774272f, 0.9545467f, 0.96051891f,
      0.9401536f, 0.95624603f, 0.96142619f,
      0.94235605f, 0.95764499f, 0.96209939f,
      0.94436539f, 0.95879495f, 0.96259862f,
      0.94619635f, 0.95973921f, 0.96296872f,
      0.94786299f, 0.96051397f, 0.96324302f,
      0.94937864f, 0.96114929f, 0.96344629f,
      0.95075587f, 0.96167006f, 0.96359692f,
      0.95200645f, 0.96209679f, 0.96370852f,
      0.95314133f, 0.9624464f, 0.9637912f,
      0.95417069f, 0.96273278f, 0.96385246f,
      0.95510395f, 0.96296733f, 0.96389785f,
      0.95594975f, 0.96315942f, 0.96393147f,
      0.95671607f, 0.96331673f, 0.96395638f,
      0.9574102f, 0.96344554f, 0.96397483f,
      0.9580388f, 0.96355102f, 0.9639885f,
      0.95860795f, 0.96363739f, 0.96399863f,
      0.95912322f, 0.96370811f, 0.96400613f,
      0.95958963f, 0.96376601f, 0.96401169f,
      0.96001179f, 0.96381342f, 0.96401581f,
      0.96039386f, 0.96385224f, 0.96401886f,
      0.96073964f, 0.96388402f, 0.96402112f,
      0.96105254f, 0.96391004f, 0.96402279f};

  LargeBatchWithClip(Y_h_data);
}

// make sure GateComputations with clipping works correctly if batch_parallel_ is true due to large batch size
TEST(LSTMTest, LargeBatchWithClip) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  std::vector<float> Y_h_data = {
      0.88572926f, 0.89251395f, 0.89655037f,
      0.89074291f, 0.90035688f, 0.90727429f,
      0.89535827f, 0.90727429f, 0.91596163f,
      0.89963124f, 0.91328279f, 0.9228067f,
      0.90358195f, 0.91843507f, 0.92809163f,
      0.90723279f, 0.9228067f, 0.93211437f,
      0.91038955f, 0.92648469f, 0.93514718f,
      0.91328279f, 0.92955856f, 0.93741938f,
      0.91596163f, 0.93211437f, 0.9391149f,
      0.91843507f, 0.93423112f, 0.94037686f,
      0.92071318f, 0.9359791f, 0.94131462f,
      0.9228067f, 0.93741938f, 0.94201073f,
      0.92472679f, 0.9386042f, 0.94252713f,
      0.92648469f, 0.9395777f, 0.94266769f,
      0.92809163f, 0.94037686f, 0.94266769f,
      0.92955856f, 0.94103248f, 0.94266769f,
      0.93089609f, 0.94157007f, 0.94266769f,
      0.93211437f, 0.94201073f, 0.94266769f,
      0.93322302f, 0.94237184f, 0.94266769f,
      0.93423112f, 0.94266769f, 0.94266769f,
      0.93514718f, 0.94266769f, 0.94266769f,
      0.9359791f, 0.94266769f, 0.94266769f,
      0.93673424f, 0.94266769f, 0.94266769f,
      0.93741938f, 0.94266769f, 0.94266769f,
      0.93804079f, 0.94266769f, 0.94266769f,
      0.9386042f, 0.94266769f, 0.94266769f,
      0.9391149f, 0.94266769f, 0.94266769f,
      0.9395777f, 0.94266769f, 0.94266769f,
      0.93999702f, 0.94266769f, 0.94266769f,
      0.94037686f, 0.94266769f, 0.94266769f,
      0.94072091f, 0.94266769f, 0.94266769f,
      0.94103248f, 0.94266769f, 0.94266769f};

  LargeBatchWithClip(Y_h_data, 4.f);
}

// ONNXRuntime tests
class LstmOpContext2x1x2x2 {
 public:
  LstmOpContext2x1x2x2(const std::string direction,
                       const std::vector<std::string>& activations = {},
                       const std::vector<float>& activation_alphas = {},
                       const std::vector<float>& activation_betas = {})
      : direction_(direction),
        num_directions_(direction == "bidirectional" ? 2 : 1),
        activation_func_names_{activations},
        activation_alphas_{activation_alphas},
        activation_betas_{activation_betas} {
    // W[iofc] (4*hidden, X_data)
    input_weights_ = {
        -0.494659f, 0.0453352f,
        -0.487793f, 0.417264f,

        -0.0175329f, 0.489074f,
        -0.446013f, 0.414029f,

        -0.0091708f, -0.255364f,
        -0.106952f, -0.266717f,

        -0.0888852f, -0.428709f,
        -0.283349f, 0.208792f};

    // R[iofc] (4*hidden, hidden)
    recurrent_weights_ = {
        0.146626f, -0.0620289f,
        -0.0815302f, 0.100482f,

        -0.219535f, -0.306635f,
        -0.28515f, -0.314112f,

        -0.228172f, 0.405972f,
        0.31576f, 0.281487f,

        -0.394864f, 0.42111f,
        -0.386624f, -0.390225f};

    // P[iof] (3*hidden)
    peephole_weights_ = {
        0.2345f, 0.5235f,
        0.4378f, 0.3475f,
        0.8927f, 0.3456f};

    // Wb[iofc], Rb[iofc] (8*hidden)
    bias_ = {
        // Wb[iofc]
        0.381619f, 0.0323954f,
        -0.14449f, 0.420804f,
        -0.258721f, 0.45056f,
        -0.250755f, 0.0967895f,

        // Rb[iofc]
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f,
        0.0f, 0.0f};

    if (num_directions_ == 2) {
      input_weights_ = DuplicateContainer(input_weights_);
      recurrent_weights_ = DuplicateContainer(recurrent_weights_);
      peephole_weights_ = DuplicateContainer(peephole_weights_);
      bias_ = DuplicateContainer(bias_);
    }

    /*

        def.add_arg()->CopyFrom(caffe2::MakeArgument("direction", direction));
        def.add_arg()->CopyFrom(caffe2::MakeArgument("hidden_size", _hidden_dim));
        def.add_arg()->CopyFrom(caffe2::MakeArgument("activations", activations));
        def.add_arg()->CopyFrom(caffe2::MakeArgument("clip", clip));
        def.add_arg()->CopyFrom(caffe2::MakeArgument("input_forget", input_forget));
        def.add_arg()->CopyFrom(caffe2::MakeArgument("output_sequence", output_sequence));

        FillTensor<caffe2::CPUContext, float, float>(&_ws, "X0", { seq_len, batch_size, _input_dim }, X_data);

        if (num_direction == 1) {
            FillTensor<caffe2::CPUContext, float, float>(&_ws, "X1", { num_direction, 4 * _hidden_dim, _input_dim }, input_weights);
            FillTensor<caffe2::CPUContext, float, float>(&_ws, "X2", { num_direction, 4 * _hidden_dim, _hidden_dim }, recurrent_weights);

            if (use_bias)
                FillTensor<caffe2::CPUContext, float, float>(&_ws, "X3", { num_direction, 8 * _hidden_dim }, bias);

            if (seq_length.size())
                FillTensor<caffe2::CPUContext, int, int>(&_ws, "X4", { batch_size }, seq_length);

            if (hidden_state.size())
                FillTensor<caffe2::CPUContext, float, float>(&_ws, "X5", { num_direction, batch_size, _hidden_dim }, hidden_state);

            if (cell_state.size())
                FillTensor<caffe2::CPUContext, float, float>(&_ws, "X6", { num_direction, batch_size, _hidden_dim }, cell_state);

            if (use_peepholes)
                FillTensor<caffe2::CPUContext, float, float>(&_ws, "X7", { num_direction, 3 * _hidden_dim }, peephole_weights);
        }
        if (num_direction == 2) {
            FillTensor<caffe2::CPUContext, float, float>(&_ws, "X1", { num_direction, 4 * _hidden_dim, _input_dim }, bi_input_weights);
            FillTensor<caffe2::CPUContext, float, float>(&_ws, "X2", { num_direction, 4 * _hidden_dim, _hidden_dim }, bi_recurrent_weights);

            if (use_bias)
                FillTensor<caffe2::CPUContext, float, float>(&_ws, "X3", { num_direction, 8 * _hidden_dim }, bi_bias);

            if (seq_length.size())
                FillTensor<caffe2::CPUContext, int, int>(&_ws, "X4", { batch_size }, seq_length);

            if (hidden_state.size())
                FillTensor<caffe2::CPUContext, float, float>(&_ws, "X5", { num_direction, batch_size, _hidden_dim }, hidden_state);

            if (cell_state.size())
                FillTensor<caffe2::CPUContext, float, float>(&_ws, "X6", { num_direction, batch_size, _hidden_dim }, cell_state);

            if (use_peepholes)
                FillTensor<caffe2::CPUContext, float, float>(&_ws, "X7", { num_direction, 3 * _hidden_dim }, bi_peephole_weights);
        }

        _op = caffe2::CreateOperator(def, &_ws);
        */

    // RunTest(seq_len, batch_size, num_direction, Y_data, output_first);
  }

  void RunTest(const std::vector<float>& X,
               const int batch_size,
               const int seq_length,
               const std::vector<float>* initial_h,
               const std::vector<float>* initial_c,
               const std::vector<float>& expected_Y,
               const std::vector<float>& expected_Y_h = {},
               const std::vector<float>& expected_Y_c = {},
               const std::vector<int>* sequence_lens = nullptr,
               bool use_bias = true,
               bool use_peepholes = true,
               float clip = 9999.f,
               bool input_forget = false,
               bool hasClip = true) {
    // run with and without output_sequence to test UniDirectionalLstm handling when Y isn't returned
    for (bool output_sequence : std::initializer_list<bool>{false, true}) {
      ::onnxruntime::test::RunLstmTest(X,
                                       input_weights_, false,
                                       recurrent_weights_, false,
                                       expected_Y, expected_Y_h, expected_Y_c,
                                       input_size_, batch_size, hidden_size_, seq_length,
                                       use_bias ? &bias_ : nullptr,
                                       use_peepholes ? &peephole_weights_ : nullptr,
                                       initial_h, initial_c,
                                       sequence_lens,
                                       direction_,
                                       clip,
                                       output_sequence,
                                       input_forget,
                                       activation_func_names_,
                                       activation_alphas_,
                                       activation_betas_,
                                       hasClip);
    }
  }

 private:
  const int input_size_ = 2;
  const int hidden_size_ = 2;
  const std::string direction_;
  int num_directions_;
  const std::vector<std::string> activation_func_names_;
  const std::vector<float> activation_alphas_;
  const std::vector<float> activation_betas_;
  std::vector<float> input_weights_;
  std::vector<float> recurrent_weights_;
  std::vector<float> bias_;
  std::vector<float> peephole_weights_;
};

TEST(LSTMTest, ONNXRuntime_TestLSTMForwardPeepHole) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  ///////////////Attributes////////////////////////
  constexpr int seq_len = 2, batch_size = 1;

  std::vector<float> input = {-0.455351f, -0.276391f, -0.185934f, -0.269585f};
  std::vector<float> Y_data = {-0.0251062475f, 0.0561261699f, -0.03277518f, 0.05935364f};
  std::vector<float> Y_h_data = {-0.03277518f, 0.05935364f};
  std::vector<float> Y_c_data = {-0.0780206f, 0.098829f};

  // Run Test
  LstmOpContext2x1x2x2 context("forward");
  context.RunTest(input, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMBidirectionalBasic) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2, batch_size = 1;

  std::vector<float> X_data = {-0.455351f, -0.276391f,
                               -0.185934f, -0.269585f};
  std::vector<float> Y_data = {-0.0251062f, 0.0561262f,
                               -0.0318928f, 0.0762679f,
                               -0.0327752f, 0.0593536f,
                               -0.0306872f, 0.028035f};
  std::vector<float> Y_h_data = {-0.0327752f, 0.0593536f,
                                 -0.0318928f, 0.0762679f};
  std::vector<float> Y_c_data = {-0.0780206f, 0.098829f,
                                 -0.0753684f, 0.120794f};

  LstmOpContext2x1x2x2 context("bidirectional");
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMForwardNoBiasUsePeepholes) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2, batch_size = 1;

  bool use_bias = false;
  bool use_peepholes = true;
  std::vector<float> X_data = {-0.455351f, -0.276391f,
                               -0.185934f, -0.269585f};

  std::vector<float> Y_data = {0.04154162f, 0.01969122f,
                               0.05298181f, 0.0030589f};
  std::vector<float> Y_h_data = {0.05298181f, 0.0030589f};
  std::vector<float> Y_c_data = {0.11169686f, 0.00625722f};

  LstmOpContext2x1x2x2 context("forward");
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data, nullptr,
                  use_bias, use_peepholes);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMForwardInputForget) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2, batch_size = 1;

  bool use_bias = true;
  bool use_peepholes = true;
  bool input_forget = true;
  float clip = 999.0f;

  std::vector<float> X_data = {-0.455351f, -0.276391f, -0.185934f, -0.269585f};

  std::vector<float> Y_data = {-0.02510626f, 0.05612619f,
                               -0.0314321f, 0.05087372f};
  std::vector<float> Y_h_data = {-0.0314321f, 0.05087372f};
  std::vector<float> Y_c_data = {-0.07474898f, 0.08480116f};

  LstmOpContext2x1x2x2 context("forward");
  // cudnn don't support peepholes
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data, nullptr,
                  use_bias, use_peepholes, clip, input_forget);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMForwardClip) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2, batch_size = 1;

  bool use_bias = true;
  bool use_peepholes = true;
  float clip = 0.1f;

  std::vector<float> X_data = {-0.455351f, -0.276391f, -0.185934f, -0.269585f};

  std::vector<float> Y_data = {-0.02280854f, 0.02744377f,
                               -0.03516197f, 0.03875681f};
  std::vector<float> Y_h_data = {-0.03516197f, 0.03875681f};
  std::vector<float> Y_c_data = {-0.07415761f, 0.07395997f};

  LstmOpContext2x1x2x2 context("forward");
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data, nullptr,
                  use_bias, use_peepholes, clip);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMBackward) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2, batch_size = 1;

  std::vector<float> X_data = {-0.455351f, -0.276391f, -0.185934f, -0.269585f};

  std::vector<float> Y_data = {-0.03189282f, 0.07626793f,
                               -0.03068724f, 0.02803503f};
  std::vector<float> Y_h_data = {-0.03189282f, 0.07626793f};
  std::vector<float> Y_c_data = {-0.07536839f, 0.12079399f};

  LstmOpContext2x1x2x2 context("reverse");
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMBackward_gpu) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2, batch_size = 1;

  std::vector<float> X_data = {-0.455351f, -0.276391f, -0.185934f, -0.269585f};

  std::vector<float> Y_data = {-0.033075746f, 0.074455738f,
                               -0.031248707f, 0.027853041f};
  std::vector<float> Y_h_data = {-0.033075746f, 0.074455738f};
  std::vector<float> Y_c_data = {-0.076699793f, 0.11975205f};

  LstmOpContext2x1x2x2 context("reverse");
  // Disable peephole since cudnn doesn't support it
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data, nullptr, true, false);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMForwardHiddenState) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2, batch_size = 1;

  bool use_bias = true;
  bool use_peepholes = false;

  std::vector<float> X_data = {-0.455351f, -0.276391f, -0.185934f, -0.269585f};
  std::vector<float> hidden_state = {0.34f, 0.72f};

  std::vector<float> Y_data = {0.01797521f, -0.07104912f,
                               -0.03174796f, -0.0152949f};
  std::vector<float> Y_h_data = {-0.03174796f, -0.0152949f};
  std::vector<float> Y_c_data = {-0.07285583f, -0.02545788f};

  LstmOpContext2x1x2x2 context("forward");
  context.RunTest(X_data, batch_size, seq_len, &hidden_state, nullptr, Y_data, Y_h_data, Y_c_data,
                  nullptr, use_bias, use_peepholes);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMForwardCellState) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2, batch_size = 1;

  bool use_bias = true;
  bool use_peepholes = false;

  std::vector<float> X_data = {-0.455351f, -0.276391f, -0.185934f, -0.269585f};
  std::vector<float> hidden_state = {0.34f, 0.72f};
  std::vector<float> cell_state = {0.63f, 0.21f};

  std::vector<float> Y_data = {0.12797015f, 0.0097284f,
                               0.02716939f, 0.01842997f};
  std::vector<float> Y_h_data = {0.02716939f, 0.01842997f};
  std::vector<float> Y_c_data = {0.06408449f, 0.03139432f};

  LstmOpContext2x1x2x2 context("forward");
  context.RunTest(X_data, batch_size, seq_len, &hidden_state, &cell_state, Y_data, Y_h_data, Y_c_data,
                  nullptr, use_bias, use_peepholes);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMActivation) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2, batch_size = 1;

  std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};

  bool use_bias = true;
  bool use_peepholes = false;

  std::vector<float> X_data = {-0.455351f, -0.276391f, -0.185934f, -0.269585f};

  std::vector<float> Y_data = {-0.0660155f, 0.0351227f,
                               -0.04236888f, 0.0177365f};
  std::vector<float> Y_h_data = {-0.04236888f, 0.0177365f};
  std::vector<float> Y_c_data = {0.1624992f, 0.04672481f};

  LstmOpContext2x1x2x2 context("forward", activations);
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data,
                  nullptr, use_bias, use_peepholes);
}

// Original comments:
//   test correctness for batch size > 1 and
//   memory reallocation due to change in batch size
// The reallocation doesn't apply any more so this mainly tests larger batches with non-default activations.
TEST(LSTMTest, ONNXRuntime_TestLSTMBatchReallocation) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  ///////////////Attributes////////////////////////
  constexpr int seq_len = 2;
  int batch_size = 1;
  constexpr bool use_bias = true;
  constexpr bool use_peepholes = false;

  std::vector<std::string> activations = {"tanh", "sigmoid", "tanh"};

  //////////////////Inputs///////////////////////////////////
  std::string direction = "forward";

  std::vector<float> X_data = {-0.455351f, -0.276391f,
                               -0.185934f, -0.269585f};
  std::vector<float> Y_data = {-0.0660155f, 0.0351227f,
                               -0.04236888f, 0.0177365f};
  std::vector<float> Y_h_data = {-0.04236888f, 0.0177365f};
  std::vector<float> Y_c_data = {0.1624992f, 0.04672481f};

  LstmOpContext2x1x2x2 context(direction, activations);
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data,
                  nullptr, use_bias, use_peepholes);

  batch_size = 3;

  // updated from ONNXRuntime test so that it's not the same 2 values repeated 6 times each which potentially hides issues
  X_data = {-0.455351f, -0.476391f,
            -0.555351f, -0.376391f,
            -0.655351f, -0.276391f,
            -0.185934f, -0.869585f,
            -0.285934f, -0.769585f,
            -0.385934f, -0.669585f};

  /* numpy */
  Y_data = {-0.090715f, 0.011908f,
            -0.083193f, 0.037192f,
            -0.073643f, 0.068889f,
            -0.10545f, -0.01573f,
            -0.10621f, -0.0056667f,
            -0.10559f, 0.015734f};

  Y_h_data = {-0.10545f, -0.01573f,
              -0.10621f, -0.0056667f,
              -0.10559f, 0.015734f};

  Y_c_data = {0.21381f, -0.096022f,
              0.23038f, -0.0239f,
              0.24572f, 0.051626f};

  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data,
                  nullptr, use_bias, use_peepholes);
}

// Original comments:
//   test memory reallocation when sequence length increases
//   test correctness for batch size > 1 and
//   memory reallocation due to change in batch size
//   also tests the tricky Y_data write used to avoid copying data
// Most of these aren't relevant anymore as we don't re-use buffers given Compute is stateless.
// It does test a batch > 1 with bidirectional output and custom activations though.
TEST(LSTMTest, ONNXRuntime_TestLSTMOutputWrite) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2;
  int batch_size = 1;
  std::vector<std::string> activations = {"tanh", "sigmoid", "tanh", "tanh", "sigmoid", "tanh"};

  bool use_bias = true;
  bool use_peepholes = false;

  std::vector<float> X_data = {-0.455351f, -0.276391f, -0.185934f, -0.269585f};

  std::vector<float> Y_data = {-0.06601551f, 0.03512269f,
                               -0.05520744f, 0.03879774f,

                               -0.04236888f, 0.01773649f,
                               -0.05332068f, 0.00207076f};
  std::vector<float> Y_h_data = {-0.04236888f, 0.01773649f,
                                 -0.05520744f, 0.03879774f};
  std::vector<float> Y_c_data = {0.1624992f, 0.04672481f,
                                 0.22009919f, 0.08087098f};

  std::string direction = "bidirectional";
  LstmOpContext2x1x2x2 context(direction, activations);
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data,
                  nullptr, use_bias, use_peepholes);

  batch_size = 3;

  X_data = {-0.455351f, -0.776391f,
            -0.355351f, -0.576391f,
            -0.255351f, -0.376391f,

            -0.185934f, -0.169585f,
            -0.285934f, -0.469585f,
            -0.385934f, -0.669585f};

  Y_data = {-0.1269719f, -0.01049645f,
            -0.09596697f, -0.00592083f,
            -0.06777587f, -0.00001902f,

            -0.12206709f, -0.0051103f,
            -0.08422903f, -0.00768428f,
            -0.05224226f, -0.0042149f,

            -0.02778835f, 0.00775075f,
            -0.06541093f, -0.00667958f,
            -0.09953593f, -0.00899231f,

            -0.04350187f, 0.01127771f,
            -0.07949658f, -0.00425178f,
            -0.10883409f, -0.00926061f};

  Y_h_data = {-0.02778835f, 0.00775075f,
              -0.06541093f, -0.00667958f,

              -0.09953593f, -0.00899231f,
              -0.12206709f, -0.0051103f,

              -0.08422903f, -0.00768428f,
              -0.05224226f, -0.0042149f};

  Y_c_data = {0.14675268f, 0.01759163f,
              0.19898503f, -0.01828078f,
              0.24029977f, -0.02784352f,

              0.26577898f, -0.01694398f,
              0.22469461f, -0.02200207f,
              0.18284359f, -0.01078442f};

  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data,
                  nullptr, use_bias, use_peepholes);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMSequenceLengthAllZeros) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2;
  int batch_size = 2;
  std::vector<std::string> activations = {"tanh", "sigmoid", "tanh", "tanh", "sigmoid", "tanh"};

  bool use_bias = true;
  bool use_peepholes = false;

  std::vector<float> X_data = {-0.455351f, -0.776391f,
                               -0.355351f, -0.576391f,

                               -0.185934f, -0.169585f,
                               -0.285934f, -0.469585f};

  std::vector<int> sequence_length = {0, 0};

  std::vector<float> Y_data = {0.0f, 0.0f,
                               0.0f, 0.0f,
                               0.0f, 0.0f,
                               0.0f, 0.0f,

                               0.0f, 0.0f,
                               0.0f, 0.0f,
                               0.0f, 0.0f,
                               0.0f, 0.0f};

  std::vector<float> Y_h_data = {0.0f, 0.0f,
                                 0.0f, 0.0f,

                                 0.0f, 0.0f,
                                 0.0f, 0.0f};

  std::vector<float> Y_c_data = {0.0f, 0.0f,
                                 0.0f, 0.0f,

                                 0.0f, 0.0f,
                                 0.0f, 0.0f};

  std::string direction = "bidirectional";
  LstmOpContext2x1x2x2 context(direction, activations);
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data,
                  &sequence_length, use_bias, use_peepholes);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMSequenceLengthPartialZeros) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2;
  int batch_size = 2;
  std::vector<std::string> activations = {"tanh", "sigmoid", "tanh", "tanh", "sigmoid", "tanh"};

  bool use_bias = true;
  bool use_peepholes = false;

  std::vector<float> X_data = {-0.455351f, -0.776391f,
                               0.0f, 0.0f,

                               -0.185934f, -0.169585f,
                               0.0f, 0.0f};

  std::vector<int> sequence_length = {2, 0};

  std::vector<float> Y_data = {-0.1269719f, -0.01049645f,
                               0.0f, 0.0f,

                               -0.12206709f, -0.0051103f,
                               0.0f, 0.0f,

                               -0.02778835f, 0.00775075f,
                               0.0f, 0.0f,

                               -0.04350187f, 0.01127771f,
                               0.0f, 0.0f};

  std::vector<float> Y_h_data = {-0.02778835f, 0.00775075f,
                                 0.0f, 0.0f,

                                 -0.12206709f, -0.0051103f,
                                 0.0f, 0.0f};

  std::vector<float> Y_c_data = {0.14675268f, 0.01759163f,
                                 0.0f, 0.0f,

                                 0.26577898f, -0.01694398f,
                                 0.0f, 0.0f};

  std::string direction = "bidirectional";
  LstmOpContext2x1x2x2 context(direction, activations);
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data,
                  &sequence_length, use_bias, use_peepholes);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMSequenceLengthShorterThanInputSequenceLength) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2;
  constexpr int batch_size = 1;

  std::vector<float> X_data = {-0.455351f, -0.276391f,
                               -0.185934f, -0.269585f};

  std::vector<int> sequence_length = {1};

  std::vector<float> initial_h = {0.0f, 0.0f,
                                  -0.0306872f, 0.028035f};

  std::vector<float> initial_c = {0.0f, 0.0f,
                                  -0.07243599f, 0.0467052f};

  std::vector<float> Y_data = {-0.0251062f, 0.0561262f,
                               -0.0318928f, 0.0762679f,

                               0.0f, 0.0f,
                               0.0f, 0.0f};

  std::vector<float> Y_h_data = {-0.0251062f, 0.0561262f,
                                 -0.0318928f, 0.0762679f};

  std::string direction = "bidirectional";

  LstmOpContext2x1x2x2 context(direction);
  context.RunTest(X_data, batch_size, seq_len, &initial_h, &initial_c, Y_data, Y_h_data, {}, &sequence_length);
}

TEST(LSTMTest, ONNXRuntime_TestLSTMSequenceLengthShorterThanInputSequenceLengthNoP) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2;
  constexpr int batch_size = 1;

  std::vector<float> X_data = {-0.455351f, -0.276391f,
                               -0.185934f, -0.269585f};

  std::vector<int> sequence_length = {1};

  std::vector<float> initial_h = {0.0f, 0.0f,
                                  -0.0306872f, 0.028035f};

  std::vector<float> initial_c = {0.0f, 0.0f,
                                  -0.07243599f, 0.0467052f};

  std::vector<float> Y_data = {0.0415416f, 0.0196912f,
                               0.0295027f, 0.0334400f,

                               0.0f, 0.0f,
                               0.0f, 0.0f};

  std::vector<float> Y_h_data = {0.0415416f, 0.0196912f,
                                 0.0295027f, 0.0334400f};

  std::string direction = "bidirectional";

  LstmOpContext2x1x2x2 context(direction);
  // CUDA implementation doesn't support peephole
  context.RunTest(X_data, batch_size, seq_len, &initial_h, &initial_c, Y_data, Y_h_data, {}, &sequence_length, false);
}

// Doesn't work with CUDA 11.4 on Windows. Need investigation.
#if defined(USE_CUDA) && defined(_WIN32)
TEST(LSTMTest, DISABLED_ONNXRuntime_TestLSTMShorterSeqInMiddle) {
#else
TEST(LSTMTest, ONNXRuntime_TestLSTMShorterSeqInMiddle) {
#endif

  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2;
  int batch_size = 3;
  std::vector<std::string> activations = {"sigmoid", "tanh", "tanh", "sigmoid", "tanh", "tanh"};

  bool use_bias = true;
  bool use_peepholes = false;

  std::vector<float> X_data = {-0.455351f, -0.776391f,
                               0.0f, 0.0f,
                               0.348763f, 0.678345f,

                               -0.185934f, -0.169585f,
                               0.0f, 0.0f,
                               0.078053f, 0.163457f};

  std::vector<int> sequence_length = {2, 1, 2};

  std::vector<float> Y_data = {0.02907280f, 0.01765226f, -0.06724346f, 0.02957184f, -0.15355367f, 0.04701351f,

                               0.01841230f, 0.04093486f, -0.06724346f, 0.02957184f, -0.17994503f, 0.07397783f,

                               -0.02912546f, 0.04120104f, 0.0f, 0.0f, -0.12768818f, 0.07457943f,

                               -0.04350187f, 0.03531464f, 0.0f, 0.0f, -0.08877515f, 0.03413615f};

  std::vector<float> Y_h_data = {-0.0291254f, 0.04120104f, -0.06724346f, 0.02957184f, -0.12768818f, 0.07457943f,

                                 0.01841230f, 0.04093486f, -0.06724346f, 0.02957184f, -0.17994503f, 0.07397783f};

  std::vector<float> Y_c_data = {-0.06609819f, 0.06838701f, -0.14596788f, 0.04902556f, -0.26768601f, 0.12119407f,

                                 0.04934450f, 0.07126625f, -0.14596788f, 0.04902556f, -0.34139895f, 0.11673255f};

  std::string direction = "bidirectional";
  LstmOpContext2x1x2x2 context(direction, activations);
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data,
                  &sequence_length, use_bias, use_peepholes, 0.0f, false, false);
}

// Doesn't work with CUDA 11.4 on Windows. Need investigation.
#if defined(USE_CUDA) && defined(_WIN32)
TEST(LSTMTest, DISABLED_ONNXRuntime_TestLSTMZeroSeqInMiddle) {
#else
TEST(LSTMTest, ONNXRuntime_TestLSTMZeroSeqInMiddle) {
#endif

  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  constexpr int seq_len = 2;
  int batch_size = 4;
  std::vector<std::string> activations = {"sigmoid", "tanh", "tanh", "sigmoid", "tanh", "tanh"};

  bool use_bias = true;
  bool use_peepholes = false;

  std::vector<float> X_data = {-0.455351f, -0.776391f,
                               0.0f, 0.0f,
                               0.348763f, 0.678345f,
                               0.877836f, 0.543859f,

                               -0.185934f, -0.169585f,
                               0.0f, 0.0f,
                               0.078053f, 0.163457f,
                               0.846098f, 0.987531f};

  std::vector<int> sequence_length = {2, 0, 1, 2};

  std::vector<float> Y_data = {0.02907280f, 0.01765226f, 0.0f, 0.0f, -0.15355367f, 0.04701351f, -0.12951779f, -0.00989562f,
                               0.01841230f, 0.04093486f, 0.0f, 0.0f, -0.15355367f, 0.04701351f, -0.17956293f, 0.01607513f,

                               -0.02912546f, 0.04120104f, 0.0f, 0.0f, 0.0f, 0.0f, -0.22162350f, 0.03132058f,
                               -0.04350187f, 0.03531464f, 0.0f, 0.0f, 0.0f, 0.0f, -0.17885581f, 0.01959856f};

  std::vector<float> Y_h_data = {-0.02912546f, 0.04120104f, 0.0f, 0.0f, -0.15355367f, 0.04701351f, -0.22162350f, 0.03132058f,

                                 0.01841230f, 0.04093486f, 0.0f, 0.0f, -0.15355367f, 0.04701351f, -0.17956293f, 0.01607513f};

  std::vector<float> Y_c_data = {-0.06609819f, 0.06838701f, 0.0f, 0.0f, -0.2894889f, 0.07438067f, -0.39655977f, 0.05050645f,

                                 0.04934450f, 0.07126625f, 0.0f, 0.0f, -0.28948891f, 0.07438067f, -0.34931409f, 0.02799958f};

  std::string direction = "bidirectional";
  LstmOpContext2x1x2x2 context(direction, activations);
  context.RunTest(X_data, batch_size, seq_len, nullptr, nullptr, Y_data, Y_h_data, Y_c_data,
                  &sequence_length, use_bias, use_peepholes, 0.0f, false, false);
}

#ifndef ENABLE_TRAINING
// Prepacking is disabled in full training build so no need to test the feature in a training build.
TEST(LSTMTest, SharedPrepackedWeights) {
  int64_t seq_length = 2;
  int batch_size = 2;
  int64_t input_size = 1;
  int64_t hidden_size = 3;
  int num_directions = 1;

  std::vector<float> X_data{1.f, 2.f, 10.f, 11.f};

  std::vector<float> W_data{
      0.1f, 0.2f, 0.3f, 0.4f,
      1.f, 2.f, 3.f, 4.f,
      10.f, 11.f, 12.f, 13.f};

  std::vector<float> R_data(num_directions * 4 * hidden_size * hidden_size, 0.1f);

  std::vector<float> Y_data{
      0.28828835f, 0.36581863f, 0.45679406f,
      0.34526032f, 0.47220859f, 0.55850911f,

      0.84196719f, 0.89402526f, 0.91073048f,
      0.85882828f, 0.90703777f, 0.92382453f};

  OpTester test("LSTM");

  std::vector<std::string> activations = {"sigmoid", "tanh", "tanh"};

  test.AddAttribute<std::vector<string>>("activations", activations);

  test.AddAttribute("direction", "forward");
  test.AddAttribute("hidden_size", hidden_size);
  test.AddAttribute<int64_t>("input_forget", false);
  test.AddAttribute<float>("clip", 9999.f);

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<int64_t> W_dims = {num_directions, 4 * hidden_size, input_size};
  std::vector<int64_t> R_dims = {num_directions, 4 * hidden_size, hidden_size};

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("W", W_dims, W_data, true);  // Trigger pre-packing
  test.AddInput<float>("R", R_dims, R_data, true);  // Trigger pre-packing

  // B data
  test.AddOptionalInputEdge<float>();

  // sequence
  test.AddOptionalInputEdge<int>();

  // initial_h
  test.AddOptionalInputEdge<float>();

  // initial_c
  test.AddOptionalInputEdge<float>();

  // P_data
  test.AddOptionalInputEdge<float>();

  std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y", Y_dims, Y_data);

  // Y_h
  test.AddOptionalOutputEdge<float>();

  // Y_c
  test.AddOptionalOutputEdge<float>();

  // W
  OrtValue W;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape(W_dims),
                       W_data.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), W);

  // R
  OrtValue R;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(), TensorShape(R_dims),
                       R_data.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator), R);

  SessionOptions so;

  // Set up weight(s) as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("W", &W), Status::OK());
  ASSERT_EQ(so.AddInitializer("R", &R), Status::OK());

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  test.EnableSharingOfPrePackedWeightsAcrossSessions();

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  size_t number_of_pre_packed_weights_counter_session_1 = 0;
  size_t number_of_shared_pre_packed_weights_counter = 0;

  // Session 1
  {
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &ep_vec, {}, &number_of_pre_packed_weights_counter_session_1, &number_of_shared_pre_packed_weights_counter);
    // Assert that no pre-packed weights have been shared thus far
    ASSERT_EQ(number_of_shared_pre_packed_weights_counter, static_cast<size_t>(0));
  }

  auto number_of_elements_in_shared_prepacked_buffers_container =
      test.GetNumPrePackedWeightsShared();
  // Assert that the number of elements in the shared container
  // is the same as the number of weights that have been pre-packed
  ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_elements_in_shared_prepacked_buffers_container);

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // that have been pre-packed will be zero in which case we do not continue with the testing
  // of "sharing" of pre-packed weights as there are no pre-packed weights to be shared at all.
  if (number_of_pre_packed_weights_counter_session_1 == 0)
    return;

  // Session 2
  {
    size_t number_of_pre_packed_weights_counter_session_2 = 0;
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr,
             &ep_vec, {}, &number_of_pre_packed_weights_counter_session_2, &number_of_shared_pre_packed_weights_counter);

    // Assert that the same number of weights were pre-packed in both sessions
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_1, number_of_pre_packed_weights_counter_session_2);

    // Assert that the number of pre-packed weights that were shared equals
    // the number of pre-packed weights in the second session
    ASSERT_EQ(number_of_pre_packed_weights_counter_session_2,
              static_cast<size_t>(number_of_shared_pre_packed_weights_counter));
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime

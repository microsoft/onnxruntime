// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if !defined(ORT_MINIMAL_BUILD)

#include <string>
#include <unordered_map>

#include "test/optimizer/qdq_test_utils.h"
#include "test/providers/qnn/qnn_test_utils.h"
#include "test/providers/tester_types.h"

#include "core/graph/onnx_protobuf.h"

#include "gtest/gtest.h"

namespace onnxruntime {
namespace test {

/*
  ONNX LSTM inputs:
  in[0]: X [seq_length, batch_size, input_size]
  in[1]: W [num_directions, 4*hidden_size, input_size]
  in[2]: R [num_directions, 4*hidden_size, hidden_size]

  ONNX LSTM optional inputs:
  in[3]: B [num_directions, 8*hidden_size]
  in[4]:
  in[5]: initial_h [num_directions, batch_size, hidden_size].
  in[6]: initial_c [num_directions, batch_size, hidden_size].
  in[7]: P [num_directions, 3*hidde_size]

  ONNX LSTM Parameters:
  - activation_alpha ---> Not supported by QNN.
  - activation_beta  ---> Not supported by QNN.
  - activations      ---> Not supported by QNN.
  - clip             ---> Not supported by QNN since the clip in ONNX applied to iofc while QNN only apply to c. Refer
                          https://github.com/microsoft/onnxruntime/blob/v1.21.0/onnxruntime/core/providers/cpu/rnn/uni_directional_lstm.cc
  - direction
  - hidden_size
  - input_forget     ---> Not supported by QNN
  - layout: The shape format of inputs X, initial_h, initial_c and outputs Y, Y_h, Y_c.
            If 0, the following shapes are expected:
                X.shape = [seq_length, batch_size, input_size],
                Y.shape = [seq_length, num_directions, batch_size, hidden_size],
                initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = [num_directions, batch_size, hidden_size].
            If 1, the following shapes are expected:
                X.shape = [batch_size, seq_length, input_size],
                Y.shape = [batch_size, seq_length, num_directions, hidden_size],
                initial_h.shape = Y_h.shape = initial_c.shape = Y_c.shape = [batch_size, num_directions, hidden_size].

  ONNX LSTM optional outputs:
  out[0]: Y [seq_length, num_directions, batch_size, hidden_size]
  out[1]: Y_h [num_directions, batch_size, hidden_size]
  out[2]: Y_c [num_directions, batch_size, hidden_size]

*/
template <typename InputType>
static GetTestModelFn BuildLSTMTestCase(const TestInputDef<float>& X_def,
                                        const TestInputDef<float>& W_def,
                                        const TestInputDef<float>& R_def,
                                        const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                        const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                        const std::optional<std::reference_wrapper<TestInputDef<float>>> C_def,
                                        const std::optional<std::reference_wrapper<TestInputDef<float>>> P_def,
                                        const std::string direction,
                                        const int64_t hidden_size,
                                        const int64_t layout) {
  ORT_UNUSED_PARAMETER(P_def);
  return [X_def, W_def, R_def, B_def,
          H_def, C_def, P_def,
          direction, hidden_size, layout](ModelTestBuilder& builder) {
    auto convert_input = [](ModelTestBuilder& builder, const TestInputDef<float>& def) {
      if (std::is_same<InputType, MLFloat16>::value) {
        TestInputDef<MLFloat16> Fp16_def = ConvertToFP16InputDef(def);
        return MakeTestInput(builder, Fp16_def);
      }
      return MakeTestInput(builder, def);
    };

    NodeArg* inputX = convert_input(builder, X_def);
    NodeArg* inputW = convert_input(builder, W_def);
    NodeArg* inputR = convert_input(builder, R_def);
    std::vector<NodeArg*> input_args = {inputX, inputW, inputR};

    // optional inputs
    // B
    if (B_def) {
      input_args.push_back(convert_input(builder, B_def->get()));
    } else {
      input_args.push_back(builder.MakeOptionalInput<InputType>());
    }

    // sequence length
    input_args.push_back(builder.MakeOptionalInput<int32_t>());

    // H
    if (H_def) {
      input_args.push_back(convert_input(builder, H_def->get()));
    } else {
      input_args.push_back(builder.MakeOptionalInput<InputType>());
    }

    // C
    if (C_def) {
      input_args.push_back(convert_input(builder, C_def->get()));
    } else {
      input_args.push_back(builder.MakeOptionalInput<InputType>());
    }

    // P
    if (P_def) {
      input_args.push_back(convert_input(builder, P_def->get()));
    } else {
      input_args.push_back(builder.MakeOptionalInput<InputType>());
    }

    auto* lstm_output_Y = builder.MakeOutput();
    auto* lstm_output_Y_h = builder.MakeOutput();
    auto* lstm_output_Y_c = builder.MakeOutput();

    Node& lstm_node = builder.AddNode("LSTM",
                                      input_args,
                                      {lstm_output_Y, lstm_output_Y_h, lstm_output_Y_c});
    lstm_node.AddAttribute("direction", direction);
    lstm_node.AddAttribute("hidden_size", hidden_size);
    lstm_node.AddAttribute("layout", layout);
  };
}

// Creates a graph with a single Q/DQ LSTM operator. Used for testing HTP backend.
template <typename InputQType = uint8_t>
static GetTestQDQModelFn<InputQType> BuildQDQLSTMTestCase(const TestInputDef<float>& X_def,
                                                          const TestInputDef<float>& W_def,
                                                          const TestInputDef<float>& R_def,
                                                          const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                                          const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                                          const std::optional<std::reference_wrapper<TestInputDef<float>>> C_def,
                                                          const std::optional<std::reference_wrapper<TestInputDef<float>>> P_def,
                                                          const std::string direction,
                                                          const int64_t hidden_size,
                                                          const int64_t layout) {
  ORT_UNUSED_PARAMETER(P_def);
  return [X_def, W_def, R_def, B_def,
          H_def, C_def, P_def,
          direction, hidden_size, layout](ModelTestBuilder& builder,
                                          std::vector<QuantParams<InputQType>>& output_qparams) {
    auto AddQDQ = [](ModelTestBuilder& builder, const TestInputDef<float>& def) {
      NodeArg* input = MakeTestInput(builder, def);
      QuantParams<InputQType> qparams = GetTestInputQuantParams<InputQType>(def);
      NodeArg* qdq = AddQDQNodePair<InputQType>(builder, input, qparams.scale, qparams.zero_point);
      return qdq;
    };

    NodeArg* X_qdq = AddQDQ(builder, X_def);
    NodeArg* W_qdq = AddQDQ(builder, W_def);
    NodeArg* R_qdq = AddQDQ(builder, R_def);
    std::vector<NodeArg*> input_args = {X_qdq, W_qdq, R_qdq};

    // optional inputs
    // B
    if (B_def) {
      input_args.push_back(AddQDQ(builder, B_def->get()));
    } else {
      input_args.push_back(builder.MakeOptionalInput<float>());
    }

    // sequence length
    input_args.push_back(builder.MakeOptionalInput<int32_t>());

    // H
    if (H_def) {
      input_args.push_back(AddQDQ(builder, H_def->get()));
    } else {
      input_args.push_back(builder.MakeOptionalInput<float>());
    }

    // C
    if (C_def) {
      input_args.push_back(AddQDQ(builder, C_def->get()));
    } else {
      input_args.push_back(builder.MakeOptionalInput<float>());
    }

    // P
    if (P_def) {
      input_args.push_back(AddQDQ(builder, P_def->get()));
    } else {
      input_args.push_back(builder.MakeOptionalInput<float>());
    }

    // LSTM
    NodeArg* lstm_output_Y = builder.MakeIntermediate();
    NodeArg* lstm_output_Y_h = builder.MakeIntermediate();
    NodeArg* lstm_output_Y_c = builder.MakeIntermediate();
    Node& lstm_node = builder.AddNode("LSTM",
                                      input_args,
                                      {lstm_output_Y, lstm_output_Y_h, lstm_output_Y_c});
    lstm_node.AddAttribute("direction", direction);
    lstm_node.AddAttribute("hidden_size", hidden_size);
    lstm_node.AddAttribute("layout", layout);

    // LSTM output -> Q -> DQ -> final output
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, lstm_output_Y, output_qparams[0].scale,
                                                      output_qparams[0].zero_point);
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, lstm_output_Y_h, output_qparams[1].scale,
                                                      output_qparams[1].zero_point);
    AddQDQNodePairWithOutputAsGraphOutput<InputQType>(builder, lstm_output_Y_c, output_qparams[2].scale,
                                                      output_qparams[2].zero_point);
  };
}

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)
// Runs an LSTM model on the QNN HTP backend. Checks the graph node assignment, and that inference
// outputs for QNN EP and CPU EP match.
template <typename QuantType>
static void RunHtpQDQLSTMOpTest(const TestInputDef<float>& X_def,
                                const TestInputDef<float>& W_def,
                                const TestInputDef<float>& R_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> C_def,
                                const std::optional<std::reference_wrapper<TestInputDef<float>>> P_def,
                                const std::string direction,
                                const int64_t hidden_size,
                                const int64_t layout,
                                ExpectedEPNodeAssignment expected_ep_assignment,
                                int opset = 22,
                                QDQTolerance tolerance = QDQTolerance()) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";
  provider_options["offload_graph_io_quantization"] = "0";

  TestQDQModelAccuracy(BuildLSTMTestCase<float>(X_def, W_def, R_def, B_def, H_def, C_def, P_def, direction, hidden_size, layout),
                       BuildQDQLSTMTestCase<QuantType>(X_def, W_def, R_def, B_def, H_def, C_def, P_def, direction, hidden_size, layout),
                       provider_options,
                       opset,
                       expected_ep_assignment,
                       tolerance);
}

static void RunHtpFp16LSTMOpTest(const TestInputDef<float>& X_def,
                                 const TestInputDef<float>& W_def,
                                 const TestInputDef<float>& R_def,
                                 const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                 const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                 const std::optional<std::reference_wrapper<TestInputDef<float>>> C_def,
                                 const std::optional<std::reference_wrapper<TestInputDef<float>>> P_def,
                                 const std::string direction,
                                 const int64_t hidden_size,
                                 const int64_t layout,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 22,
                                 float tolerance = 0.004f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "htp";

  TestFp16ModelAccuracy(BuildLSTMTestCase<float>(X_def, W_def, R_def, B_def, H_def, C_def, P_def, direction, hidden_size, layout),
                        BuildLSTMTestCase<MLFloat16>(X_def, W_def, R_def, B_def, H_def, C_def, P_def, direction, hidden_size, layout),
                        provider_options,
                        opset,
                        expected_ep_assignment,
                        tolerance);
}

static void RunCpuFP32LSTMOpTest(const TestInputDef<float>& X_def,
                                 const TestInputDef<float>& W_def,
                                 const TestInputDef<float>& R_def,
                                 const std::optional<std::reference_wrapper<TestInputDef<float>>> B_def,
                                 const std::optional<std::reference_wrapper<TestInputDef<float>>> H_def,
                                 const std::optional<std::reference_wrapper<TestInputDef<float>>> C_def,
                                 const std::optional<std::reference_wrapper<TestInputDef<float>>> P_def,
                                 const std::string direction,
                                 const int64_t hidden_size,
                                 const int64_t layout,
                                 ExpectedEPNodeAssignment expected_ep_assignment,
                                 int opset = 22,
                                 float tolerance = 0.004f) {
  ProviderOptions provider_options;
  provider_options["backend_type"] = "cpu";

  RunQnnModelTest(BuildLSTMTestCase<float>(X_def, W_def, R_def, B_def, H_def, C_def, P_def, direction, hidden_size, layout),
                  provider_options,
                  opset,
                  expected_ep_assignment,
                  tolerance);
}

// QNN failed to finalize when P is provided
// TODO: Add P to unit test below once finalize issue is resolved

// HTP QDQ
TEST_F(QnnHTPBackendTests, LSTM_QDQ_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQLSTMOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                               TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                               TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                               std::ref(B_def),                                                                         // B
                               std::ref(H_def),                                                                         // initial_h
                               std::ref(C_def),                                                                         // initial_c
                               std::nullopt,                                                                            // P
                               direction,                                                                               // direction
                               hidden_size,                                                                             // hidden_size
                               0,                                                                                       // layout
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_QDQ_sanity_reverse) {
  std::string direction = "reverse";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQLSTMOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                               TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                               TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                               std::ref(B_def),                                                                         // B
                               std::ref(H_def),                                                                         // initial_h
                               std::ref(C_def),                                                                         // initial_c
                               std::nullopt,                                                                            // P
                               direction,                                                                               // direction
                               hidden_size,                                                                             // hidden_size
                               0,                                                                                       // layout
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_QDQ_sanity_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQLSTMOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                               TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                               TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                               std::ref(B_def),                                                                         // B
                               std::ref(H_def),                                                                         // initial_h
                               std::ref(C_def),                                                                         // initial_c
                               std::nullopt,                                                                            // P
                               direction,                                                                               // direction
                               hidden_size,                                                                             // hidden_size
                               0,                                                                                       // layout
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_QDQ_sanity_bidirectional_wo_B) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQLSTMOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                               TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                               TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                               std::nullopt,                                                                            // B
                               std::ref(H_def),                                                                         // initial_h
                               std::ref(C_def),                                                                         // initial_c
                               std::nullopt,                                                                            // P
                               direction,                                                                               // direction
                               hidden_size,                                                                             // hidden_size
                               0,                                                                                       // layout
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_QDQ_sanity_bidirectional_wo_H) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQLSTMOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                               TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                               TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                               std::ref(B_def),                                                                         // B
                               std::nullopt,                                                                            // initial_h
                               std::ref(C_def),                                                                         // initial_c
                               std::nullopt,                                                                            // P
                               direction,                                                                               // direction
                               hidden_size,                                                                             // hidden_size
                               0,                                                                                       // layout
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_QDQ_sanity_bidirectional_wo_C) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpQDQLSTMOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                               TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                               TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                               std::ref(B_def),                                                                         // B
                               std::ref(H_def),                                                                         // initial_h
                               std::nullopt,                                                                            // initial_c
                               std::nullopt,                                                                            // P
                               direction,                                                                               // direction
                               hidden_size,                                                                             // hidden_size
                               0,                                                                                       // layout
                               ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_QDQ_sanity_bidirectional_all_initializer) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, true, -0.5f, 0.5f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, true, -0.5f, 0.5f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, true, -0.5f, 0.5f);
  RunHtpQDQLSTMOpTest<uint8_t>(TestInputDef<float>({seq_len, batch_size, input_size}, false, -0.5f, 0.5f),             // X
                               TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, true, -0.5f, 0.5f),   // W
                               TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, true, -0.5f, 0.5f),  // R
                               std::ref(B_def),                                                                        // B
                               std::ref(H_def),                                                                        // initial_h
                               std::ref(C_def),                                                                        // initial_c
                               std::nullopt,                                                                           // P
                               direction,                                                                              // direction
                               hidden_size,                                                                            // hidden_size
                               0,                                                                                      // layout
                               ExpectedEPNodeAssignment::All,
                               22,
                               QDQTolerance(0.008f));
}

// HTP Fp16
TEST_F(QnnHTPBackendTests, LSTM_Fp16_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16LSTMOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                       TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                       TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                       std::ref(B_def),                                                                         // B
                       std::ref(H_def),                                                                         // initial_h
                       std::ref(C_def),                                                                         // initial_c
                       std::nullopt,                                                                            // P
                       direction,                                                                               // direction
                       hidden_size,                                                                             // hidden_size
                       0,                                                                                       // layout
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_Fp16_sanity_reverse) {
  std::string direction = "reverse";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16LSTMOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                       TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                       TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                       std::ref(B_def),                                                                         // B
                       std::ref(H_def),                                                                         // initial_h
                       std::ref(C_def),                                                                         // initial_c
                       std::nullopt,                                                                            // P
                       direction,                                                                               // direction
                       hidden_size,                                                                             // hidden_size
                       0,                                                                                       // layout
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_Fp16_sanity_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16LSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::nullopt,                                                                            // P
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      0,                                                                                       // layout
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_Fp16_sanity_bidirectional_wo_B) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16LSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::nullopt,                                                                            // B
      std::ref(H_def),                                                                         // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::nullopt,                                                                            // P
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      0,                                                                                       // layout
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_Fp16_sanity_bidirectional_wo_H) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16LSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::nullopt,                                                                            // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::nullopt,                                                                            // P
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      0,                                                                                       // layout
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_Fp16_sanity_bidirectional_wo_C) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunHtpFp16LSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      std::nullopt,                                                                            // initial_c
      std::nullopt,                                                                            // P
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      0,                                                                                       // layout
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnHTPBackendTests, LSTM_Fp16_sanity_bidirectional_all_initializer) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, true, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, true, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, true, -1.0f, 1.0f);
  RunHtpFp16LSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, true, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, true, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                        // B
      std::ref(H_def),                                                                        // initial_h
      std::ref(C_def),                                                                        // initial_c
      std::nullopt,                                                                           // P
      direction,                                                                              // direction
      hidden_size,                                                                            // hidden_size
      0,                                                                                      // layout
      ExpectedEPNodeAssignment::All);
}

// CPU FP32
TEST_F(QnnCPUBackendTests, LSTM_FP32_sanity_forward) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto P_def = TestInputDef<float>({num_direction, 3 * hidden_size}, false, -1.0f, 1.0f);
  RunCpuFP32LSTMOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                       TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                       TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                       std::ref(B_def),                                                                         // B
                       std::ref(H_def),                                                                         // initial_h
                       std::ref(C_def),                                                                         // initial_c
                       std::ref(P_def),                                                                         // P
                       direction,                                                                               // direction
                       hidden_size,                                                                             // hidden_size
                       0,                                                                                       // layout
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LSTM_FP32_sanity_reverse) {
  std::string direction = "reverse";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto P_def = TestInputDef<float>({num_direction, 3 * hidden_size}, false, -1.0f, 1.0f);
  RunCpuFP32LSTMOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                       TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                       TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                       std::ref(B_def),                                                                         // B
                       std::ref(H_def),                                                                         // initial_h
                       std::ref(C_def),                                                                         // initial_c
                       std::ref(P_def),                                                                         // P
                       direction,                                                                               // direction
                       hidden_size,                                                                             // hidden_size
                       0,                                                                                       // layout
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LSTM_FP32_sanity_bidirectional) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto P_def = TestInputDef<float>({num_direction, 3 * hidden_size}, false, -1.0f, 1.0f);
  RunCpuFP32LSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::ref(P_def),                                                                         // P
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      0,                                                                                       // layout
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LSTM_FP32_sanity_bidirectional_wo_B) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto P_def = TestInputDef<float>({num_direction, 3 * hidden_size}, false, -1.0f, 1.0f);
  RunCpuFP32LSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::nullopt,                                                                            // B
      std::ref(H_def),                                                                         // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::ref(P_def),                                                                         // P
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      0,                                                                                       // layout
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LSTM_FP32_sanity_bidirectional_wo_H) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto P_def = TestInputDef<float>({num_direction, 3 * hidden_size}, false, -1.0f, 1.0f);
  RunCpuFP32LSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::nullopt,                                                                            // initial_h
      std::ref(C_def),                                                                         // initial_c
      std::ref(P_def),                                                                         // P
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      0,                                                                                       // layout
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LSTM_FP32_sanity_bidirectional_wo_C) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto P_def = TestInputDef<float>({num_direction, 3 * hidden_size}, false, -1.0f, 1.0f);
  RunCpuFP32LSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                         // B
      std::ref(H_def),                                                                         // initial_h
      std::nullopt,                                                                            // initial_c
      std::ref(P_def),                                                                         // P
      direction,                                                                               // direction
      hidden_size,                                                                             // hidden_size
      0,                                                                                       // layout
      ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LSTM_FP32_sanity_bidirectional_wo_P) {
  std::string direction = "forward";
  uint32_t num_direction = 1;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, false, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, false, -1.0f, 1.0f);
  RunCpuFP32LSTMOpTest(TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),              // X
                       TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, false, -1.0f, 1.0f),   // W
                       TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, false, -1.0f, 1.0f),  // R
                       std::ref(B_def),                                                                         // B
                       std::ref(H_def),                                                                         // initial_h
                       std::ref(C_def),                                                                         // initial_c
                       std::nullopt,                                                                            // P
                       direction,                                                                               // direction
                       hidden_size,                                                                             // hidden_size
                       0,                                                                                       // layout
                       ExpectedEPNodeAssignment::All);
}

TEST_F(QnnCPUBackendTests, LSTM_FP32_sanity_bidirectional_all_initializer) {
  std::string direction = "bidirectional";
  uint32_t num_direction = 2;
  uint32_t batch_size = 3;
  uint32_t hidden_size = 4;
  uint32_t input_size = 5;
  uint32_t seq_len = 6;
  auto B_def = TestInputDef<float>({num_direction, 8 * hidden_size}, true, -1.0f, 1.0f);
  auto H_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, true, -1.0f, 1.0f);
  auto C_def = TestInputDef<float>({num_direction, batch_size, hidden_size}, true, -1.0f, 1.0f);
  auto P_def = TestInputDef<float>({num_direction, 3 * hidden_size}, true, -1.0f, 1.0f);
  RunCpuFP32LSTMOpTest(
      TestInputDef<float>({seq_len, batch_size, input_size}, false, -1.0f, 1.0f),             // X
      TestInputDef<float>({num_direction, 4 * hidden_size, input_size}, true, -1.0f, 1.0f),   // W
      TestInputDef<float>({num_direction, 4 * hidden_size, hidden_size}, true, -1.0f, 1.0f),  // R
      std::ref(B_def),                                                                        // B
      std::ref(H_def),                                                                        // initial_h
      std::ref(C_def),                                                                        // initial_c
      std::ref(P_def),                                                                        // P
      direction,                                                                              // direction
      hidden_size,                                                                            // hidden_size
      0,                                                                                      // layout
      ExpectedEPNodeAssignment::All);
}

#endif  // defined(__aarch64__) || defined(_M_ARM64) || defined(__linux__)

}  // namespace test
}  // namespace onnxruntime

#endif  // !defined(ORT_MINIMAL_BUILD)

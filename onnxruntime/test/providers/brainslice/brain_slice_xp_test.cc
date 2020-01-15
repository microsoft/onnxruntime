// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <fstream>
#include <random>
#include "core/providers/brainslice/brain_slice_execution_provider.h"
#include "core/providers/brainslice/fpga_handle.h"
#include "gtest/gtest.h"
#include "3rdparty/half.hpp"
#include "core/session/inference_session.h"
#include "test/framework/test_utils.h"
#include "test/test_environment.h"
#include "test/providers/brainslice/brain_slice_tester.h"

namespace onnxruntime {
namespace test {
static void VerifyOutputs(const std::vector<MLValue>& fetches,
                          const std::vector<int64_t>& expected_dims,
                          const std::vector<float>& expected_values) {
  ASSERT_EQ(1, fetches.size());
  auto& rtensor = fetches.front().Get<Tensor>();
  TensorShape expected_shape(expected_dims);
  ASSERT_EQ(expected_shape, rtensor.Shape());
  const std::vector<float> found(rtensor.Data<float>(), rtensor.Data<float>() + expected_values.size());
  for (size_t i = 0; i < found.size(); ++i) {
    ASSERT_NEAR(expected_values[i], found[i], 1e-1);
  }
}

//TODO: refactory this to avoid duplicate code
static void RunModel(InferenceSession& session_object,
                     const RunOptions& run_options,
                     bool is_preallocate_output_vec = false) {
  // prepare inputs
  std::vector<int64_t> dims_mul_x = {2, 1, 2};
  std::vector<float> values_mul_x = {-0.455351f, -0.276391f, -0.185934f, -0.269585f};
  //std::vector<float> values_mul_x = {1.0f, 1.0f, 1.0f, 1.0f};
  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  if (is_preallocate_output_vec) {
    fetches.resize(output_names.size());
    for (auto& elem : fetches) {
      CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), dims_mul_x, values_mul_x, &elem);
    }
  }

  // prepare expected inputs and outputs
  std::vector<int64_t> expected_dims_mul_y = {2, 1, 1, 2};
  std::vector<float> expected_values_mul_y = {-0.03255286f, 0.0774838f, -0.05556786f, 0.0785508f};
  //std::vector<float> expected_values_mul_y = {0.114914894f, 0.114914894f, 0.216131598f, 0.216131598f};

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, expected_dims_mul_y, expected_values_mul_y);
}
static const std::string MODEL_URI = "testdata/gru_1.pb";

TEST(BrainSliceExecutionProviderTest, BasicTest) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/onnx_rnns/instructions.bin", "testdata/firmwares/onnx_rnns/data.bin", "testdata/firmwares/onnx_rnns/schema.bin"};
  auto provider = onnxruntime::make_unique<brainslice::BrainSliceExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  SessionOptions so;

  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{so, &DefaultLoggingManager()};
  auto status = session_object.RegisterExecutionProvider(std::move(provider));
  ASSERT_TRUE(status.IsOK());

  ASSERT_TRUE(session_object.Load(MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  RunModel(session_object, run_options);
}

template <class T>
static void FillRandom(std::vector<T>& val, T min, T max) {
  static std::default_random_engine generator;
  std::uniform_real_distribution<float> distribution(min, max);
  for (size_t i = 0; i < val.size(); ++i) {
    val[i] = T(distribution(generator));
  }
}

void RunGRUCompareTest(int64_t input_size,
  int batch_size,
  int64_t hidden_size,
  int64_t seq_length,
  const std::vector<float>* initial_h_data = nullptr,
  const std::vector<int>* sequence_lengths = nullptr,
  const std::string& direction = "forward",
  bool output_sequence = true,
  bool linear_before_reset = false,
  // copy the following vectors as we may modify them
  std::vector<std::string> activations = { "sigmoid", "tanh" },
  std::vector<float> activation_alphas = {},
  std::vector<float> activation_betas = {}) {
  BrainSliceTestor test("GRU");

  test.AddShapeToTensorData();

  int num_directions = (direction == "bidirectional") ? 2 : 1;

  if (num_directions == 2 && activations.size() == 2) {
    activations.reserve(4);  // need to avoid reallocation when inserting
                             // default to copying the activations so the same are used for forward and backwards
    std::copy(activations.cbegin(), activations.cend(), std::back_inserter(activations));
  }

  test.AddAttribute<std::vector<std::string>>("activations", activations);
  if (!activation_alphas.empty())
    test.AddAttribute<std::vector<float>>("activation_alpha", activation_alphas);
  if (!activation_betas.empty())
    test.AddAttribute<std::vector<float>>("activation_beta", activation_betas);

  test.AddAttribute("direction", direction);
  test.AddAttribute("hidden_size", hidden_size);
  // test.AddAttribute<int64_t>("output_sequence", output_sequence);
  test.AddAttribute<int64_t>("linear_before_reset", linear_before_reset);

  std::vector<int64_t> X_dims = { seq_length, batch_size, input_size };
  std::vector<int64_t> W_dims = { num_directions, 3 * hidden_size, input_size };
  std::vector<int64_t> R_dims = { num_directions, 3 * hidden_size, hidden_size };
  std::vector<int64_t> B_dims = { num_directions, 6 * hidden_size };
  std::vector<int64_t> Y_h_dims = { num_directions, batch_size, hidden_size };

  // create rand inputs
  std::vector<float> X_data(seq_length * batch_size * input_size, 1.0f);
  std::vector<float> W_data(num_directions * 3 * hidden_size * input_size, 1.0f);
  std::vector<float> R_data(num_directions * 3 * hidden_size * hidden_size, 0);
  std::vector<float> B_data(num_directions * 6 * hidden_size, 0);
  std::vector<float> Y_data(seq_length * num_directions * batch_size * hidden_size);
  std::vector<float> Y_h_data(num_directions * batch_size * hidden_size);

  FillRandom<float>(X_data, 0.0f, 1.0f);
  FillRandom<float>(W_data, 0.0f, 1.0f);
  FillRandom<float>(R_data, 0.0f, 1.0f);
  FillRandom<float>(B_data, 0.0f, 1.0f);

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("W", W_dims, W_data, true);
  test.AddInput<float>("R", R_dims, R_data, true);
  test.AddInput<float>("B", B_dims, B_data, true);

  if (sequence_lengths) {
    std::vector<int64_t> sequence_lens_dims{ batch_size };
    test.AddInput<int>("sequence_lens", sequence_lens_dims, *sequence_lengths);
  }

  if (initial_h_data) {
    std::vector<int64_t> initial_h_dims = { num_directions, batch_size, hidden_size };
    test.AddInput<float>("initial_h", initial_h_dims, *initial_h_data);
  }

  if (output_sequence) {
    std::vector<int64_t> Y_dims = { seq_length, num_directions, batch_size, hidden_size };
    test.AddOutput<float>("Y", Y_dims, Y_data);
  }
  else {
    test.AddMissingOptionalOutput<float>();
  }

  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  test.CompareWithCPU(0.005, 0.001);
}

TEST(BrainSliceExecutionProviderTest, GRU_128_128_128) {
  RunGRUCompareTest(128, 1, 128, 128);
}

TEST(BrainSliceExecutionProviderTest, GRU_128_256_256) {
  RunGRUCompareTest(256, 1, 256, 128);
}

TEST(BrainSliceExecutionProviderTest, GRU_128_512_512) {
  RunGRUCompareTest(512, 1, 512, 128);
}

TEST(BrainSliceExecutionProviderTest, GRU_128_128) {
  RunGRUCompareTest(128, 1, 128, 10);
}

TEST(BrainSliceExecutionProviderTest, GRU_256_256) {
  RunGRUCompareTest(256, 1, 256, 10);
}

TEST(BrainSliceExecutionProviderTest, GRU_512_512) {
  RunGRUCompareTest(512, 1, 512, 10);
}

TEST(BrainSliceExecutionProviderTest, GRU_backward_128_128) {
  RunGRUCompareTest(128, 1, 128, 10, nullptr, nullptr, "reverse");
}

TEST(BrainSliceExecutionProviderTest, GRU_backward_256_256) {
  RunGRUCompareTest(256, 1, 256, 10, nullptr, nullptr, "reverse");
}

TEST(BrainSliceExecutionProviderTest, GRU_backward_512_512) {
  RunGRUCompareTest(512, 1, 512, 10, nullptr, nullptr, "reverse");
}

TEST(BrainSliceExecutionProviderTest, GRU_bi_128_128) {
  RunGRUCompareTest(128, 1, 128, 10, nullptr, nullptr, "bidirectional");
}

TEST(BrainSliceExecutionProviderTest, GRU_bi_256_256) {
  RunGRUCompareTest(256, 1, 256, 10, nullptr, nullptr, "bidirectional");
}

TEST(BrainSliceExecutionProviderTest, GRU_bi_512_512) {
  RunGRUCompareTest(512, 1, 512, 10, nullptr, nullptr, "bidirectional");
}

TEST(BrainSliceExecutionProviderTest, LSTMBasicTest) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/onnx_rnns/instructions.bin", "testdata/firmwares/onnx_rnns/data.bin", "testdata/firmwares/onnx_rnns/schema.bin"};
  auto provider = std::make_unique<brainslice::BrainSliceExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  SessionOptions so;
  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{ so, &DefaultLoggingManager() };
  auto status = session_object.RegisterExecutionProvider(std::move(provider));
  ASSERT_TRUE(status.IsOK());

  static const std::string LSTM_MODEL_URI = "testdata/lstm_1.onnx";
  ASSERT_TRUE(session_object.Load(LSTM_MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  std::vector<int64_t> X_dims = { 2, 1, 2 };
  std::vector<float> X = { -0.455351f, -0.276391f, -0.185934f, -0.269585f };

  std::vector<int64_t> Y_dims = { 2, 1, 1, 2 };
  std::vector<float> Y_data = { -0.0254789405f, 0.0554842576f, -0.0339087546f, 0.0577334240f };

  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), X_dims, X, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, Y_dims, Y_data);
}

TEST(BrainSliceExecutionProviderTest, DISABLED_LSTMForwardHiddenState) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/onnx_rnns/instructions.bin", "testdata/firmwares/onnx_rnns/data.bin", "testdata/firmwares/onnx_rnns/schema.bin"};
  auto provider = std::make_unique<brainslice::BrainSliceExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  SessionOptions so;
  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{ so, &DefaultLoggingManager() };
  auto status = session_object.RegisterExecutionProvider(std::move(provider));
  ASSERT_TRUE(status.IsOK());

  static const std::string LSTM_MODEL_URI = "testdata/lstm_2.onnx";
  ASSERT_TRUE(session_object.Load(LSTM_MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  std::vector<int64_t> X_dims = { 2, 1, 2 };
  std::vector<float> X = { -0.455351f, -0.276391f, -0.185934f, -0.269585f };

  std::vector<int64_t> Y_dims = { 2, 1, 1, 2 };
  std::vector<float> Y_data = { 0.01797521f, -0.07104912f, -0.03174796f, -0.0152949f };
  std::vector<float> Y_h_data = { -0.03174796f, -0.0152949f };
  std::vector<float> Y_c_data = { -0.07285583f, -0.02545788f };

  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), X_dims, X, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, Y_dims, Y_data);
}

TEST(BrainSliceExecutionProviderTest, DISABLED_LSTMForwardCellState) {
  fpga::FPGAInfo info = {0, true, "testdata/firmwares/onnx_rnns/instructions.bin", "testdata/firmwares/onnx_rnns/data.bin", "testdata/firmwares/onnx_rnns/schema.bin"};
  auto provider = std::make_unique<brainslice::BrainSliceExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  SessionOptions so;
  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{ so, &DefaultLoggingManager() };
  auto status = session_object.RegisterExecutionProvider(std::move(provider));
  ASSERT_TRUE(status.IsOK());

  static const std::string LSTM_MODEL_URI = "testdata/lstm_3.onnx";
  ASSERT_TRUE(session_object.Load(LSTM_MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";
  std::vector<int64_t> X_dims = { 2, 1, 2 };
  std::vector<float> X = { -0.455351f, -0.276391f, -0.185934f, -0.269585f };

  std::vector<int64_t> Y_dims = { 2, 1, 1, 2 };
  std::vector<float> Y_data = { 0.12797015f, 0.0097284f, 0.02716939f, 0.01842997f };
  std::vector<float> Y_h_data = { 0.02716939f, 0.01842997f };
  std::vector<float> Y_c_data = { 0.06408449f, 0.03139432f };

  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), X_dims, X, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("X", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("Y");
  std::vector<MLValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, Y_dims, Y_data);
}

void RunLSTMCompareTest(int64_t input_size,
                        int batch_size,
                        int64_t hidden_size,
                        int64_t seq_length,
                        const std::string& direction = "forward",
                        const std::vector<float>* initial_h_data = nullptr,
                        const std::vector<float>* initial_c_data = nullptr,
                        const std::vector<int>* sequence_lengths = nullptr,
                        bool output_sequence = true,
                        std::vector<std::string> activations = {"sigmoid", "tanh", "tanh"},
                        std::vector<float> activation_alphas = {},
                        std::vector<float> activation_betas = {}) {
  BrainSliceTestor test("LSTM");

  test.AddShapeToTensorData();

  int num_directions = (direction == "bidirectional") ? 2 : 1;

  if (num_directions == 2 && activations.size() == 3) {
    activations.reserve(6);  // need to avoid reallocation when inserting
                             // default to copying the activations so the same are used for forward and backwards
    std::copy(activations.cbegin(), activations.cend(), std::back_inserter(activations));
  }

  test.AddAttribute<std::vector<std::string>>("activations", activations);
  if (!activation_alphas.empty())
    test.AddAttribute<std::vector<float>>("activation_alpha", activation_alphas);
  if (!activation_betas.empty())
    test.AddAttribute<std::vector<float>>("activation_beta", activation_betas);

  test.AddAttribute("direction", direction);
  test.AddAttribute("hidden_size", hidden_size);

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<int64_t> W_dims = {num_directions, 4 * hidden_size, input_size};
  std::vector<int64_t> R_dims = {num_directions, 4 * hidden_size, hidden_size};
  std::vector<int64_t> B_dims = {num_directions, 8 * hidden_size};
  std::vector<int64_t> Y_h_dims = {num_directions, batch_size, hidden_size};

  // create rand inputs
  std::vector<float> X_data(seq_length * batch_size * input_size, 1.0f);
  std::vector<float> W_data(num_directions * 4 * hidden_size * input_size, 1.0f);
  std::vector<float> R_data(num_directions * 4 * hidden_size * hidden_size, 0);
  std::vector<float> B_data(num_directions * 8 * hidden_size, 0);
  std::vector<float> Y_data(seq_length * num_directions * batch_size * hidden_size);
  std::vector<float> Y_h_data(num_directions * batch_size * hidden_size);

  FillRandom<float>(X_data, 0.0f, 1.0f);
  FillRandom<float>(W_data, 0.0f, 1.0f);
  FillRandom<float>(R_data, 0.0f, 1.0f);
  FillRandom<float>(B_data, 0.0f, 1.0f);

  test.AddInput<float>("X", X_dims, X_data);
  test.AddInput<float>("W", W_dims, W_data, true);
  test.AddInput<float>("R", R_dims, R_data, true);
  test.AddInput<float>("B", B_dims, B_data, true);

  if (sequence_lengths) {
    std::vector<int64_t> sequence_lens_dims{batch_size};
    test.AddInput<int>("sequence_lens", sequence_lens_dims, *sequence_lengths);
  }

  if (initial_h_data) {
    std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
    test.AddInput<float>("initial_h", initial_h_dims, *initial_h_data);
  }

  if (initial_c_data) {
    std::vector<int64_t> initial_c_dims = {num_directions, batch_size, hidden_size};
    test.AddInput<float>("initial_c", initial_c_dims, *initial_c_data);
  }

  if (output_sequence) {
    std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
    test.AddOutput<float>("Y", Y_dims, Y_data);
  } else {
    test.AddMissingOptionalOutput<float>();
  }

  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  test.CompareWithCPU(1e-1, 1e-1);
}

TEST(BrainSliceExecutionProviderTest, LSTM_128_128_128) {
  RunLSTMCompareTest(128, 1, 128, 128);
}

TEST(BrainSliceExecutionProviderTest, DISABLED_LSTM_128_256_256) {
  RunLSTMCompareTest(256, 1, 256, 128);
}

TEST(BrainSliceExecutionProviderTest, LSTM_128_128) {
  RunLSTMCompareTest(128, 1, 128, 10);
}

TEST(BrainSliceExecutionProviderTest, DISABLED_LSTM_256_256) {
  RunLSTMCompareTest(256, 1, 256, 10);
}

TEST(BrainSliceExecutionProviderTest, DISABLED_LSTM_backward_128_128) {
  RunLSTMCompareTest(128, 1, 128, 10, "reverse");
}

TEST(BrainSliceExecutionProviderTest, DISABLED_LSTM_backward_256_256) {
  RunLSTMCompareTest(256, 1, 256, 10, "reverse");
}

TEST(BrainSliceExecutionProviderTest, DISABLED_LSTM_bi_128_128) {
  RunLSTMCompareTest(128, 1, 128, 10, "bidirectional");
}

static std::vector<float> LoadFloats(std::string filename, size_t size) {
  std::vector<float> values(size, 0.0f);

  std::ifstream file(filename, std::ios_base::binary);
  EXPECT_TRUE(file.is_open()) << "File '" << filename << "' is missing or invalid.";
  file.read((char*)values.data(), values.size() * sizeof(float));

  return values;
}

bool CheckDenseNet121Compatibility(const BrainSlice_Parameters& bsParameters);

TEST(BrainSliceExecutionProviderTest, DenseNet121Test) {
  fpga::FPGAInfo info = {0, false, "", "", ""};
  auto provider = std::make_unique<brainslice::BrainSliceExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  if (!CheckDenseNet121Compatibility(provider->GetFPGAHandle().GetParameters()))
    return; // Cannot run unit test due to incompatible BrainSlice image.

  SessionOptions so;
  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{ so, &DefaultLoggingManager() };
  auto status = session_object.RegisterExecutionProvider(std::move(provider));
  ASSERT_TRUE(status.IsOK());

  static const std::string RESNET_MODEL_URI = "testdata/densenet121.onnx";
  ASSERT_TRUE(session_object.Load(RESNET_MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  std::vector<int64_t> input_dims = { 1, 224, 224, 3 };
  std::vector<float> input_data = LoadFloats("testdata/densenet121.in", std::accumulate(begin(input_dims), end(input_dims), 1LL, std::multiplies<>{}));

  std::vector<int64_t> output_dims = { 1, 1, 1, 1024 };
  std::vector<float> output_data = LoadFloats("testdata/densenet121.out", std::accumulate(begin(output_dims), end(output_dims), 1LL, std::multiplies<>{}));

  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), input_dims, input_data, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("InputImage:0", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("densenet121/final_block/global_avg_pool/Mean:0");
  std::vector<MLValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, output_dims, output_data);
}

bool CheckResNet50Compatibility(const BrainSlice_Parameters& bsParameters);

TEST(BrainSliceExecutionProviderTest, ResNet50Test) {
  fpga::FPGAInfo info = {0, false, "", "", ""};
  auto provider = std::make_unique<brainslice::BrainSliceExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  if (!CheckResNet50Compatibility(provider->GetFPGAHandle().GetParameters()))
    return; // Cannot run unit test due to incompatible BrainSlice image.

  SessionOptions so;
  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{ so, &DefaultLoggingManager() };
  auto status = session_object.RegisterExecutionProvider(std::move(provider));
  ASSERT_TRUE(status.IsOK());

  static const std::string RESNET_MODEL_URI = "testdata/resnet50.onnx";
  ASSERT_TRUE(session_object.Load(RESNET_MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  std::vector<int64_t> input_dims = { 1, 224, 224, 3 };
  std::vector<float> input_data = LoadFloats("testdata/resnet50.in", std::accumulate(begin(input_dims), end(input_dims), 1LL, std::multiplies<>{}));

  std::vector<int64_t> output_dims = { 1, 1, 1, 2048 };
  std::vector<float> output_data = LoadFloats("testdata/resnet50.out", std::accumulate(begin(output_dims), end(output_dims), 1LL, std::multiplies<>{}));

  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), input_dims, input_data, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("InputImage:0", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("resnet_v1_50/pool5:0");
  std::vector<MLValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, output_dims, output_data);
}

bool CheckResNet152Compatibility(const BrainSlice_Parameters& bsParameters);

TEST(BrainSliceExecutionProviderTest, ResNet152Test) {
  fpga::FPGAInfo info = {0, false, "", "", ""};
  auto provider = std::make_unique<brainslice::BrainSliceExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  if (!CheckResNet152Compatibility(provider->GetFPGAHandle().GetParameters()))
    return; // Cannot run unit test due to incompatible BrainSlice image.

  SessionOptions so;
  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{ so, &DefaultLoggingManager() };
  auto status = session_object.RegisterExecutionProvider(std::move(provider));
  ASSERT_TRUE(status.IsOK());

  static const std::string RESNET_MODEL_URI = "testdata/resnet152.onnx";
  ASSERT_TRUE(session_object.Load(RESNET_MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  std::vector<int64_t> input_dims = { 1, 224, 224, 3 };
  std::vector<float> input_data = LoadFloats("testdata/resnet152.in", std::accumulate(begin(input_dims), end(input_dims), 1LL, std::multiplies<>{}));

  std::vector<int64_t> output_dims = { 1, 1, 1, 2048 };
  std::vector<float> output_data = LoadFloats("testdata/resnet152.out", std::accumulate(begin(output_dims), end(output_dims), 1LL, std::multiplies<>{}));

  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), input_dims, input_data, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("InputImage:0", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("resnet_v1_152/pool5:0");
  std::vector<MLValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, output_dims, output_data);
}

bool CheckVGG16Compatibility(const BrainSlice_Parameters& bsParameters);

TEST(BrainSliceExecutionProviderTest, VGG16Test) {
  fpga::FPGAInfo info = {0, false, "", "", ""};
  auto provider = std::make_unique<brainslice::BrainSliceExecutionProvider>(info);
  EXPECT_TRUE(provider != nullptr);

  if (!CheckVGG16Compatibility(provider->GetFPGAHandle().GetParameters()))
    return; // Cannot run unit test due to incompatible BrainSlice image.

  SessionOptions so;
  so.session_logid = "InferenceSessionTests.NoTimeout";

  InferenceSession session_object{ so, &DefaultLoggingManager() };
  auto status = session_object.RegisterExecutionProvider(std::move(provider));
  ASSERT_TRUE(status.IsOK());

  static const std::string RESNET_MODEL_URI = "testdata/vgg16.onnx";
  ASSERT_TRUE(session_object.Load(RESNET_MODEL_URI).IsOK());
  ASSERT_TRUE(session_object.Initialize().IsOK());

  RunOptions run_options;
  run_options.run_tag = "one session/one tag";

  std::vector<int64_t> input_dims = { 1, 224, 224, 3 };
  std::vector<float> input_data = LoadFloats("testdata/vgg16.in", std::accumulate(begin(input_dims), end(input_dims), 1LL, std::multiplies<>{}));

  std::vector<int64_t> output_dims = { 1, 1, 1, 4096 };
  std::vector<float> output_data = LoadFloats("testdata/vgg16.out", std::accumulate(begin(output_dims), end(output_dims), 1LL, std::multiplies<>{}));

  MLValue ml_value;
  CreateMLValue<float>(TestCPUExecutionProvider()->GetAllocator(0, OrtMemTypeDefault), input_dims, input_data, &ml_value);
  NameMLValMap feeds;
  feeds.insert(std::make_pair("InputImage:0", ml_value));

  // prepare outputs
  std::vector<std::string> output_names;
  output_names.push_back("vgg_16/fc7/Relu:0");
  std::vector<MLValue> fetches;

  // Now run
  common::Status st = session_object.Run(run_options, feeds, output_names, &fetches);
  if (!st.IsOK()) {
    std::cout << "Run returned status: " << st.ErrorMessage() << std::endl;
  }
  ASSERT_TRUE(st.IsOK());
  VerifyOutputs(fetches, output_dims, output_data);
}

}  // namespace test
}  // namespace onnxruntime

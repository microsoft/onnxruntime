
#include "gtest/gtest.h"

#include <algorithm>
#include <vector>

#include "core/mlas/inc/mlas.h"
#include "core/util/qmath.h"
#include "core/providers/cpu/rnn/deep_cpu_lstm.h"
#include "test/providers/provider_test_utils.h"

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

static std::vector<float> ApplyQDQ(const std::vector<float>& X_data) {
  uint8_t zp = 128;
  float scale = 1.0f;
  GetQuantizationParameter(X_data.data(), X_data.size(), scale, zp);

  std::vector<uint8_t> X_quant(X_data.size());
  MlasQuantizeLinear(X_data.data(), X_quant.data(), X_data.size(), scale, zp);

  std::vector<float> result(X_data.size());
  std::transform(X_quant.begin(), X_quant.end(), result.begin(), [&zp, &scale](uint8_t q) {
    return (static_cast<int32_t>(q) - zp) * scale;
  });

  return result;
}

void QuantizeWeight(std::vector<uint8_t>& w_quant,
                    float& scale,
                    uint8_t& zp,
                    const std::vector<float>& w,
                    size_t num_direction,
                    size_t row,
                    size_t col) {
  std::vector<float> w_transpose(w.size());

  for (size_t dir_idx = 0; dir_idx < num_direction; dir_idx++) {
    const float* w_buffer = w.data() + dir_idx * row * col;
    float* w_transpose_buffer = w_transpose.data() + dir_idx * row * col;
    for (size_t r = 0; r < row; r++) {
      for (size_t c = 0; c < col; c++) {
        *(w_transpose_buffer + r + c * row) = *w_buffer++;
      }
    }
  }

  GetQuantizationParameter(w_transpose.data(), w_transpose.size(), scale, zp);

  w_quant.resize(w.size());
  MlasQuantizeLinear(w_transpose.data(), w_quant.data(), w.size(), scale, zp);
}

static void CalculateRefResult(std::vector<float>& Y_data,
                               std::vector<float>& Y_h_data,
                               std::vector<float>& Y_c_data,
                               int64_t input_size,
                               int64_t batch_size,
                               int64_t hidden_size,
                               const std::vector<float>& X_data,
                               const std::vector<float>& W_data,
                               const std::vector<float>& R_data,
                               const std::vector<float>* B_data = nullptr,
                               const std::vector<float>* P_data = nullptr,
                               const std::vector<float>* initial_h_data = nullptr,
                               const std::vector<float>* initial_c_data = nullptr,
                               const std::vector<int>* sequence_lengths = nullptr,
                               const std::string& direction = "forward",
                               float clip = 9999.f,
                               bool input_forget = false,
                               // copy the following vectors as we may modify them
                               std::vector<std::string> activations = {},
                               std::vector<float> activation_alphas = {},
                               std::vector<float> activation_betas = {},
                               bool hasClip = true) {
  int64_t seq_length = 1;  // only use seq length 1

  OpTester test("LSTM", 7 /*opset_version*/, onnxruntime::kOnnxDomain /*domain*/, false /*verify_output*/);

  int num_directions = (direction == "bidirectional") ? 2 : 1;

  test.AddAttribute<std::vector<std::string>>("activations", activations);
  if (!activation_alphas.empty())
    test.AddAttribute<std::vector<float>>("activation_alpha", activation_alphas);
  if (!activation_betas.empty())
    test.AddAttribute<std::vector<float>>("activation_beta", activation_betas);

  test.AddAttribute("direction", direction);
  test.AddAttribute("hidden_size", hidden_size);
  test.AddAttribute<int64_t>("input_forget", input_forget);
  if (hasClip) {
    test.AddAttribute<float>("clip", clip);
  }

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<int64_t> W_dims = {num_directions, 4 * hidden_size, input_size};
  std::vector<int64_t> R_dims = {num_directions, 4 * hidden_size, hidden_size};

  test.AddInput<float>("X", X_dims, ApplyQDQ(X_data));
  test.AddInput<float>("W", W_dims, ApplyQDQ(W_data));
  test.AddInput<float>("R", R_dims, ApplyQDQ(R_data));
  // test.AddInput<float>("W", W_dims, W_data);
  // test.AddInput<float>("R", R_dims, R_data);

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

  size_t y_data_size = seq_length * num_directions * batch_size * hidden_size;
  Y_data.resize(seq_length * num_directions * batch_size * hidden_size);
  std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y", Y_dims, Y_data);

  size_t y_h_data_size = num_directions * batch_size * hidden_size;
  Y_h_data.resize(num_directions * batch_size * hidden_size);
  std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  size_t y_c_data_size = num_directions * batch_size * hidden_size;
  Y_c_data.resize(num_directions * batch_size * hidden_size);
  std::vector<int64_t> Y_c_dims{num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y_c", Y_c_dims, Y_c_data);

  // TensorRT failed on LSTM tests
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});

  std::vector<MLValue> outputs = test.GetFetches();

  const float* y_buffer = outputs[0].Get<Tensor>().Data<float>();
  std::copy(y_buffer, y_buffer + y_data_size, Y_data.begin());

  const float* y_h_buffer = outputs[1].Get<Tensor>().Data<float>();
  std::copy(y_h_buffer, y_h_buffer + y_h_data_size, Y_h_data.begin());

  const float* y_c_buffer = outputs[2].Get<Tensor>().Data<float>();
  std::copy(y_c_buffer, y_c_buffer + y_c_data_size, Y_c_data.begin());
}

static void RunQuantLSTM(int64_t input_size,
                         int64_t batch_size,
                         int64_t hidden_size,
                         const std::vector<float>& X_data,
                         const std::vector<float>& W_data,
                         const std::vector<float>& R_data,
                         const std::vector<float>* B_data = nullptr,
                         const std::vector<float>* P_data = nullptr,
                         const std::vector<float>* initial_h_data = nullptr,
                         const std::vector<float>* initial_c_data = nullptr,
                         const std::vector<int>* sequence_lengths = nullptr,
                         const std::string& direction = "forward",
                         float clip = 9999.f,
                         bool input_forget = false,
                         // copy the following vectors as we may modify them
                         std::vector<std::string> activations = {},
                         std::vector<float> activation_alphas = {},
                         std::vector<float> activation_betas = {},
                         bool hasClip = true) {
  int64_t seq_length = 1;  // only use seq length 1

  OpTester test("DynamicQuantizeLSTM", 1 /*opset_version*/, onnxruntime::kMSDomain /*domain*/);

  int num_directions = (direction == "bidirectional") ? 2 : 1;

  if (activations.empty()) {
      activations = { "sigmoid", "tanh", "tanh" };
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
  test.AddAttribute<int64_t>("input_forget", input_forget);
  if (hasClip) {
    test.AddAttribute<float>("clip", clip);
  }

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<int64_t> W_dims = {num_directions, input_size, 4 * hidden_size};
  std::vector<int64_t> R_dims = {num_directions, hidden_size, 4 * hidden_size};

  test.AddInput<float>("X", X_dims, X_data);

  float w_scale = 1.f;
  uint8_t w_zp = 128;
  std::vector<uint8_t> w_quant;
  QuantizeWeight(w_quant, w_scale, w_zp, W_data, num_directions, 4 * hidden_size, input_size);
  test.AddInput<uint8_t>("W", W_dims, w_quant);

  float r_scale = 1.f;
  uint8_t r_zp = 128;
  std::vector<uint8_t> r_quant;
  QuantizeWeight(r_quant, r_scale, r_zp, R_data, num_directions, 4 * hidden_size, hidden_size);
  test.AddInput<uint8_t>("R", R_dims, r_quant);

  if (B_data) {
    std::vector<int64_t> B_dims = {num_directions, 8 * hidden_size};
    test.AddInput<float>("B", B_dims, *B_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  test.AddMissingOptionalInput<int>();

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

  test.AddInput<float>("W_scale", {1}, {w_scale});
  test.AddInput<uint8_t>("W_zero_point", {1}, {w_zp});

  test.AddInput<float>("R_scale", {1}, {r_scale});
  test.AddInput<uint8_t>("R_zero_point", {1}, {r_zp});

  std::vector<float> Y_data;
  std::vector<float> Y_h_data;
  std::vector<float> Y_c_data;
  CalculateRefResult(Y_data,
                     Y_h_data,
                     Y_c_data,
                     input_size,
                     batch_size,
                     hidden_size,
                     X_data,
                     W_data,
                     R_data,
                     B_data,
                     P_data,
                     initial_h_data,
                     initial_c_data,
                     sequence_lengths,
                     direction,
                     clip,
                     input_forget,
                     activations,
                     activation_alphas,
                     activation_betas,
                     hasClip);

  std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y", Y_dims, Y_data);

  std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  std::vector<int64_t> Y_c_dims{num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y_c", Y_c_dims, Y_c_data);

  test.Run();
}

TEST(DynamicQuantLSTMTest, ForwardSimpleWeightsNoBiasTwoRows) {
  int batch_size = 2;
  int64_t input_size = 1;
  int64_t hidden_size = 2;

  int num_directions = 1;

  std::vector<float> X_data{1.f, 2.f};

  std::vector<float> W_data{
      0.1f, 0.2f, 0.3f, 0.4f,
      1.f, 2.f, 3.f, 4.f};

  std::vector<float> R_data(num_directions * 4 * hidden_size * hidden_size, 0.1f);

  RunQuantLSTM(input_size, batch_size, hidden_size, X_data, W_data, R_data);
}

}  // namespace test
}  // namespace onnxruntime

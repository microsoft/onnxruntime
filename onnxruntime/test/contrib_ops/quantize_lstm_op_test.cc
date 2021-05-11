
#include "gtest/gtest.h"

#include <algorithm>
#include <vector>

#include "core/providers/cpu/rnn/deep_cpu_lstm.h"
#include "core/util/qmath.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/util/include/default_providers.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

template <typename QType,
          typename std::enable_if<is_quant_type<QType>::value, int>::type = 0>
static std::vector<float> ApplyQDQ(const std::vector<float>& data, size_t channel_count, bool per_channel = false) {
  std::vector<float> result(data.size());
  size_t size_per_dir = data.size() / channel_count;

  for (size_t dir_idx = 0; dir_idx < channel_count; dir_idx++) {
    QType zp = 0;
    float scale = 1.0f;
    const float* data_buf = data.data() + size_per_dir * dir_idx;
    if (per_channel) {
      GetQuantizationParameter<QType, true, true>(data_buf, size_per_dir, scale, zp, nullptr);
    } else {
      GetQuantizationParameter<QType, true, false>(data_buf, size_per_dir, scale, zp, nullptr);
    }

    std::vector<QType> quant_data(size_per_dir);
    MlasQuantizeLinear(data_buf, quant_data.data(), size_per_dir, scale, zp);

    std::transform(quant_data.begin(),
                   quant_data.end(),
                   result.begin() + size_per_dir * dir_idx,
                   [&zp, &scale](QType q) {
                     return (static_cast<int32_t>(q) - zp) * scale;
                   });
  }

  return result;
}

template <typename QType,
          typename std::enable_if<is_quant_type<QType>::value, int>::type = 0>
void QuantizeWeight(std::vector<QType>& w_quant,
                    std::vector<float>& scale,
                    std::vector<QType>& zp,
                    const std::vector<float>& w,
                    size_t num_direction,
                    size_t row,
                    size_t col,
                    bool per_channel) {
  std::vector<QType> w_quant_tmp(w.size());

  size_t quant_param_size = per_channel ? num_direction * row : num_direction;
  size_t quant_span = per_channel ? col : row * col;
  scale.resize(quant_param_size);
  zp.resize(quant_param_size);

  for (size_t quant_param_idx = 0; quant_param_idx < quant_param_size; quant_param_idx++) {
    if (per_channel) {
      GetQuantizationParameter<QType, true, true>(w.data() + quant_param_idx * quant_span, quant_span, scale[quant_param_idx], zp[quant_param_idx], nullptr);
    } else {
      GetQuantizationParameter<QType, true, false>(w.data() + quant_param_idx * quant_span, quant_span, scale[quant_param_idx], zp[quant_param_idx], nullptr);
    }

    MlasQuantizeLinear(w.data() + quant_param_idx * quant_span,
                       w_quant_tmp.data() + quant_param_idx * quant_span,
                       quant_span,
                       scale[quant_param_idx],
                       zp[quant_param_idx]);
  }

  w_quant.resize(w.size());
  for (size_t dir_idx = 0; dir_idx < num_direction; dir_idx++) {
    QType* w_quant_tmp_buf = w_quant_tmp.data() + dir_idx * row * col;
    QType* w_quant_buf = w_quant.data() + dir_idx * row * col;
    for (size_t c = 0; c < col; c++) {
      for (size_t r = 0; r < row; r++) {
        *w_quant_buf++ = *(w_quant_tmp_buf + r * col + c);
      }
    }
  }

  // transpose row and col
}

template <typename QType,
          typename std::enable_if<is_quant_type<QType>::value, int>::type = 0>
static void ComputeRefOutput(std::vector<float>& Y_data,
                             std::vector<float>& Y_h_data,
                             std::vector<float>& Y_c_data,
                             int64_t input_size,
                             int64_t batch_size,
                             int64_t hidden_size,
                             const std::vector<float>& X_data,
                             const std::vector<float>& W_data,
                             const std::vector<float>& R_data,
                             const std::vector<float>* B_data,
                             const std::vector<float>* P_data,
                             const std::vector<float> initial_h_data,
                             const std::vector<float> initial_c_data,
                             const std::string& direction,
                             const std::vector<std::string>& activations,
                             bool per_channel) {
  OpTester test("LSTM", 7 /*opset_version*/, onnxruntime::kOnnxDomain /*domain*/, false /*verify_output*/);

  test.AddAttribute<std::vector<std::string>>("activations", activations);
  test.AddAttribute("direction", direction);
  test.AddAttribute("hidden_size", hidden_size);
  test.AddAttribute<int64_t>("input_forget", 0);

  int64_t seq_length = 1;  // only use seq length 1
  int64_t num_directions = (direction == "bidirectional") ? 2 : 1;
  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<int64_t> W_dims = {num_directions, 4 * hidden_size, input_size};
  std::vector<int64_t> R_dims = {num_directions, 4 * hidden_size, hidden_size};

  test.AddInput<float>("X", X_dims, ApplyQDQ<uint8_t>(X_data, 1));
  test.AddInput<float>("W", W_dims, ApplyQDQ<QType>(W_data, per_channel ? num_directions * 4 * hidden_size : num_directions, per_channel));
  test.AddInput<float>("R", R_dims, ApplyQDQ<QType>(R_data, per_channel ? num_directions * 4 * hidden_size : num_directions, per_channel));

  if (B_data) {
    std::vector<int64_t> B_dims = {num_directions, 8 * hidden_size};
    test.AddInput<float>("B", B_dims, *B_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  // sequence_lens
  test.AddMissingOptionalInput<int>();

  std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
  test.AddInput<float>("initial_h", initial_h_dims, ApplyQDQ<uint8_t>(initial_h_data, num_directions));

  std::vector<int64_t> initial_c_dims = {num_directions, batch_size, hidden_size};
  test.AddInput<float>("initial_c", initial_c_dims, initial_c_data);

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

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCpuExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);

  std::vector<MLValue> outputs = test.GetFetches();

  const float* y_buffer = outputs[0].Get<Tensor>().Data<float>();
  std::copy(y_buffer, y_buffer + y_data_size, Y_data.begin());

  const float* y_h_buffer = outputs[1].Get<Tensor>().Data<float>();
  std::copy(y_h_buffer, y_h_buffer + y_h_data_size, Y_h_data.begin());

  const float* y_c_buffer = outputs[2].Get<Tensor>().Data<float>();
  std::copy(y_c_buffer, y_c_buffer + y_c_data_size, Y_c_data.begin());
}

template <typename QType,
          typename std::enable_if<std::is_same<QType, uint8_t>::value || std::is_same<QType, int8_t>::value, int>::type = 0>
static void RunQuantLSTM(int64_t input_size,
                         int64_t batch_size,
                         int64_t hidden_size,
                         bool has_bias,
                         bool has_P,
                         bool is_initializer_W,
                         bool is_initializer_R,
                         bool per_channel,
                         const std::string& direction) {
  OpTester test("DynamicQuantizeLSTM", 1 /*opset_version*/, onnxruntime::kMSDomain /*domain*/);

  int num_directions = (direction == "bidirectional") ? 2 : 1;

  std::vector<std::string> activations;
  if (num_directions == 2) {
    activations = {"sigmoid", "tanh", "tanh", "sigmoid", "tanh", "tanh"};
  } else {
    activations = {"sigmoid", "tanh", "tanh"};
  }
  test.AddAttribute<std::vector<std::string>>("activations", activations);

  test.AddAttribute("direction", direction);
  test.AddAttribute("hidden_size", hidden_size);
  test.AddAttribute<int64_t>("input_forget", 0);

  RandomValueGenerator rand_gen;

  // X
  int64_t seq_len = 1;  // only use seq length 1 to model the test
  std::vector<int64_t> X_dims = {seq_len, batch_size, input_size};
  std::vector<float> X_data = rand_gen.Gaussian<float>({seq_len, batch_size, input_size}, 0.0f, 0.25f);
  test.AddInput<float>("X", X_dims, X_data);

  // W
  std::vector<int64_t> W_dims = {num_directions, input_size, 4 * hidden_size};
  std::vector<float> W_data = rand_gen.Gaussian<float>({num_directions, 4 * hidden_size, input_size}, 0.0f, 0.25f);

  std::vector<float> w_scale;
  std::vector<QType> w_zp;
  std::vector<QType> w_quant;
  QuantizeWeight(w_quant, w_scale, w_zp, W_data, num_directions, 4 * hidden_size, input_size, per_channel);
  test.AddInput<QType>("W", W_dims, w_quant, is_initializer_W);

  // R
  std::vector<int64_t> R_dims = {num_directions, hidden_size, 4 * hidden_size};
  std::vector<float> R_data = rand_gen.Gaussian<float>({num_directions, 4 * hidden_size, hidden_size}, 0.0f, 0.25f);

  std::vector<float> r_scale;
  std::vector<QType> r_zp;
  std::vector<QType> r_quant;
  QuantizeWeight(r_quant, r_scale, r_zp, R_data, num_directions, 4 * hidden_size, hidden_size, per_channel);
  test.AddInput<QType>("R", R_dims, r_quant, is_initializer_R);

  std::vector<float> B_data;
  if (has_bias) {
    std::vector<int64_t> B_dims = {num_directions, 8 * hidden_size};
    B_data = rand_gen.Gaussian<float>(B_dims, 0.0f, 0.25f);

    test.AddInput<float>("B", B_dims, B_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  // sequence_lens
  test.AddMissingOptionalInput<int>();

  // initial_h
  std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_h_data = rand_gen.Gaussian<float>(initial_h_dims, 0.0f, 0.25f);
  test.AddInput<float>("initial_h", initial_h_dims, initial_h_data);

  // initial_c
  std::vector<int64_t> initial_c_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_c_data = rand_gen.Gaussian<float>(initial_c_dims, 0.0f, 0.25f);
  test.AddInput<float>("initial_c", initial_c_dims, initial_c_data);

  std::vector<float> P_data;
  if (has_P) {
    std::vector<int64_t> P_dims = {num_directions, 3 * hidden_size};
    P_data = rand_gen.Gaussian<float>(P_dims, 0.0f, 0.25f);
    test.AddInput<float>("P", P_dims, P_data);
  } else {
    test.AddMissingOptionalInput<float>();
  }

  std::vector<int64_t> per_tensor_dims = {num_directions};
  std::vector<int64_t> per_channel_dims = {num_directions, 4 * hidden_size};
  test.AddInput<float>("W_scale", per_channel ? per_channel_dims : per_tensor_dims, w_scale);
  test.AddInput<QType>("W_zero_point", per_channel ? per_channel_dims : per_tensor_dims, w_zp);

  test.AddInput<float>("R_scale", per_channel ? per_channel_dims : per_tensor_dims, r_scale);
  test.AddInput<QType>("R_zero_point", per_channel ? per_channel_dims : per_tensor_dims, r_zp);

  std::vector<float> Y_data;
  std::vector<float> Y_h_data;
  std::vector<float> Y_c_data;
  ComputeRefOutput<QType>(Y_data, Y_h_data, Y_c_data,
                          input_size, batch_size, hidden_size,
                          X_data, W_data, R_data,
                          has_bias ? &B_data : nullptr,
                          has_P ? &P_data : nullptr,
                          initial_h_data, initial_c_data,
                          direction, activations, per_channel);

  std::vector<int64_t> Y_dims = {seq_len, num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y", Y_dims, Y_data);

  std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  std::vector<int64_t> Y_c_dims{num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y_c", Y_c_dims, Y_c_data);

  test.Run();
}

template <typename QType,
          typename std::enable_if<std::is_same<QType, uint8_t>::value || std::is_same<QType, int8_t>::value, int>::type = 0>
static void RunQuantLSTM(int64_t input_size,
                         int64_t batch_size,
                         int64_t hidden_size,
                         bool per_channel = false) {
  // bias + P: 0, prepacking: 0, bidirectional: 0
  RunQuantLSTM<QType>(input_size, batch_size, hidden_size,
                      false /*has_bias*/, false /*has_P*/,
                      false /*is_initializer_W*/, false /*is_initializer_R*/,
                      per_channel, "forward");

  // bias + P: 0, prepacking: 0, bidirectional: 1
  RunQuantLSTM<QType>(input_size, batch_size, hidden_size,
                      false /*has_bias*/, false /*has_P*/,
                      false /*is_initializer_W*/, false /*is_initializer_R*/,
                      per_channel, "bidirectional");

  // bias + P: 0, prepacking: 1, bidirectional: 0
  RunQuantLSTM<QType>(input_size, batch_size, hidden_size,
                      false /*has_bias*/, false /*has_P*/,
                      true /*is_initializer_W*/, true /*is_initializer_R*/,
                      per_channel, "forward");

  // bias + P: 0, prepacking: 1, bidirectional: 1
  RunQuantLSTM<QType>(input_size, batch_size, hidden_size,
                      false /*has_bias*/, false /*has_P*/,
                      true /*is_initializer_W*/, true /*is_initializer_R*/,
                      per_channel, "bidirectional");

  // bias + P: 1, prepacking: 0, bidirectional: 0
  RunQuantLSTM<QType>(input_size, batch_size, hidden_size,
                      true /*has_bias*/, true /*has_P*/,
                      false /*is_initializer_W*/, false /*is_initializer_R*/,
                      per_channel, "forward");

  // bias + P: 1, prepacking: 0, bidirectional: 1
  RunQuantLSTM<QType>(input_size, batch_size, hidden_size,
                      true /*has_bias*/, true /*has_P*/,
                      false /*is_initializer_W*/, false /*is_initializer_R*/,
                      per_channel, "bidirectional");

  // bias + P: 1, prepacking: 1, bidirectional: 0
  RunQuantLSTM<QType>(input_size, batch_size, hidden_size,
                      true /*has_bias*/, true /*has_P*/,
                      true /*is_initializer_W*/, true /*is_initializer_R*/,
                      per_channel, "forward");

  // bias + P: 1, prepacking: 1, bidirectional: 1
  RunQuantLSTM<QType>(input_size, batch_size, hidden_size,
                      true /*has_bias*/, true /*has_P*/,
                      true /*is_initializer_W*/, true /*is_initializer_R*/,
                      per_channel, "bidirectional");
}

TEST(DynamicQuantLSTMTest, SmallSize) {
  RunQuantLSTM<int8_t>(2, 1, 16);
  RunQuantLSTM<int8_t>(2, 1, 16, true /*per_channel*/);
  RunQuantLSTM<uint8_t>(2, 1, 16);
}

TEST(DynamicQuantLSTMTest, LargeSize) {
  RunQuantLSTM<int8_t>(12, 3, 278);
  RunQuantLSTM<int8_t>(12, 3, 278, true /*per_channel*/);
  RunQuantLSTM<uint8_t>(12, 3, 278);
}

#ifndef ENABLE_TRAINING  // Prepacking is enabled only on non-training builds
TEST(DynamicQuantLSTMTest, SharedPrepackedWeights) {
  OpTester test("DynamicQuantizeLSTM", 1 /*opset_version*/, onnxruntime::kMSDomain /*domain*/);

  int num_directions = 1;
  int input_size = 2;
  int batch_size = 1;
  int hidden_size = 16;

  std::vector<std::string> activations;
  activations = {"sigmoid", "tanh", "tanh"};
  test.AddAttribute<std::vector<std::string>>("activations", activations);

  test.AddAttribute("direction", "forward");
  test.AddAttribute("hidden_size", static_cast<int64_t>(hidden_size));
  test.AddAttribute<int64_t>("input_forget", 0);

  RandomValueGenerator rand_gen;

  // X
  int64_t seq_len = 1;  // only use seq length 1 to model the test
  std::vector<int64_t> X_dims = {seq_len, batch_size, input_size};
  std::vector<float> X_data = rand_gen.Gaussian<float>({seq_len, batch_size, input_size}, 0.0f, 0.25f);
  test.AddInput<float>("X", X_dims, X_data);

  // W
  std::vector<int64_t> W_dims = {num_directions, input_size, 4 * hidden_size};
  std::vector<float> W_data = rand_gen.Gaussian<float>({num_directions, 4 * hidden_size, input_size}, 0.0f, 0.25f);

  std::vector<float> w_scale;
  std::vector<int8_t> w_zp;
  std::vector<int8_t> w_quant;
  QuantizeWeight(w_quant, w_scale, w_zp, W_data, num_directions, 4 * hidden_size, input_size, false);
  test.AddInput<int8_t>("W", W_dims, w_quant, true);  // Trigger pre-packing

  // R
  std::vector<int64_t> R_dims = {num_directions, hidden_size, 4 * hidden_size};
  std::vector<float> R_data = rand_gen.Gaussian<float>({num_directions, 4 * hidden_size, hidden_size}, 0.0f, 0.25f);

  std::vector<float> r_scale;
  std::vector<int8_t> r_zp;
  std::vector<int8_t> r_quant;
  QuantizeWeight(r_quant, r_scale, r_zp, R_data, num_directions, 4 * hidden_size, hidden_size, false);
  test.AddInput<int8_t>("R", R_dims, r_quant, true);  // Trigger pre-packing

  // B
  test.AddMissingOptionalInput<float>();

  // sequence_lens
  test.AddMissingOptionalInput<int>();

  // initial_h
  std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_h_data = rand_gen.Gaussian<float>(initial_h_dims, 0.0f, 0.25f);
  test.AddInput<float>("initial_h", initial_h_dims, initial_h_data);

  // initial_c
  std::vector<int64_t> initial_c_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_c_data = rand_gen.Gaussian<float>(initial_c_dims, 0.0f, 0.25f);
  test.AddInput<float>("initial_c", initial_c_dims, initial_c_data);

  test.AddMissingOptionalInput<float>();

  std::vector<int64_t> per_tensor_dims = {num_directions};
  test.AddInput<float>("W_scale", per_tensor_dims, w_scale);
  test.AddInput<int8_t>("W_zero_point", per_tensor_dims, w_zp);

  test.AddInput<float>("R_scale", per_tensor_dims, r_scale);
  test.AddInput<int8_t>("R_zero_point", per_tensor_dims, r_zp);

  std::vector<float> Y_data;
  std::vector<float> Y_h_data;
  std::vector<float> Y_c_data;
  ComputeRefOutput<int8_t>(Y_data, Y_h_data, Y_c_data,
                           input_size, batch_size, hidden_size,
                           X_data, W_data, R_data,
                           nullptr,
                           nullptr,
                           initial_h_data, initial_c_data,
                           "forward", activations, false);

  std::vector<int64_t> Y_dims = {seq_len, num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y", Y_dims, Y_data);

  std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  std::vector<int64_t> Y_c_dims{num_directions, batch_size, hidden_size};
  test.AddOutput<float>("Y_c", Y_c_dims, Y_c_data);

  auto W_quant_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<int8_t>(), TensorShape(W_dims),
                                                 w_quant.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator));
  OrtValue W;

  W.Init(W_quant_tensor.release(), DataTypeImpl::GetType<Tensor>(),
         DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());

  auto R_quant_tensor = std::make_unique<Tensor>(DataTypeImpl::GetType<int8_t>(), TensorShape(R_dims),
                                                 r_quant.data(), OrtMemoryInfo(CPU, OrtAllocatorType::OrtDeviceAllocator));
  OrtValue R;

  R.Init(R_quant_tensor.release(), DataTypeImpl::GetType<Tensor>(),
         DataTypeImpl::GetType<Tensor>()->GetDeleteFunc());
  SessionOptions so;

  // Set up weight(s) as a shared initializer to be shared between sessions
  ASSERT_EQ(so.AddInitializer("W", &W), Status::OK());
  ASSERT_EQ(so.AddInitializer("R", &R), Status::OK());

  // We want all sessions running using this OpTester to be able to share pre-packed weights if applicable
  test.AddPrePackedSharedContainerToSessions();

  size_t used_cached_pre_packed_weights_counter = 0;

  // Pre-packing is limited just to the CPU EP for now and we will only test the CPU EP
  // and we want to ensure that it is available in this build
  auto cpu_ep = []() -> std::vector<std::unique_ptr<IExecutionProvider>> {
    std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
    execution_providers.push_back(DefaultCpuExecutionProvider());
    return execution_providers;
  };

  // Session 1
  {
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {},
             nullptr, &ep_vec, {}, &used_cached_pre_packed_weights_counter);
    ASSERT_EQ(used_cached_pre_packed_weights_counter, static_cast<size_t>(0));  // No pre-packed weights have been shared thus far
  }

  // On some platforms/architectures MLAS may choose to not do any pre-packing and the number of elements
  // in the shared container will be zero in which case this test will be a no-op
  auto number_of_elements_in_shared_prepacked_buffers_container =
      test.GetNumberOfElementsInPrePackedSharedContainer();

  // Session 2
  {
    auto ep_vec = cpu_ep();
    test.Run(so, OpTester::ExpectResult::kExpectSuccess, "", {},
             nullptr, &ep_vec, {}, &used_cached_pre_packed_weights_counter);
    ASSERT_EQ(used_cached_pre_packed_weights_counter, static_cast<size_t>(number_of_elements_in_shared_prepacked_buffers_container));
  }
}
#endif

}  // namespace test
}  // namespace onnxruntime

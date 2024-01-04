// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/cpu/rnn/rnn.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"
using namespace std;
namespace onnxruntime {
namespace test {

// test input data is generated from CNTK with shape of [batch_size, seq_length, input_size]
// onnxruntime takes input of shape [seq_length, batch_size, input_size)
template <typename T>
void MemoryLayoutTransposeRNNInputCNTKToONNXRuntime(const T* X_data_cntk, T* X_data_onnx, int64_t seq_length, int64_t batch_size, int64_t input_size) {
  for (int seq = 0; seq < seq_length; seq++) {
    for (int batch = 0; batch < batch_size; batch++) {
      for (int feature = 0; feature < input_size; feature++) {
        X_data_onnx[(seq * batch_size + batch) * input_size + feature] =
            X_data_cntk[(batch * seq_length + seq) * input_size + feature];
      }
    }
  }
}

// test output data is generated from CNTK with shape of [batch_size, seq_length, num_directions, hidden_size].
// onnxruntime takes output of shape [seq_length, num_directions, batch_size, hidden_size)
template <typename T>
void MemoryLayoutTransposeRNNOutputCNTKToONNXRuntime(const T* X_data_cntk, T* X_data_onnx,
                                                     int64_t seq_length, int64_t num_directions, int64_t batch_size, int64_t hidden_size) {
  for (int seq = 0; seq < seq_length; seq++) {
    for (int dir = 0; dir < num_directions; dir++) {
      for (int batch = 0; batch < batch_size; batch++) {
        for (int feature = 0; feature < hidden_size; feature++) {
          X_data_onnx[((seq * num_directions + dir) * batch_size + batch) * hidden_size + feature] =
              // [batch, sequence, direction, feature]
              X_data_cntk[((batch * seq_length + seq) * num_directions + dir) * hidden_size + feature];
        }
      }
    }
  }
}

TEST(RNNTest, RNN_bidirectional_bias_initial_zigged_batch) {
  OpTester test("RNN");
  int64_t num_directions = 2, input_size = 2, hidden_size = 3, seq_length = 5;

  test.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
  test.AddAttribute("direction", "bidirectional");
  test.AddAttribute("hidden_size", hidden_size);

  int batch_size = 2;

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<float> X_data_in_batchs{1.64644051F, 2.14556813F,
                                      1.80829012F, 1.63464952F,
                                      1.27096438F, 1.93768239F,
                                      1.31276166F, 2.67531896F,
                                      2.89098835F, 1.15032458F,

                                      // batch 2
                                      1.30798471F, 0.0777787F,
                                      1.64898741F, 1.30596721F,
                                      1.26110339F, 0.99100447F,
                                      //
                                      0.0F, 0.0F,
                                      0.0F, 0.0F};
  std::vector<float> X_data(seq_length * batch_size * input_size);
  MemoryLayoutTransposeRNNInputCNTKToONNXRuntime(&X_data_in_batchs[0], &X_data[0], seq_length, batch_size, input_size);
  test.AddInput<float>("X", X_dims, X_data);

  std::vector<int64_t> W_dims = {num_directions, hidden_size, input_size};
  std::vector<float> W_data({0.4317745F, 0.37378395F, -1.0386457F, -0.22681296F, 0.4418987F, 0.49973935F,
                             0.47248289F, -0.63369429F, 0.89542073F, 0.69698066F, 0.65118814F, 1.0828459F});
  test.AddInput<float>("W", W_dims, W_data);

  std::vector<int64_t> R_dims = {num_directions, hidden_size, hidden_size};
  std::vector<float> R_data({-0.24072374F, -0.29326528F, -0.91741192F,
                             0.5447638F, 0.53938544F, 0.79502326F,
                             -0.59813821F, 0.020413321F, -0.52225035F,

                             -0.4292987F, -0.14766316F, -0.91084105F,
                             0.23699039F, 0.064034894F, 0.089069292F,
                             -0.12803128F, -0.081178986F, 0.967533F});
  test.AddInput<float>("R", R_dims, R_data);

  std::vector<int64_t> B_dims = {num_directions, 2 * hidden_size};
  std::vector<float> B_data({-0.44529742F, 0.80094892F, -1.0028138F, 0.0F, 0.0F, 0.0F,
                             0.57412368F, 0.13440208F, -0.85748988F, 0.0F, 0.0F, 0.0F});
  test.AddInput<float>("B", B_dims, B_data);

  std::vector<int64_t> sequence_lens_dims({batch_size});
  std::vector<int> sequence_lens_data{5, 3};
  test.AddInput<int>("sequence_lens", sequence_lens_dims, sequence_lens_data);

  std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_h_data({1.2F, 1.2F, 1.2F, 1.2F, 1.2F, 1.2F, 1.2F, 1.2F, 1.2F, 1.2F, 1.2F, 1.2F});
  test.AddInput<float>("initial_h", initial_h_dims, initial_h_data);

  std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
  std::vector<float> Y_data_in_batchs({-0.58767605F, 0.69586837F, -0.48001164F, -0.71697658F, 0.99646497F, 0.9980582F,
                                       0.8678354F, -0.94409049F, 0.8424542F, -0.37213817F, 0.99391747F, 0.99555576F,
                                       0.12221602F, -0.31430662F, -0.42285997F, -0.62726945F, 0.988343F, 0.9956606F,
                                       0.91737539F, -0.92293501F, 0.78396499F, -0.87013513F, 0.99671143F, 0.9990834F,
                                       0.51060104F, -0.95055139F, -0.12672578F, -0.51847482F, 0.99931973F, 0.99655205F,
                                       // batch 2
                                       -0.92063117F, 0.93283325F, -0.93614483F, 0.1513377F, 0.90150106F, 0.74947751F,
                                       0.91569924F, -0.96036619F, 0.89311725F, -0.13006453F, 0.98576784F, 0.98875856F,
                                       -0.28076148F, -0.04275616F, -0.75480938F, -0.84641802F, 0.98438591F, 0.96007115F,
                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  std::vector<float> Y_data(seq_length * num_directions * batch_size * hidden_size);
  MemoryLayoutTransposeRNNOutputCNTKToONNXRuntime(&Y_data_in_batchs[0], &Y_data[0],
                                                  seq_length, num_directions, batch_size, hidden_size);
  test.AddOutput<float>("Y", Y_dims, Y_data);

  std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
  std::vector<float> Y_h_data({0.51060104F, -0.95055139F, -0.12672578F, -0.28076148F, -0.04275616F, -0.75480938F,
                               -0.71697658F, 0.99646497F, 0.9980582F, 0.1513377F, 0.90150106F, 0.74947751F});
  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  // TensorRT failed on RNN tests
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kTensorrtExecutionProvider});
}

// Doesn't work with CUDA 11.4 on Windows. Need investigation.
#if defined(USE_CUDA) && defined(_WIN32)
TEST(RNNTest, DISABLED_RNN_bidirectional_zigged_batch) {
#else
TEST(RNNTest, RNN_bidirectional_zigged_batch) {
#endif
  OpTester test("RNN");
  int64_t num_directions = 2, input_size = 2, hidden_size = 3, seq_length = 5;

  test.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
  test.AddAttribute("direction", "bidirectional");
  test.AddAttribute("hidden_size", hidden_size);

  int batch_size = 2;

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<float> X_data_in_batchs{1.64644051F, 2.14556813F,
                                      1.80829012F, 1.63464952F,
                                      1.27096438F, 1.93768239F,
                                      1.31276166F, 2.67531896F,
                                      2.89098835F, 1.15032458F,
                                      // batch 2
                                      1.30798471F, 0.0777787F,
                                      1.64898741F, 1.30596721F,
                                      1.26110339F, 0.99100447F,
                                      0.0F, 0.0F,
                                      0.0F, 0.0F};
  std::vector<float> X_data(seq_length * batch_size * input_size);
  MemoryLayoutTransposeRNNInputCNTKToONNXRuntime(&X_data_in_batchs[0], &X_data[0], seq_length, batch_size, input_size);
  test.AddInput<float>("X", X_dims, X_data);

  std::vector<int64_t> W_dims = {num_directions, hidden_size, input_size};
  std::vector<float> W_data({-0.68526405F, 0.3758406F, 0.13007233F, 0.6596455F, -0.68564546F, 0.22745803F,
                             0.37704858F, -0.075543992F, -0.92860377F, -0.014112951F, -1.0042796F, 0.83100969F});
  test.AddInput<float>("W", W_dims, W_data);

  std::vector<int64_t> R_dims = {num_directions, hidden_size, hidden_size};
  std::vector<float> R_data({0.22057047F, -0.25696567F, 0.93817306F,
                             -0.1917963F, -0.41374302F, -0.76374459F,
                             -0.96291065F, 0.098433927F, 0.049011F,
                             0.56542879F, 0.50024462F, 0.33647421F,
                             -0.80293375F, 0.59855759F, -0.74431759F,
                             -0.003538545F, -0.73175585F, 0.65632182F});
  test.AddInput<float>("R", R_dims, R_data);

  std::vector<int64_t> B_dims = {num_directions, 2 * hidden_size};
  std::vector<float> B_data({0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  test.AddInput<float>("B", B_dims, B_data);

  std::vector<int64_t> sequence_lens_dims({batch_size});
  std::vector<int> sequence_lens_data{5, 3};
  test.AddInput<int>("sequence_lens", sequence_lens_dims, sequence_lens_data);

  std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_h_data({0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  test.AddInput<float>("initial_h", initial_h_dims, initial_h_data);

  std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
  std::vector<float> Y_data_in_batchs({-0.31118321F, 0.92598617F, -0.56547648F, 0.39222997F, -0.99489242F, 0.86467457F,
                                       -0.8980186F, 0.89000309F, -0.46600604F, 0.38946036F, -0.99521333F, 0.69356728F,
                                       -0.76437593F, 0.92218089F, 0.46116444F, 0.06449185F, -0.97850645F, 0.90903103F,
                                       0.13221112F, 0.87366635F, 0.50636965F, -0.09428534F, -0.94113714F, 0.76040554F,
                                       -0.85353446F, 0.34633741F, -0.93988168F, 0.76291096F, -0.99102205F, -0.96011895F,
                                       // batch 2
                                       -0.69988877F, 0.21788915F, -0.70597935F, 0.0274523F, -0.9431532F, -0.60166585F,
                                       -0.90726709F, 0.93011433F, -0.17109135F, 0.18146965F, -0.96685904F, -0.23413686F,
                                       -0.79737622F, 0.62769204F, 0.30727068F, 0.38049027F, -0.82903779F, -0.41610005F,
                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                                       0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  std::vector<float> Y_data(seq_length * num_directions * batch_size * hidden_size);
  MemoryLayoutTransposeRNNOutputCNTKToONNXRuntime(&Y_data_in_batchs[0], &Y_data[0],
                                                  seq_length, num_directions, batch_size, hidden_size);
  test.AddOutput<float>("Y", Y_dims, Y_data);

  std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
  std::vector<float> Y_h_data({-0.85353446F, 0.34633741F, -0.93988168F, -0.79737622F, 0.62769204F, 0.30727068F,
                               0.39222997F, -0.99489242F, 0.86467457F, 0.0274523F, -0.9431532F, -0.60166585F});
  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

TEST(RNNTest, RNN_reverse_direction_zigged_batch) {
  OpTester test("RNN");
  int64_t num_directions = 1, input_size = 2, hidden_size = 3, seq_length = 5;

  test.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
  test.AddAttribute("direction", "reverse");
  test.AddAttribute("hidden_size", hidden_size);

  int batch_size = 2;

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<float> X_data_in_batchs{0.54881352F, 0.71518934F,
                                      0.60276335F, 0.54488319F,
                                      0.42365479F, 0.64589411F,
                                      0.4375872F, 0.89177299F,
                                      0.96366274F, 0.38344151F,
                                      // batch 2
                                      0.417021990F, 0.720324516F,
                                      0.0001143748F, 0.302332580F,
                                      0.146755889F, 0.0923385918F,
                                      0.0F, 0.0F,
                                      0.0F, 0.0F};
  std::vector<float> X_data(seq_length * batch_size * input_size);
  MemoryLayoutTransposeRNNInputCNTKToONNXRuntime(&X_data_in_batchs[0], &X_data[0], seq_length, batch_size, input_size);
  test.AddInput<float>("X", X_dims, X_data);

  std::vector<int64_t> W_dims = {num_directions, hidden_size, input_size};
  std::vector<float> W_data({0.60482931F, 0.67304987F, 0.13166776F, -0.33417314F, 0.66345924F, -0.49411628F});
  test.AddInput<float>("W", W_dims, W_data);

  std::vector<int64_t> R_dims = {num_directions, hidden_size, hidden_size};
  std::vector<float> R_data({0.50877059F, 0.78382635F, 0.665046F,
                             0.89860243F, -0.71745688F, 0.80142093F,
                             -0.76517141F, -0.88981366F, -0.48568386F});
  test.AddInput<float>("R", R_dims, R_data);

  std::vector<int64_t> B_dims = {num_directions, 2 * hidden_size};
  std::vector<float> B_data({0.0F, 0.0F, 0.0F,
                             0.0F, 0.0F, 0.0F});
  test.AddInput<float>("B", B_dims, B_data);

  std::vector<int64_t> sequence_lens_dims({batch_size});
  std::vector<int> sequence_lens_data{5, 3};
  test.AddInput<int>("sequence_lens", sequence_lens_dims, sequence_lens_data);

  std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_h_data({0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  test.AddInput<float>("initial_h", initial_h_dims, initial_h_data);

  std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
  std::vector<float> Y_data_in_batchs({0.87014002F, 0.09402763F, -0.54269236F,
                                       0.37661949F, 0.28492415F, 0.15850827F,
                                       0.8218801F, -0.33996487F, -0.7320742F,
                                       0.90398145F, 0.61396617F, -0.70602065F,
                                       0.68629962F, -0.00125255F, 0.4218055F,
                                       // batch 2
                                       0.64809889F, -0.19472955F, -0.24271242F,
                                       0.29596764F, 0.08308408F, -0.27175695F,
                                       0.14977546F, -0.01153355F, 0.05169443F,
                                       0.0F, 0.0F, 0.0F,
                                       0.0F, 0.0F, 0.0F});
  std::vector<float> Y_data(seq_length * num_directions * batch_size * hidden_size);
  MemoryLayoutTransposeRNNOutputCNTKToONNXRuntime(&Y_data_in_batchs[0], &Y_data[0],
                                                  seq_length, num_directions, batch_size, hidden_size);
  test.AddOutput<float>("Y", Y_dims, Y_data);

  std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
  std::vector<float> Y_h_data({0.87014002F, 0.09402763F, -0.54269236F, 0.64809889F, -0.19472955F, -0.24271242F});
  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kTensorrtExecutionProvider});
}

// Doesn't work with CUDA 11.4 on Windows. Need investigation.
#if defined(USE_CUDA) && defined(_WIN32)
TEST(RNNTest, DISABLED_RNN_forward_direction_zigged_batch) {
#else
TEST(RNNTest, RNN_forward_direction_zigged_batch) {
#endif
  OpTester test("RNN");
  int64_t num_directions = 1, input_size = 2, hidden_size = 3, seq_length = 5;

  test.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
  test.AddAttribute("direction", "forward");
  test.AddAttribute("hidden_size", hidden_size);

  int batch_size = 2;

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<float> X_data_in_batchs{0.061169811F, 0.26296741F,
                                      0.80939841F, 0.080034949F,
                                      0.21000224F, 0.65772671F,
                                      0.20081005F, 0.95461535F,
                                      0.93818879F, 0.76034665F,
                                      // batch 2
                                      0.34715694F, 0.0032335778F,
                                      0.72840774F, 0.20933059F,
                                      0.01131162F, 0.15063381F,
                                      0.0F, 0.0F,
                                      0.0F, 0.0F};
  std::vector<float> X_data(seq_length * batch_size * input_size);
  MemoryLayoutTransposeRNNInputCNTKToONNXRuntime(&X_data_in_batchs[0], &X_data[0], seq_length, batch_size, input_size);

  test.AddInput<float>("X", X_dims, X_data);

  std::vector<int64_t> W_dims = {num_directions, hidden_size, input_size};
  std::vector<float> W_data({-0.49937296F, -0.082866333F, 0.40978807F, -0.33496389F, -0.40066367F, -0.72275674F});
  test.AddInput<float>("W", W_dims, W_data);

  std::vector<int64_t> R_dims = {num_directions, hidden_size, hidden_size};
  std::vector<float> R_data({0.16146433F, -0.36291042F, 0.61149812F,
                             -0.018460333F, -0.19345543F, 0.35175204F,
                             0.84270394F, 0.94917566F, -0.76469761F});
  test.AddInput<float>("R", R_dims, R_data);

  std::vector<int64_t> B_dims = {num_directions, 2 * hidden_size};
  std::vector<float> B_data({0.0F, 0.0F, 0.0F,
                             0.0F, 0.0F, 0.0F});
  test.AddInput<float>("B", B_dims, B_data);

  std::vector<int64_t> sequence_lens_dims({batch_size});
  std::vector<int> sequence_lens_data{5, 3};
  test.AddInput<int>("sequence_lens", sequence_lens_dims, sequence_lens_data);

  std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_h_data({0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  test.AddInput<float>("initial_h", initial_h_dims, initial_h_data);

  std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
  std::vector<float> Y_data_in_batchs({-0.0522899628F, -0.0629346371F, -0.211336553F,
                                       -0.482055902F, 0.238964200F, -0.313421130F,
                                       -0.474286675F, -0.274602413F, -0.461531579F,
                                       -0.412429035F, -0.325635254F, -0.792385221F,
                                       -0.746264696F, -0.0781838298F, -0.751394153F,
                                       // batch 2
                                       -0.171904743F, 0.140247226F, -0.140494764F,
                                       -0.497260034F, 0.153767705F, -0.334113181F,
                                       -0.343922496F, -0.181868196F, -0.130254388F,
                                       0.0F, 0.0F, 0.0F,
                                       0.0F, 0.0F, 0.0F});
  std::vector<float> Y_data(seq_length * num_directions * batch_size * hidden_size);
  MemoryLayoutTransposeRNNOutputCNTKToONNXRuntime(&Y_data_in_batchs[0], &Y_data[0],
                                                  seq_length, num_directions, batch_size, hidden_size);
  test.AddOutput<float>("Y", Y_dims, Y_data);

  std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
  std::vector<float> Y_h_data({-0.746264696F, -0.0781838298F, -0.751394153F, -0.343922496F, -0.181868196F, -0.130254388F});
  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Doesn't work with CUDA 11.4 on Windows. Need investigation.
#if defined(USE_CUDA) && defined(_WIN32)
TEST(RNNTest, DISABLED_RNN_bidirectional_0) {
#else
TEST(RNNTest, RNN_bidirectional_0) {
#endif
  OpTester test("RNN");
  int64_t num_directions = 2, input_size = 2, hidden_size = 3, batch_size = 1, seq_length = 5;

  test.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
  test.AddAttribute("direction", "bidirectional");
  test.AddAttribute("hidden_size", hidden_size);

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<float> X_data({0.54881352F, 0.71518934F,
                             0.60276335F, 0.54488319F,
                             0.42365479F, 0.64589411F,
                             0.4375872F, 0.891773F,
                             0.96366274F, 0.38344151F});

  test.AddInput<float>("X", X_dims, X_data);

  std::vector<int64_t> W_dims = {num_directions, hidden_size, input_size};
  std::vector<float> W_data({-0.74535543F, 0.21360011F, 1.0782362F, 0.092641734F, -1.0087538F, -0.97021431F,
                             0.88425213F, 0.93182313F, 0.767329F, -0.541361F, 0.6218195F, -0.7977342F});

  test.AddInput<float>("W", W_dims, W_data);

  std::vector<int64_t> R_dims = {num_directions, hidden_size, hidden_size};
  std::vector<float> R_data({// forward
                             -0.7322467F, -0.95795155F, -0.058495734F,
                             -0.7271859F, -0.29820377F, -0.85114992F,
                             -0.097570196F, 0.82271612F, 0.1396943F,
                             // reverse
                             0.11753198F, -0.30726218F, 0.47448817F,
                             -0.60847247F, 0.11959127F, -0.15468557F,
                             0.18048254F, -0.27739462F, 0.40944993F});
  test.AddInput<float>("R", R_dims, R_data);

  std::vector<int64_t> B_dims = {num_directions, 2 * hidden_size};
  std::vector<float> B_data({0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F,
                             0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  test.AddInput<float>("B", B_dims, B_data);

  std::vector<int64_t> sequence_lens_dims({batch_size});
  std::vector<int> sequence_lens_data(batch_size, (int)seq_length);
  test.AddInput<int>("sequence_lens", sequence_lens_dims, sequence_lens_data);

  std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_h_data({0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F});
  test.AddInput<float>("initial_h", initial_h_dims, initial_h_data);

  std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
  std::vector<float> Y_data({-0.25082839F, 0.57703555F, -0.84758246F, 0.89708149F, -0.50691134F, 0.10560472F,
                             -0.57328993F, 0.89210528F, -0.63864726F, 0.85242939F, -0.35763535F, 0.20078957F,
                             -0.51920897F, 0.83700335F, -0.33934233F, 0.80431187F, -0.51605088F, -0.060805645F,
                             -0.49105126F, 0.74924558F, -0.54746729F, 0.86223149F, -0.56618357F, -0.29732516F,
                             -0.74539614F, 0.93210655F, -0.63887376F, 0.83650553F, 0.48680621F, 0.28520593F});
  test.AddOutput<float>("Y", Y_dims, Y_data);

  std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
  std::vector<float> Y_h_data({-0.74539614F, 0.93210655F, -0.63887376F, 0.89708149F, -0.50691134F, 0.10560472F});
  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

// Doesn't work with CUDA 11.4 on Windows. Need investigation.
#if defined(USE_CUDA) && defined(_WIN32)
TEST(RNNTest, DISABLED_RNN_bidirectional_1) {
#else
TEST(RNNTest, RNN_bidirectional_1) {
#endif
  OpTester test("RNN");
  int64_t num_directions = 2, input_size = 2, hidden_size = 2, batch_size = 1, seq_length = 1;

  test.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
  test.AddAttribute("direction", "bidirectional");
  test.AddAttribute("hidden_size", hidden_size);

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<float> X_data({1.0F, 1.0F});

  test.AddInput<float>("X", X_dims, X_data);

  std::vector<int64_t> W_dims = {num_directions, hidden_size, input_size};
  std::vector<float> W_data({1.0F, 1.0F, 1.0F, 1.0F,
                             1.0F, 1.0F, 1.0F, 1.0F});

  test.AddInput<float>("W", W_dims, W_data);

  std::vector<int64_t> R_dims = {num_directions, hidden_size, hidden_size};
  std::vector<float> R_data({// forward
                             1.0F, 1.0F,
                             1.0F, 1.0F,
                             // reverse
                             1.0F, 1.0F,
                             1.0F, 1.0F});
  test.AddInput<float>("R", R_dims, R_data);

  std::vector<int64_t> B_dims = {num_directions, 2 * hidden_size};
  std::vector<float> B_data({0.0F, 0.0F, 0.0F, 0.0F,
                             0.0F, 0.0F, 0.0F, 0.0F});
  test.AddInput<float>("B", B_dims, B_data);

  std::vector<int64_t> sequence_lens_dims({batch_size});
  std::vector<int> sequence_lens_data(batch_size, (int)seq_length);
  test.AddInput<int>("sequence_lens", sequence_lens_dims, sequence_lens_data);

  std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
  std::vector<float> initial_h_data({0.1F, 0.2F, 0.3F, 0.4F});
  test.AddInput<float>("initial_h", initial_h_dims, initial_h_data);

  std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
  std::vector<float> Y_data({0.98009639F, 0.98009639F, 0.99100745F, 0.99100745F});
  test.AddOutput<float>("Y", Y_dims, Y_data);

  std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
  std::vector<float> Y_h_data({0.98009639F, 0.98009639F, 0.99100745F, 0.99100745F});
  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
}

typedef enum {
  RNNOutputY,
  RNNOutputY_h,
  RNNOutputBoth
} RNNOutputOption;

TEST(RNNTest, DISABLED_RNN_default_attributes_and_forward_direction) {
  int64_t num_directions = 1, input_size = 2, hidden_size = 3, batch_size = 1, seq_length = 5;

  // In case of useDefault, attributes, inputs or outputs are not set.
  // Otherwise they are set (with values may or may not be the same as ONNX default values).
  auto run_test = [&](OpTester& test, bool useDefault, RNNOutputOption outputOption) {
    std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
    std::vector<float> X_data({0.061169811F, 0.26296741F,
                               0.80939841F, 0.080034949F,
                               0.21000224F, 0.65772671F,
                               0.20081005F, 0.95461535F,
                               0.93818879F, 0.76034665F});

    test.AddInput<float>("X", X_dims, X_data);

    std::vector<int64_t> W_dims = {num_directions, hidden_size, input_size};
    std::vector<float> W_data({-0.49937296F, -0.082866333F, 0.40978807F, -0.33496389F, -0.40066367F, -0.72275674F});
    test.AddInput<float>("W", W_dims, W_data);

    std::vector<int64_t> R_dims = {num_directions, hidden_size, hidden_size};
    std::vector<float> R_data({0.16146433F, -0.36291042F, 0.61149812F,
                               -0.018460333F, -0.19345543F, 0.35175204F,
                               0.84270394F, 0.94917566F, -0.76469761F});
    test.AddInput<float>("R", R_dims, R_data);

    if (useDefault) {
      std::vector<int64_t> B_dims = {num_directions, 2 * hidden_size};
      std::vector<float> B_data({0.0F, 0.0F, 0.0F,
                                 0.0F, 0.0F, 0.0F});
      test.AddInput<float>("B", B_dims, B_data);

      std::vector<int64_t> sequence_lens_dims({batch_size});
      std::vector<int> sequence_lens_data(batch_size, (int)seq_length);
      test.AddInput<int>("sequence_lens", sequence_lens_dims, sequence_lens_data);

      std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
      std::vector<float> initial_h_data({0.0F, 0.0F, 0.0F});
      test.AddInput<float>("initial_h", initial_h_dims, initial_h_data);
    } else {
      test.AddOptionalInputEdge<float>();
      test.AddOptionalInputEdge<int>();
      test.AddOptionalInputEdge<float>();
    }

    if (outputOption == RNNOutputY || outputOption == RNNOutputBoth) {
      std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
      std::vector<float> Y_data({-0.052289959F, -0.062934637F, -0.21133657F,
                                 -0.48205593F, 0.23896417F, -0.31342113F,
                                 -0.47428668F, -0.27460238F, -0.46153161F,
                                 -0.41242906F, -0.32563525F, -0.79238516F,
                                 -0.74626476F, -0.07818383F, -0.75139415F});
      test.AddOutput<float>("Y", Y_dims, Y_data);
    } else {
      test.AddOptionalOutputEdge<float>();
    }

    if (outputOption == RNNOutputY_h || outputOption == RNNOutputBoth) {
      std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
      std::vector<float> Y_h_data({-0.74626476F, -0.07818383F, -0.75139415F});
      test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);
    } else {
      test.AddOptionalOutputEdge<float>();
    }

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  };

  {
    OpTester test_use_default("RNN");
    test_use_default.AddAttribute("hidden_size", hidden_size);
    run_test(test_use_default, true, RNNOutputY);
  }

  {
    OpTester test_use_default("RNN");
    test_use_default.AddAttribute("hidden_size", hidden_size);
    run_test(test_use_default, true, RNNOutputY_h);
  }

  {
    OpTester test_use_default("RNN");
    test_use_default.AddAttribute("hidden_size", hidden_size);
    run_test(test_use_default, true, RNNOutputBoth);
  }

  {
    OpTester test_do_not_use_default("RNN");
    test_do_not_use_default.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
    test_do_not_use_default.AddAttribute("direction", "forward");
    test_do_not_use_default.AddAttribute("hidden_size", hidden_size);

    run_test(test_do_not_use_default, false, RNNOutputY);
  }
  {
    OpTester test_do_not_use_default("RNN");
    test_do_not_use_default.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
    test_do_not_use_default.AddAttribute("direction", "forward");
    test_do_not_use_default.AddAttribute("hidden_size", hidden_size);
    run_test(test_do_not_use_default, false, RNNOutputY_h);
  }

  {
    OpTester test_do_not_use_default("RNN");
    test_do_not_use_default.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
    test_do_not_use_default.AddAttribute("direction", "forward");
    test_do_not_use_default.AddAttribute("hidden_size", hidden_size);
    run_test(test_do_not_use_default, false, RNNOutputBoth);
  }
}

TEST(RNNTest, DISABLED_RNN_reverse_direction) {
  int64_t num_directions = 1, input_size = 2, hidden_size = 3, batch_size = 1, seq_length = 5;

  // In case of useDefault, attributes, inputs or outputs are not set.
  // Otherwise they are set (with values may or may not be the same as ONNX default values).
  auto runTest = [&](OpTester& test, bool useDefault, RNNOutputOption outputOption) {
    std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
    std::vector<float> X_data({0.54881352F, 0.71518934F,
                               0.60276335F, 0.54488319F,
                               0.42365479F, 0.64589411F,
                               0.4375872F, 0.891773F,
                               0.96366274F, 0.38344151F});
    test.AddInput<float>("X", X_dims, X_data);

    std::vector<int64_t> W_dims = {num_directions, hidden_size, input_size};
    std::vector<float> W_data({-0.74535543F, 0.21360011F, 1.0782362F, 0.092641734F, -1.0087538F, -0.97021431F});
    test.AddInput<float>("W", W_dims, W_data);

    std::vector<int64_t> R_dims = {num_directions, hidden_size, hidden_size};
    std::vector<float> R_data({-0.7322467F, -0.95795155F, -0.058495734F,
                               -0.7271859F, -0.29820377F, -0.85114992F,
                               -0.097570196F, 0.82271612F, 0.1396943F});
    test.AddInput<float>("R", R_dims, R_data);

    if (!useDefault) {
      std::vector<int64_t> B_dims = {num_directions, 2 * hidden_size};
      std::vector<float> B_data({0.0F, 0.0F, 0.0F,
                                 0.0F, 0.0F, 0.0F});
      test.AddInput<float>("B", B_dims, B_data);

      std::vector<int64_t> sequence_lens_dims({batch_size});
      std::vector<int> sequence_lens_data(batch_size, (int)seq_length);
      test.AddInput<int>("sequence_lens", sequence_lens_dims, sequence_lens_data);

      std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
      std::vector<float> initial_h_data({0.0F, 0.0F, 0.0F});
      test.AddInput<float>("initial_h", initial_h_dims, initial_h_data);
    } else {
      test.AddOptionalInputEdge<float>();
      test.AddOptionalInputEdge<int>();
      test.AddOptionalInputEdge<float>();
    }

    std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
    std::vector<float> Y_data({-0.55397642F, 0.83026606F, -0.51471221F,
                               -0.55358219F, 0.8341592F, -0.44313878F,
                               -0.60828412F, 0.78948581F, -0.34582433F,
                               -0.40591392F, 0.89962566F, -0.61860478F,
                               -0.56242156F, 0.79118007F, -0.872658F});
    if (outputOption == RNNOutputY || outputOption == RNNOutputBoth) {
      test.AddOutput<float>("Y", Y_dims, Y_data);
    } else {
      test.AddOptionalOutputEdge<float>();
    }

    std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
    std::vector<float> Y_h_data({-0.55397642F, 0.83026606F, -0.51471221F});
    if (outputOption == RNNOutputY_h || outputOption == RNNOutputBoth) {
      test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);
    } else {
      test.AddOptionalOutputEdge<float>();
    }

    test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kTensorrtExecutionProvider});
  };

  // TODO: bring in these tests
  {
    OpTester test_use_default("RNN");
    test_use_default.AddAttribute("direction", "reverse");
    test_use_default.AddAttribute("hidden_size", hidden_size);
    runTest(test_use_default, true, RNNOutputY);
  }
  {
    OpTester test_use_default("RNN");
    test_use_default.AddAttribute("direction", "reverse");
    test_use_default.AddAttribute("hidden_size", hidden_size);
    runTest(test_use_default, true, RNNOutputY_h);
  }

  {
    OpTester test_use_default("RNN");
    test_use_default.AddAttribute("direction", "reverse");
    test_use_default.AddAttribute("hidden_size", hidden_size);
    runTest(test_use_default, true, RNNOutputBoth);
  }

  {
    OpTester test_do_not_use_default("RNN");
    test_do_not_use_default.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
    test_do_not_use_default.AddAttribute("direction", "reverse");
    test_do_not_use_default.AddAttribute("hidden_size", hidden_size);
    runTest(test_do_not_use_default, false, RNNOutputY);
  }
  {
    OpTester test_do_not_use_default("RNN");
    test_do_not_use_default.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
    test_do_not_use_default.AddAttribute("direction", "reverse");
    test_do_not_use_default.AddAttribute("hidden_size", hidden_size);
    runTest(test_do_not_use_default, false, RNNOutputY_h);
  }
  {
    OpTester test_do_not_use_default("RNN");
    test_do_not_use_default.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
    test_do_not_use_default.AddAttribute("direction", "reverse");
    test_do_not_use_default.AddAttribute("hidden_size", hidden_size);
    runTest(test_do_not_use_default, false, RNNOutputBoth);
  }
}

TEST(RNNTest, RNN_invalid_sequence_lens) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  int64_t num_directions = 1, input_size = 2, hidden_size = 3, batch_size = 2, seq_length = 3;

  auto run_test = [&](const std::vector<int>& sequence_lens,
                      const std::string& error_msg) {
    OpTester test("RNN");

    test.AddAttribute("activations", vector<string>(num_directions, "Tanh"));
    test.AddAttribute("direction", "forward");
    test.AddAttribute("hidden_size", hidden_size);

    std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
    std::vector<float> X_data({0.1f, 0.2f, 0.3f,
                               0.4f, 0.5f, 0.6f,
                               0.7f, 0.8f, 0.9f,
                               0.11f, 0.12f, 0.13f});

    test.AddInput<float>("X", X_dims, X_data);

    std::vector<int64_t> W_dims = {num_directions, hidden_size, input_size};
    std::vector<float> W_data({-0.1f, 0.2f, 1.f, -2.f, -1.f, 3.f});

    test.AddInput<float>("W", W_dims, W_data);

    std::vector<int64_t> R_dims = {num_directions, hidden_size, hidden_size};
    std::vector<float> R_data({0.f, 0.f, 0.f,
                               0.f, 0.f, 0.f,
                               0.f, 0.f, 0.f});
    test.AddInput<float>("R", R_dims, R_data);

    std::vector<int64_t> B_dims = {num_directions, 2 * hidden_size};
    std::vector<float> B_data{0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    test.AddInput<float>("B", B_dims, B_data);

    // std::vector<int64_t> sequence_lens_dims({batch_size});
    std::vector<int64_t> sequence_lens_dims{gsl::narrow_cast<int64_t>(sequence_lens.size())};
    test.AddInput<int>("sequence_lens", sequence_lens_dims, sequence_lens);

    std::vector<int64_t> initial_h_dims = {num_directions, batch_size, hidden_size};
    std::vector<float> initial_h_data{0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    test.AddInput<float>("initial_h", initial_h_dims, initial_h_data);

    // Y
    test.AddOptionalOutputEdge<float>();

    std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
    std::vector<float> Y_h_data{0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

    // the CUDA RNN version allows the invalid sequence lengths, so disable testing on CUDA and TensorRT
    test.Run(OpTester::ExpectResult::kExpectFailure, error_msg, {kCudaExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
  };

  // should batch batch_size to be valid
  std::vector<int> invalid_num_seq_len_entries(batch_size - 1, 5);

  run_test(invalid_num_seq_len_entries, "Input sequence_lens must have shape {2}. Actual:{1}");

  // 0 is an invalid value
  std::vector<int> bad_seq_len_entry{0, 5};
  run_test(bad_seq_len_entry, "Invalid value/s in sequence_lens. All values must be > 0 and < seq_length.");
}

TEST(RNNTest, RNN_bidirectional_with_sequence_lens) {
  // values copied from WinML RNN test
  int64_t num_directions = 2, input_size = 2, hidden_size = 4, batch_size = 3, seq_length = 2;

  OpTester test("RNN");

  test.AddAttribute("activations", vector<string>{"Relu", "Tanh"});
  test.AddAttribute("direction", "bidirectional");
  test.AddAttribute("hidden_size", hidden_size);

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<float> X_data({1.f, 2.f,
                             3.f, 4.f,
                             5.f, 6.f,

                             3.f, 1.f,
                             4.f, 4.f,
                             6.f, 3.f});

  test.AddInput<float>("X", X_dims, X_data);

  std::vector<int64_t> W_dims = {num_directions, hidden_size, input_size};
  std::vector<float> W_data{0.1f, 0.2f,
                            0.3f, 0.4f,
                            0.5f, 0.6f,
                            0.5f, 0.6f,
                            0.7f, 0.3f,
                            0.8f, 0.4f,
                            0.9f, 0.5f,
                            0.1f, 0.6f};

  test.AddInput<float>("W", W_dims, W_data);

  std::vector<int64_t> R_dims = {num_directions, hidden_size, hidden_size};
  std::vector<float> R_data{0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f,
                            0.2f, 0.2f, 0.2f, 0.2f,
                            0.2f, 0.2f, 0.2f, 0.2f,
                            0.2f, 0.2f, 0.2f, 0.2f,
                            0.2f, 0.2f, 0.2f, 0.2f};
  test.AddInput<float>("R", R_dims, R_data);

  std::vector<int64_t> B_dims = {num_directions, 2 * hidden_size};
  std::vector<float> B_data{0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f,
                            0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f, 0.1f};
  test.AddInput<float>("B", B_dims, B_data);

  std::vector<int64_t> sequence_lens_dims{batch_size};
  std::vector<int> sequence_len_data{1, 2, 2};
  test.AddInput<int>("sequence_lens", sequence_lens_dims, sequence_len_data);

  // Y
  std::vector<int64_t> Y_dims{seq_length, num_directions, batch_size, hidden_size};
  std::vector<float> Y_data{
      0.69999999f, 1.3000001f, 1.9000001f, 1.9000001f,
      1.3000001f, 2.7f, 4.0999999f, 4.0999999f,
      1.9000001f, 4.0999999f, 6.3000002f, 6.3000002f,

      0.90514827f, 0.94680595f, 0.97045195f, 0.90514827f,
      0.99963105f, 0.99990892f, 0.99997759f, 0.99877548f,
      0.99999332f, 0.99999928f, 1.f, 0.99992537f,

      0.f, 0.f, 0.f, 0.f,
      2.6200001f, 4.2200003f, 5.8199997f, 5.8199997f,
      3.2600002f, 5.0600004f, 6.8600001f, 6.8600001f,

      0.f, 0.f, 0.f, 0.f,
      0.99955046f, 0.99990916f, 0.99998164f, 0.99505484f,
      0.99995017f, 0.99999166f, 0.99999857f, 0.9890275f};

  test.AddOutput<float>("Y", Y_dims, Y_data);

  std::vector<int64_t> Y_h_dims{num_directions, batch_size, hidden_size};
  std::vector<float> Y_h_data{0.69999999f, 1.3000001f, 1.9000001f, 1.9000001f,
                              2.6200001f, 4.2200003f, 5.8199997f, 5.8199997f,
                              3.2600002f, 5.0600004f, 6.8600001f, 6.8600001f,

                              0.90514827f, 0.94680595f, 0.97045195f, 0.90514827f,
                              0.99963105f, 0.99990892f, 0.99997759f, 0.99877548f,
                              0.99999332f, 0.99999928f, 1.f, 0.99992537f};

  test.AddOutput<float>("Y_h", Y_h_dims, Y_h_data);

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kCudaExecutionProvider, kTensorrtExecutionProvider, kOpenVINOExecutionProvider});
}

TEST(RNNTest, RNN_with_invalid_activation_load_failure) {
  // TODO: Unskip when fixed #41968513
  if (DefaultDmlExecutionProvider().get() != nullptr) {
    GTEST_SKIP() << "Skipping because of the following error: MLOperatorAuthorImpl.cpp(1817): The parameter is incorrect.";
  }

  OpTester test("RNN");
  int64_t num_directions = 1, input_size = 1, hidden_size = 1, seq_length = 1;

  test.AddAttribute("activations", vector<string>(num_directions, "Invalid_activation"));
  test.AddAttribute("direction", "reverse");
  test.AddAttribute("hidden_size", hidden_size);

  int batch_size = 1;

  std::vector<int64_t> X_dims = {seq_length, batch_size, input_size};
  std::vector<float> X_data{0.F};
  test.AddInput<float>("X", X_dims, X_data);

  std::vector<int64_t> W_dims = {num_directions, hidden_size, input_size};
  std::vector<float> W_data({0.F});
  test.AddInput<float>("W", W_dims, W_data);

  std::vector<int64_t> R_dims = {num_directions, hidden_size, hidden_size};
  std::vector<float> R_data({0.F});
  test.AddInput<float>("R", R_dims, R_data);

  std::vector<int64_t> Y_dims = {seq_length, num_directions, batch_size, hidden_size};
  std::vector<float> Y_data({0.F});
  test.AddOutput<float>("Y", Y_dims, Y_data);

  test.Run(OpTester::ExpectResult::kExpectFailure, "RNN op: Invalid activation attribute - Invalid_activation",
           {kCudaExecutionProvider, kTensorrtExecutionProvider});
}

}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST(LSTMTest, ForwardCompute) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCpuExecutionProvider());

  OpTester test("LSTMInternal", 1, onnxruntime::kMSDomain);

  const int sequence_length = 2;
  const int batch_size = 2;
  const int hidden_size = 3;
  const int input_size = 2;
  const int directions = 1;

  test.AddAttribute<int64_t>("hidden_size", hidden_size);

  test.AddInput<float>("input_tensor", {sequence_length, batch_size, input_size},
                       {1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f});
  test.AddInput<float>("weights", {directions, 4 * hidden_size, input_size},
                       {0.1f, 0.2f,
                        0.3f, 0.4f,
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        10.0f, 11.0f,
                        12.0f, 13.0f,
                        0.6f, 0.7f,
                        0.8f, 0.9f,
                        6.0f, 7.0f,
                        8.0f, 9.0f,
                        16.0f, 17.0f,
                        18.0f, 19.0f});
  test.AddInput<float>("recurrence_weights", {directions, 4 * hidden_size, hidden_size},
                       {8.0f, 9.0f, 16.0f,
                        17.0f, 18.0f, 19.0,
                        0.1f, 0.2f, 0.3f,
                        0.4f, 1.0f, 2.0f,
                        0.6f, 0.7f, 0.8f,
                        0.9f, 6.0f, 7.0f,
                        3.0f, 4.0f, 10.0f,
                        11.0f, 12.0f, 13.0f,
                        0.1f, 0.3f, 0.5f,
                        0.7f, 1.0f, 3.0f,
                        5.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f});
  test.AddInput<float>("bias", {directions, 8 * hidden_size},
                       {3.0f, 4.0f, 10.0f, 11.0f, 12.0f, 13.0f, 0.6f, 0.7f,
                        0.8f, 0.9f, 6.0f, 7.0f, 0.1f, 0.2f, 0.3f, 0.4f,
                        1.0f, 2.0f, 8.0f, 9.0f, 16.0f, 17.0f, 18.0f, 19.0f});

  test.AddOutput<float>(
      "HAll", {sequence_length, directions, batch_size, hidden_size},
      {0.75225f, 0.76019f, 0.761594f,
       0.758769f, 0.761509f, 0.761594f,
       0.962449f, 0.963791f, 0.964028f,
       0.963552f, 0.964013f, 0.964028f});
  test.AddOutput<float>(
      "final_h", {directions, batch_size, hidden_size},
      {0.962449f, 0.963791f, 0.964028f, 0.963552f, 0.964013f, 0.964028f});
  test.AddOutput<float>(
      "final_c", {directions, batch_size, hidden_size},
      {1.97812f, 1.99667f, 2.0f, 1.99331f, 1.9998f, 2.0f});
  test.AddOutput<float>(
      "CAll", {sequence_length, directions, batch_size, hidden_size},
      {0.978119f, 0.996665f, 1.0f,
       0.993307f, 0.999797f, 1.0f,
       1.97812f, 1.99667f, 2.0f,
       1.99331f, 1.9998f, 2.0f});
  test.AddOutput<float>(
      "iofc", {sequence_length, directions, batch_size, 4 * hidden_size},
      {0.978119f, 0.996665f, 1.0f, 1.0f, 1.0f, 1.0f, 0.999988f, 0.999998f,
       1.0f, 1.0f, 1.0f, 1.0f, 0.993307f, 0.999797f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(LSTMTest, BackwardCompute) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCpuExecutionProvider());

  OpTester test("LSTMGrad", 1, onnxruntime::kMSDomain);

  const int sequence_length = 2;
  const int batch_size = 2;
  const int hidden_size = 3;
  const int input_size = 2;
  const int directions = 1;

  test.AddAttribute<int64_t>("hidden_size", hidden_size);

  test.AddInput<float>("input_tensor", {sequence_length, batch_size, input_size},
                       {1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f});
  test.AddInput<float>("weights", {directions, 4 * hidden_size, input_size},
                       {0.1f, 0.2f,
                        0.3f, 0.4f,
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        10.0f, 11.0f,
                        12.0f, 13.0f,
                        0.6f, 0.7f,
                        0.8f, 0.9f,
                        6.0f, 7.0f,
                        8.0f, 9.0f,
                        16.0f, 17.0f,
                        18.0f, 19.0f});
  test.AddInput<float>("recurrence_weights", {directions, 4 * hidden_size, hidden_size},
                       {8.0f, 9.0f, 16.0f,
                        17.0f, 18.0f, 19.0,
                        0.1f, 0.2f, 0.3f,
                        0.4f, 1.0f, 2.0f,
                        0.6f, 0.7f, 0.8f,
                        0.9f, 6.0f, 7.0f,
                        3.0f, 4.0f, 10.0f,
                        11.0f, 12.0f, 13.0f,
                        0.1f, 0.3f, 0.5f,
                        0.7f, 1.0f, 3.0f,
                        5.0f, 7.0f, 8.0f,
                        9.0f, 10.0f, 11.0f});
  test.AddInput<float>("bias", {directions, 8 * hidden_size},
                       {3.0f, 4.0f, 10.0f, 11.0f, 12.0f, 13.0f, 0.6f, 0.7f,
                        0.8f, 0.9f, 6.0f, 7.0f, 0.1f, 0.2f, 0.3f, 0.4f,
                        1.0f, 2.0f, 8.0f, 9.0f, 16.0f, 17.0f, 18.0f, 19.0f});

  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();

  test.AddInput<float>(
      "HAll", {sequence_length, directions, batch_size, hidden_size},
      {0.7522504, 0.76019, 0.7615942,
       0.75876904, 0.76150876, 0.7615942,
       0.9624486, 0.9637912, 0.9640276,
       0.96355164, 0.9640132, 0.9640276});
  test.AddInput<float>(
      "CAll", {sequence_length, directions, batch_size, hidden_size},
      {0.97811866, 0.9966652, 1.0,
       0.9933072, 0.99979657, 1.0,
       1.9781187, 1.9966652, 2.0,
       1.9933071, 1.9997966, 2.0});
  test.AddInput<float>(
      "iofc", {sequence_length, directions, batch_size, 4 * hidden_size},
      {0.22702627, 0.96168727, 0.92005587, 0.11435849, 0.42692447, 0.8240583, 0.44825846, 0.2687109,
       0.5906311, 0.40993997, 0.48799622, 0.928153, 0.41776767, 0.03468254, 0.09252429, 0.08168347,
       0.5076645, 0.4623195, 0.6043845, 0.804295, 0.3812583, 0.7358246, 0.7646758, 0.5207596,
       0.43702832, 0.38906136, 0.02930347, 0.84817046, 0.9626478, 0.79656714, 0.22243384, 0.88983405,
       0.4643094, 0.3431123, 0.28768018, 0.49338564, 0.43890277, 0.5425567, 0.5510384, 0.39404443,
       0.23032443, 0.83058596, 0.6846567, 0.0170034, 0.57249254, 0.13880704, 0.13557956, 0.2181483});

  test.AddInput<float>(
      "grad_HAll", {sequence_length, directions, batch_size, hidden_size},
      {0.13658696, 0.37761405, 0.5353489,
       0.53866684, 0.02047455, 0.42426682,
       0.12669823, 0.28094783, 0.82699543,
       0.12687224, 0.4540311, 0.4124293});

  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();

  test.AddOutput<float>(
      "dX", {sequence_length, batch_size, input_size},
      {9.02288f, 9.77558f, 4.23378f, 4.6432f, 1.92046f, 2.09879f, 1.87627f, 2.06453f});
  test.AddOutput<float>(
      "dW", {directions, 4 * hidden_size, input_size},
      {0.030251f, 0.0453894f,
       0.0252481f, 0.0440468f,
       0.100279f, 0.204773f,
       0.459666f, 0.695113f,
       1.30497f, 2.19383f,
       1.84552f, 3.00503f,
       0.00728328f, 0.0114973f,
       0.00449891f, 0.00850378f,
       0.0586922f, 0.0936911f,
       0.0477309f, 0.0758698f,
       0.230594f, 0.623739f,
       0.231839f, 0.440448f});
  test.AddOutput<float>(
      "dR", {directions, 4 * hidden_size, hidden_size},
      {0.000595693f, 0.000601335f, 0.000602285f,
       0.00117793f, 0.00118905f, 0.00119092f,
       0.00148242f, 0.00149119f, 0.00149222f,
       0.0339609f, 0.0341655f, 0.03419f,
       0.0661984f, 0.0664883f, 0.0665086f,
       0.139635f, 0.140814f, 0.141001f,
       0.00159f, 0.00160274f, 0.00160469f,
       0.00150713f, 0.00152239f, 0.00152504f,
       0.0132026f, 0.0133107f, 0.0133275f,
       0.00346555f, 0.00349401f, 0.00349843f,
       0.0081494f, 0.00821467f, 0.00822465f,
       0.0104138f, 0.0104568f, 0.0104593f});
  test.AddOutput<float>(
      "dB", {directions, 8 * hidden_size},
      {0.00756918f, 0.00939937f, 0.0522473f, 0.117724f, 0.444431f, 0.579753f, 0.00210701f, 0.00200243f,
       0.0174995f, 0.0140694f, 0.196573f, 0.104304f, 0.00756918f, 0.00939937f, 0.0522473f, 0.117724f,
       0.444431f, 0.579753f, 0.00210701f, 0.00200243f, 0.0174995f, 0.0140694f, 0.196573f, 0.104304f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

}  // namespace test
}  // namespace onnxruntime

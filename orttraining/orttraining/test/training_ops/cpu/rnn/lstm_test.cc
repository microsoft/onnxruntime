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
      {0.75127375, 0.75987947, 0.7615942,
       0.7584724, 0.76148975, 0.7615942,
       0.9622827, 0.9637389, 0.9640276,
       0.96350163, 0.96401, 0.9640276});
  test.AddOutput<float>(
      "final_h", {directions, batch_size, hidden_size},
      {0.9622827, 0.9637389, 0.9640276, 0.96350163, 0.96401, 0.9640276});
  test.AddOutput<float>(
      "final_c", {directions, batch_size, hidden_size},
      {1.975873, 1.99593, 2.0, 1.9926085, 1.9997516, 2.0});
  test.AddOutput<float>(
      "CAll", {sequence_length, directions, batch_size, hidden_size},
      {0.975873, 0.9959299, 1.0,
       0.9926085, 0.9997515, 1.0,
       1.975873, 1.99593, 2.0,
       1.9926085, 1.9997516, 2.0});
  test.AddOutput<float>(
      "iofc", {sequence_length, directions, batch_size, 4 * hidden_size},
      {0.97811866, 0.9966652, 1.0, 1.0, 1.0, 1.0, 0.9999876, 0.9999981,
       1.0, 1.0, 1.0, 1.0, 0.9933072, 0.99979657, 1.0, 1.0,
       1.0, 1.0, 0.9999999, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
       1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0});

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
      {9.021257, 9.773498, 4.2335634, 4.642917, 1.9204623, 2.0987916, 1.8762696, 2.0645313});
  test.AddOutput<float>(
      "dW", {directions, 4 * hidden_size, input_size},
      {0.0302016, 0.04528961,
       0.02524809, 0.04404683,
       0.10027865, 0.20477335,
       0.45897973, 0.6935005,
       1.3049657, 2.1938272,
       1.845519, 3.0050251,
       0.00728328, 0.0114973,
       0.00449892, 0.00850379,
       0.05869219, 0.09369113,
       0.04764829, 0.0756738,
       0.2305943, 0.6237395,
       0.23183903, 0.44044772});
  test.AddOutput<float>(
      "dR", {directions, 4 * hidden_size, hidden_size},
      {0.00059569, 0.00060134, 0.00060228,
       0.00117793, 0.00118905, 0.00119092,
       0.00148242, 0.00149119, 0.00149222,
       0.03396086, 0.03416551, 0.03419005,
       0.06619843, 0.06648831, 0.06650861,
       0.13963531, 0.14081432, 0.14100051,
       0.00159, 0.00160274, 0.00160469,
       0.00150714, 0.00152239, 0.00152504,
       0.01320259, 0.01331073, 0.01332749,
       0.00346555, 0.00349401, 0.00349843,
       0.0081494, 0.00821468, 0.00822465,
       0.01041376, 0.01045675, 0.01045929});
  test.AddOutput<float>(
      "dB", {directions, 8 * hidden_size},
      {0.00754401, 0.00939937, 0.05224735, 0.11726041, 0.44443065, 0.57975316, 0.00210701, 0.00200243,
       0.01749947, 0.01401276, 0.1965726, 0.10430434, 0.00754401, 0.00939937, 0.05224735, 0.11726041,
       0.44443065, 0.57975316, 0.00210701, 0.00200243, 0.01749947, 0.01401276, 0.1965726, 0.10430434});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

}  // namespace test
}  // namespace onnxruntime

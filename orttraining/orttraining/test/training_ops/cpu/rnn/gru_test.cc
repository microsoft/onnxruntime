// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test/providers/compare_provider_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

TEST(GRUTest, ForwardCompute) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCpuExecutionProvider());

  OpTester test("GRUTraining", 1, onnxruntime::kMSDomain);

  constexpr int sequence_length = 2;
  constexpr int batch_size = 2;
  constexpr int hidden_size = 3;
  constexpr int input_size = 2;
  constexpr int directions = 1;

  test.AddAttribute<int64_t>("hidden_size", hidden_size);

  test.AddInput<float>("input_tensor", {sequence_length, batch_size, input_size},
                       {1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f});
  test.AddInput<float>("weights", {directions, 3 * hidden_size, input_size},
                       {0.1f, 0.2f,
                        0.3f, 0.4f,
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        10.0f, 11.0f,
                        12.0f, 13.0f,
                        0.6f, 0.7f,
                        0.8f, 0.9f,
                        6.0f, 7.0f});
  test.AddInput<float>("recurrence_weights", {directions, 3 * hidden_size, hidden_size},
                       {8.0f, 9.0f, 16.0f,
                        17.0f, 18.0f, 19.0,
                        0.1f, 0.2f, 0.3f,
                        0.4f, 1.0f, 2.0f,
                        0.6f, 0.7f, 0.8f,
                        0.9f, 6.0f, 7.0f,
                        3.0f, 4.0f, 10.0f,
                        11.0f, 12.0f, 13.0f,
                        0.1f, 0.3f, 0.5f});
  test.AddInput<float>("bias", {directions, 6 * hidden_size},
                       {3.0f, 4.0f, 10.0f, 11.0f, 12.0f, 13.0f, 0.6f, 0.7f,
                        0.8f, 0.9f, 6.0f, 7.0f, 0.1f, 0.2f, 0.3f, 0.4f,
                        1.0f, 2.0f});

  test.AddOutput<float>(
      "HAll", {sequence_length, directions, batch_size, hidden_size},
      {0.00993967f, 1.01322e-05f, 0.0f, 0.00301844f,
       5.96047e-07f, 0.0f, 0.0167009f, 1.43641e-05f,
       0.0f, 0.00519647f, 8.9407e-07f, 0.0f});
  test.AddOutput<float>(
      "final_h", {directions, batch_size, hidden_size},
      {0.0167009f, 1.43641e-05f, 0.0f, 0.00519647f, 8.9407e-07f, 0.0f});
  test.AddOutput<float>(
      "zrh", {sequence_length, directions, batch_size, 3 * hidden_size},
      {0.990048f, 0.99999f, 1.0f, 1.0f, 1.0f, 1.0f, 0.998778f, 0.999939f,
       1.0f, 0.996982f, 0.999999f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 0.99317f, 0.999996f, 1.0f, 1.0f, 1.0f, 1.0f,
       0.999914f, 0.999998f, 1.0f, 0.997815f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

TEST(GRUTest, BackwardCompute) {
  std::vector<std::unique_ptr<IExecutionProvider>> providers;
  providers.emplace_back(DefaultCpuExecutionProvider());

  OpTester test("GRUGrad", 1, onnxruntime::kMSDomain);

  constexpr int sequence_length = 2;
  constexpr int batch_size = 2;
  constexpr int hidden_size = 3;
  constexpr int input_size = 2;
  constexpr int directions = 1;

  test.AddAttribute<int64_t>("hidden_size", hidden_size);

  test.AddInput<float>("input_tensor", {sequence_length, batch_size, input_size},
                       {1.0f, 3.0f, 5.0f, 7.0f, 2.0f, 4.0f, 6.0f, 8.0f});
  test.AddInput<float>("weights", {directions, 3 * hidden_size, input_size},
                       {0.1f, 0.2f,
                        0.3f, 0.4f,
                        1.0f, 2.0f,
                        3.0f, 4.0f,
                        10.0f, 11.0f,
                        12.0f, 13.0f,
                        0.6f, 0.7f,
                        0.8f, 0.9f,
                        6.0f, 7.0f});
  test.AddInput<float>("recurrence_weights", {directions, 3 * hidden_size, hidden_size},
                       {8.0f, 9.0f, 16.0f,
                        17.0f, 18.0f, 19.0,
                        0.1f, 0.2f, 0.3f,
                        0.4f, 1.0f, 2.0f,
                        0.6f, 0.7f, 0.8f,
                        0.9f, 6.0f, 7.0f,
                        3.0f, 4.0f, 10.0f,
                        11.0f, 12.0f, 13.0f,
                        0.1f, 0.3f, 0.5f});
  test.AddInput<float>("bias", {directions, 6 * hidden_size},
                       {3.0f, 4.0f, 10.0f, 11.0f, 12.0f, 13.0f, 0.6f, 0.7f,
                        0.8f, 0.9f, 6.0f, 7.0f, 0.1f, 0.2f, 0.3f, 0.4f,
                        1.0f, 2.0f});

  test.AddOptionalInputEdge<float>();
  test.AddOptionalInputEdge<float>();

  test.AddInput<float>(
      "HAll", {sequence_length, directions, batch_size, hidden_size},
      {0.00993967f, 1.01322e-05f, 0.0f, 0.00301844f,
       5.96047e-07f, 0.0f, 0.0167009f, 1.43641e-05f,
       0.0f, 0.00519647f, 8.9407e-07f, 0.0f});
  test.AddInput<float>(
      "zrh", {sequence_length, directions, batch_size, 3 * hidden_size},
      {0.990048f, 0.99999f, 1.0f, 1.0f, 1.0f, 1.0f, 0.998778f, 0.999939f,
       1.0f, 0.996982f, 0.999999f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 0.99317f, 0.999996f, 1.0f, 1.0f, 1.0f, 1.0f,
       0.999914f, 0.999998f, 1.0f, 0.997815f, 1.0f, 1.0f, 1.0f, 1.0f,
       1.0f, 1.0f, 1.0f, 1.0f});

  test.AddInput<float>(
      "grad_HAll", {sequence_length, directions, batch_size, hidden_size},
      {0.13658696f, 0.37761405f, 0.5353489f,
       0.53866684f, 0.02047455f, 0.42426682f,
       0.12669823f, 0.28094783f, 0.82699543f,
       0.12687224f, 0.4540311f, 0.4124293f});

  test.AddOptionalInputEdge<float>();

  test.AddOutput<float>(
      "dX", {sequence_length, batch_size, input_size},
      {-0.000249756f, -0.000501315f, -0.000199651f, -0.000399207f,
       -8.53292e-05f, -0.000170508f, -2.75774e-05f, -5.51547e-05f});
  test.AddOutput<float>(
      "dW", {directions, 3 * hidden_size, input_size},
      {-0.015847f, -0.0271209f, -1.11526e-05f,
       -2.73875e-05f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       6.51075e-06f, 1.92345e-05f, 8.03933e-10f,
       2.4027e-09f, 0.0f, 0.0f});
  test.AddOutput<float>(
      "dR", {directions, 3 * hidden_size, hidden_size},
      {-9.28927e-06f, -8.78505e-09f, 0.0f,
       -1.11518e-08f, -1.13678e-11f, 0.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       0.0f, 0.0f, 0.0f,
       1.47958e-09f, 1.50824e-12f, 0.0f,
       4.52003e-14f, 4.60759e-17f, 0.0f,
       0.0f, 0.0f, 0.0f});
  test.AddOutput<float>(
      "dB", {directions, 6 * hidden_size},
      {-0.00563696f, -8.11746e-06f, 0.0f, 0.0f, 0.0f, 0.0f,
       6.36189e-06f, 7.99385e-10f, 0.0f, -0.00563696f, -8.11746e-06f, 0.0f,
       0.0f, 0.0f, 0.0f, 6.36189e-06f, 7.99385e-10f, 0.0f});

  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &providers);
}

}  // namespace test
}  // namespace onnxruntime

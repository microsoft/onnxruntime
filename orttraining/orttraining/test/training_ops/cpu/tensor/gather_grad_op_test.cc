// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

const std::vector<MLFloat16> FloatToMLFloat16(const std::vector<float>& float_data) {
  std::vector<MLFloat16> new_data;
  for (const auto& f : float_data) {
    new_data.push_back(MLFloat16(math::floatToHalf(f)));
  }
  return new_data;
}

template <typename T>
void CalculateOutput(const int64_t stride, const int64_t num_input_before_gather_axis,
                     const int64_t num_input_from_gather_axis, const std::vector<T>& grad, const std::vector<int64_t>& indices, std::vector<T>& output) {
  std::map<int64_t, std::vector<T> > indices_grad;
  for (int64_t t = 0; t < num_input_before_gather_axis; ++t) {
    auto offset1 = t * num_input_from_gather_axis;
    for (size_t i = 0; i < indices.size(); ++i) {
      auto offset2 = (t * indices.size() + i) * stride;
      auto index = offset1 + indices[i];
      if (indices_grad.count(index)) {
        for (int64_t j = 0; j < stride; ++j) {
          indices_grad[index][j] += grad[offset2 + j];
        }
      } else {
        for (int64_t j = 0; j < stride; ++j) {
          indices_grad[index].push_back(grad[offset2 + j]);
        }
      }
    }
  }
  for (auto& itr : indices_grad) {
    for (int64_t i = 0; i < stride; ++i) {
      output[itr.first * stride + i] = itr.second[i];
    }
  }
}

#ifdef USE_CUDA
//TODO: Currently this cannot pass CI, due to GPU architecture problem
TEST(GatherOpTest, Gather_axis0_indices2d_half) {
  if (NeedSkipIfCudaArchLowerThan(700)) {
    return;
  }

  OpTester test("Gather");
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<MLFloat16>("data", {3, 3},
                           FloatToMLFloat16({0.0f, 0.1f, 0.2f,
                                             1.0f, 1.1f, 1.2f,
                                             2.0f, 2.1f, 2.2f}));
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {1LL, 0LL,
                          2LL, 1LL});
  test.AddOutput<MLFloat16>("output", {2, 2, 3},
                            FloatToMLFloat16({1.0f, 1.1f, 1.2f, 0.0f, 0.1f, 0.2f,
                                              2.0f, 2.1f, 2.2f, 1.0f, 1.1f, 1.2f}));
  test.Run();
}

TEST(GatherGradOpTest, GatherGrad_axis0_indices2d_half) {
  if (NeedSkipIfCudaArchLowerThan(700)) {
    return;
  }

  OpTester test("GatherGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<int64_t>("shape", {2},
                         {3, 3});
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {0LL, 1LL,
                          0LL, 1LL});

  test.AddInput<MLFloat16>("grad", {2, 2, 3},
                           FloatToMLFloat16({0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5}));
  test.AddOutput<MLFloat16>("output", {3, 3},
                            FloatToMLFloat16({0, 2, 4, 6, 8, 10, 0, 0, 0}));
  test.Run();
}
#endif

TEST(GatherGradOpTest, GatherGrad_axis0_indices2d_float) {
  OpTester test("GatherGrad", 1, kMSDomain);
  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<int64_t>("shape", {2},
                         {3, 3});
  test.AddInput<int64_t>("indices", {2LL, 2LL},
                         {0LL, 1LL,
                          0LL, 1LL});

  test.AddInput<float>("grad", {2, 2, 3},
                       {0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5});
  test.AddOutput<float>("output", {3, 3},
                        {0, 2, 4, 6, 8, 10, 0, 0, 0});
  test.Run();
}

TEST(GatherGradOpTest, Gather_axis1_float_impl2) {
  OpTester test("GatherGrad", 1, kMSDomain);
  int64_t axis_0 = 3;
  int64_t axis_1 = 6;
  int64_t axis_2 = 128;
  int64_t output_shape = 4;
  RandomValueGenerator random{};
  std::vector<float> grad(random.Uniform<float>({axis_0 * axis_1 * axis_2}, 1.0f, 1.0f));
  std::vector<int64_t> indices(random.Uniform<int64_t>({axis_1 * axis_2}, 0, 3));

  std::vector<int64_t> shape{axis_0, output_shape};
  std::vector<float> output(axis_0 * output_shape);

  CalculateOutput(1, axis_0, 4, grad, indices, output);

  test.AddAttribute<int64_t>("axis", 1LL);
  test.AddInput<int64_t>("shape", {2},
                         shape);
  test.AddInput<int64_t>("indices", {axis_1, axis_2},
                         indices);
  test.AddInput<float>("grad", {axis_0, axis_1, axis_2},
                       grad);
  test.AddOutput<float>("output", shape,
                        output);
  test.Run();
}

TEST(GatherGradOpTest, Gather_axis0_float_impl2) {
  OpTester test("GatherGrad", 1, kMSDomain);
  int64_t axis_0 = 3;
  int64_t axis_1 = 6;
  int64_t axis_2 = 128;
  int64_t output_shape = 4;
  RandomValueGenerator random{};
  std::vector<float> grad(random.Uniform<float>({axis_1 * axis_2 * output_shape}, 1.0f, 1.0f));
  std::vector<int64_t> indices(random.Uniform<int64_t>({axis_1 * axis_2}, 0, 3));

  std::vector<int64_t> shape{axis_0, output_shape};
  std::vector<float> output(axis_0 * output_shape);

  CalculateOutput(output_shape, 1, 12, grad, indices, output);

  test.AddAttribute<int64_t>("axis", 0LL);
  test.AddInput<int64_t>("shape", {2},
                         shape);
  test.AddInput<int64_t>("indices", {axis_1, axis_2},
                         indices);
  test.AddInput<float>("grad", {axis_1, axis_2, output_shape},
                       grad);
  test.AddOutput<float>("output", shape,
                        output);
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

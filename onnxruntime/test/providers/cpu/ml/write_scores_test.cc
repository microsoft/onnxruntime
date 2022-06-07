// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "core/framework/tensor.h"
#include "core/providers/cpu/ml/ml_common.h"
#include "test/framework/dummy_allocator.h"
#include <random>

using namespace onnxruntime;
using namespace onnxruntime::ml;
class WriteScores : public ::testing::Test {
 protected:
  std::random_device r;
  std::default_random_engine rd{r()};
};

TEST_F(WriteScores, multiple_scores_transform_none) {
  POST_EVAL_TRANSFORM trans[] = {POST_EVAL_TRANSFORM::NONE,
                                 POST_EVAL_TRANSFORM::LOGISTIC,
                                 POST_EVAL_TRANSFORM::SOFTMAX,
                                 POST_EVAL_TRANSFORM::SOFTMAX_ZERO,
                                 POST_EVAL_TRANSFORM::PROBIT};
  for (POST_EVAL_TRANSFORM tran : trans) {
    InlinedVector<float> v1(100);
    std::uniform_real_distribution<float> uniform_dist(-1, 1);
    std::generate_n(v1.data(), v1.size(), [&]() -> float { return uniform_dist(rd); });
    InlinedVector<float> v2 = v1;
    auto alloc = std::make_shared<test::DummyAllocator>();
    Tensor t(DataTypeImpl::GetType<float>(), {static_cast<int64_t>(v1.size())}, alloc);
    if (tran == POST_EVAL_TRANSFORM::SOFTMAX_ZERO) {
      //random set one number as zero
      v1[3] = 0;
    }
    write_scores<float>(v1, tran, 0, &t, -1);
    const float* output_data = t.Data<float>();
    //verify the result
    if (tran == POST_EVAL_TRANSFORM::NONE) {
      for (size_t i = 0; i != v2.size(); ++i) {
        EXPECT_FLOAT_EQ(v1[i], output_data[i]);
      }
    } else if (tran == POST_EVAL_TRANSFORM::SOFTMAX || tran == POST_EVAL_TRANSFORM::SOFTMAX_ZERO) {
      //sum of the output values should be near to 1
      double sum = 0;
      for (size_t i = 0; i != v1.size(); ++i) {
        sum += output_data[i];
      }
      EXPECT_NEAR(sum, 1, 1e-6);

      if (tran == POST_EVAL_TRANSFORM::SOFTMAX_ZERO) {
        EXPECT_FLOAT_EQ(output_data[3], 0);
      }
    }
  }
}

TEST_F(WriteScores, single_score_transform_none) {
  InlinedVector<float> v1;
  std::uniform_real_distribution<float> uniform_dist(-100, 100);
  v1.push_back(uniform_dist(rd));
  InlinedVector<float> v2 = v1;
  auto alloc = std::make_shared<test::DummyAllocator>();
  Tensor t(DataTypeImpl::GetType<float>(), {static_cast<int64_t>(v1.size())}, alloc);
  write_scores<float>(v1, POST_EVAL_TRANSFORM::NONE, 0, &t, -1);
  const float* output_data = t.Data<float>();
  for (size_t i = 0; i != v2.size(); ++i) {
    EXPECT_FLOAT_EQ(v1[i], output_data[i]);
  }
}

TEST_F(WriteScores, single_score_transform_none_add_second_class) {
  for (int i = 0; i != 4; ++i) {
    InlinedVector<float> v1;
    std::uniform_real_distribution<float> uniform_dist(-100, 100);
    v1.push_back(uniform_dist(rd));
    InlinedVector<float> v2 = v1;
    auto alloc = std::make_shared<test::DummyAllocator>();
    Tensor t(DataTypeImpl::GetType<float>(), {2}, alloc);
    write_scores<float>(v1, POST_EVAL_TRANSFORM::NONE, 0, &t, i);
    const float* output_data = t.Data<float>();
	EXPECT_NEAR(output_data[1] + output_data[0], i == 0 || i == 1 ? 1 : 0, 1e-5);
    if (i == 0 || i == 1) {
      EXPECT_FLOAT_EQ(v2[0], output_data[1]);
      EXPECT_FLOAT_EQ(1 - v2[0], output_data[0]);
    }
  }
}
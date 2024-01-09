// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/common/dnnl_op_test_utils.h"
#include "core/util/math.h"
#include "default_providers.h"

namespace onnxruntime {
namespace test {

namespace test_sign_internal {

template <class T, class A>
struct make_type {
  static T make(A v) {
    return T(v);
  }
};

template <class A>
struct make_type<MLFloat16, A> {
  static MLFloat16 make(A v) {
    return MLFloat16(float(v));
  }
};

template <class A>
struct make_type<BFloat16, A> {
  static BFloat16 make(A v) {
    return BFloat16(float(v));
  }
};

template <class T, class OutputIter>
typename std::enable_if<!std::numeric_limits<T>::is_signed &&
                        !std::is_same<T, MLFloat16>::value &&
                        !std::is_same<T, BFloat16>::value>::type
GenerateSequence(OutputIter out) {
  for (int i = 0; i < 7; ++i) {
    *out = make_type<T, int>::make(i);
    ++out;
  }
}

template <class T, class OutputIter>
typename std::enable_if<std::numeric_limits<T>::is_signed ||
                        std::is_same<T, MLFloat16>::value ||
                        std::is_same<T, BFloat16>::value>::type
GenerateSequence(OutputIter out) {
  for (int i = -5; i < 2; ++i) {
    *out = make_type<T, int>::make(i);
    ++out;
  }
}

template <class T>
struct ToTestableType {
  static T to_type(T v) {
    return v;
  }
};

template <>
struct ToTestableType<MLFloat16> {
  static float to_type(MLFloat16 v) {
    return v.ToFloat();
  }
};

template <>
struct ToTestableType<BFloat16> {
  static float to_type(BFloat16 v) {
    return v.ToFloat();
  }
};

template <class T, class ForwardIter, class OutputIter>
typename std::enable_if<!std::numeric_limits<T>::is_signed &&
                        !std::is_same<T, MLFloat16>::value &&
                        !std::is_same<T, BFloat16>::value>::type
TestImpl(ForwardIter first, ForwardIter last, OutputIter out) {
  std::transform(first, last, out, [](T v) {
    auto t = ToTestableType<T>::to_type(v);
    if (t == 0) {
      t = 0;
    } else {
      t = 1;
    }
    return make_type<T, decltype(t)>::make(t);
  });
}

template <class T, class ForwardIter, class OutputIter>
typename std::enable_if<std::numeric_limits<T>::is_signed ||
                        std::is_same<T, MLFloat16>::value ||
                        std::is_same<T, BFloat16>::value>::type
TestImpl(ForwardIter first, ForwardIter last, OutputIter out) {
  std::transform(first, last, out, [](T v) {
    auto t = ToTestableType<T>::to_type(v);
    if (t == 0) {
      t = 0;
    } else if (t > 0) {
      t = 1;
    } else {
      t = -1;
    }
    return make_type<T, decltype(t)>::make(t);
  });
}
}  // namespace test_sign_internal

TEST(MathOpTest, Sign_uint64) {
  using namespace test_sign_internal;
  OpTester test("Sign", 13);

  std::vector<int64_t> input_dims{7};
  std::vector<uint64_t> input;
  GenerateSequence<uint64_t>(std::back_inserter(input));
  ASSERT_EQ(input.size(), 7U);
  test.AddInput<uint64_t>("input", input_dims, input);

  std::vector<uint64_t> output;
  TestImpl<uint64_t>(input.cbegin(), input.cend(), std::back_inserter(output));
  test.AddOutput<uint64_t>("output", input_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}
// we disable this test for openvino as openvino ep supports only FP32 Precision
TEST(MathOpTest, Sign_int64) {
  using namespace test_sign_internal;
  OpTester test("Sign", 13);

  std::vector<int64_t> input_dims{7};
  std::vector<int64_t> input;
  GenerateSequence<int64_t>(std::back_inserter(input));
  ASSERT_EQ(input.size(), 7U);
  test.AddInput<int64_t>("input", input_dims, input);

  std::vector<int64_t> output;
  TestImpl<int64_t>(input.cbegin(), input.cend(), std::back_inserter(output));
  test.AddOutput<int64_t>("output", input_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kOpenVINOExecutionProvider});
}

TEST(MathOpTest, Sign_float) {
  using namespace test_sign_internal;
  OpTester test("Sign", 13);

  std::vector<int64_t> input_dims{7};
  std::vector<float> input;
  GenerateSequence<float>(std::back_inserter(input));
  ASSERT_EQ(input.size(), 7U);
  test.AddInput<float>("input", input_dims, input);

  std::vector<float> output;
  TestImpl<float>(input.cbegin(), input.cend(), std::back_inserter(output));
  test.AddOutput<float>("output", input_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

TEST(MathOpTest, Sign_double) {
  using namespace test_sign_internal;
  OpTester test("Sign", 13);

  std::vector<int64_t> input_dims{7};
  std::vector<double> input;
  GenerateSequence<double>(std::back_inserter(input));
  ASSERT_EQ(input.size(), 7U);
  test.AddInput<double>("input", input_dims, input);

  std::vector<double> output;
  TestImpl<double>(input.cbegin(), input.cend(), std::back_inserter(output));
  test.AddOutput<double>("output", input_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}
TEST(MathOpTest, Sign_MLFloat16) {
  using namespace test_sign_internal;
  OpTester test("Sign", 13);

  std::vector<int64_t> input_dims{7};
  std::vector<MLFloat16> input;
  GenerateSequence<MLFloat16>(std::back_inserter(input));
  ASSERT_EQ(input.size(), 7U);
  test.AddInput<MLFloat16>("input", input_dims, input);

  std::vector<MLFloat16> output;
  TestImpl<MLFloat16>(input.cbegin(), input.cend(), std::back_inserter(output));
  test.AddOutput<MLFloat16>("output", input_dims, output);
  test.Run(OpTester::ExpectResult::kExpectSuccess);
}

// Currently BFloat16 is not enabled for Sign kernel
// TEST(MathOpTest, Sign_BFloat16) {
//  using namespace test_sign_internal;
//  OpTester test("Sign", 9);
//
//  std::vector<int64_t> input_dims{7};
//  std::vector<BFloat16> input;
//  GenerateSequence<BFloat16>(std::back_inserter(input));
//  ASSERT_EQ(input.size(), 7U);
//  test.AddInput<BFloat16>("input", input_dims, input);
//
//  std::vector<BFloat16> output;
//  TestImpl<BFloat16>(input.cbegin(), input.cend(), std::back_inserter(output));
//  test.AddOutput<BFloat16>("output", input_dims, output);
//  test.Run(OpTester::ExpectResult::kExpectSuccess);
//}

#if defined(USE_DNNL)
TEST(MathOpTest, Sign_bfloat16) {
#ifdef USE_DNNL
  if (!DnnlHasBF16Support()) {
    LOGS_DEFAULT(WARNING) << "Hardware does NOT support BF16";
    return;
  }
#endif
  using namespace test_sign_internal;
  OpTester test("Sign", 13);

  std::vector<int64_t> input_dims{7};
  std::vector<BFloat16> input;
  GenerateSequence<BFloat16>(std::back_inserter(input));
  ASSERT_EQ(input.size(), 7U);
  test.AddInput<BFloat16>("input", input_dims, input);

  std::vector<BFloat16> output;
  TestImpl<BFloat16>(input.cbegin(), input.cend(), std::back_inserter(output));
  test.AddOutput<BFloat16>("output", input_dims, output);
  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
#if defined(USE_DNNL)
  execution_providers.push_back(DefaultDnnlExecutionProvider());
#endif  //  USE_DNNL
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

}  // namespace test
}  // namespace onnxruntime

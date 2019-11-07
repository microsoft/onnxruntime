// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "gsl/gsl"

#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

namespace {
constexpr char kOpName[] = "Where";
constexpr int kOpVersion = 9;

template <typename TDest, typename TSrc>
std::vector<TDest> CastVector(const std::vector<TSrc>& source) {
  std::vector<TDest> target{};
  target.reserve(source.size());
  std::transform(source.begin(), source.end(), std::back_inserter(target),
                 [](TSrc n) { return static_cast<TDest>(n); });
  return target;
}

template <typename TNumeric>
void WhereBasicNumericTest() {
  OpTester test{kOpName, kOpVersion};

  const std::vector<int64_t> dims{2, 2};

  test.AddInput<bool>("condition", dims,
                      {false, true, true, false});
  test.AddInput<TNumeric>("X", dims,
                          CastVector<TNumeric, int>({1, 2, 3, 4}));
  test.AddInput<TNumeric>("Y", dims,
                          CastVector<TNumeric, int>({5, 6, 7, 8}));

  test.AddOutput<TNumeric>("output", dims,
                           CastVector<TNumeric, int>({5, 2, 3, 8}));

  test.Run();
}

template <typename T>
void WhereBroadcastTest(const T& x_value, const T& y_value) {
  auto condition_values = {true, false, true};  // std::initializer_list<bool> for OpTester::AddInput<bool>()
  const std::vector<T> X_values(3, x_value);
  const std::vector<T> Y_values(3, y_value);

  {
    OpTester test{kOpName, kOpVersion};

    test.AddInput<bool>("condition", {1, 1, 3}, condition_values);
    test.AddInput<T>("X", {1, 3, 1}, X_values);
    test.AddInput<T>("Y", {3, 1, 1}, Y_values);

    std::vector<T> result{};
    result.reserve(3 * 3 * 3);
    for (int i = 0; i < 3 * 3; ++i) {
      result.insert(result.end(), {x_value, y_value, x_value});
    }
    test.AddOutput<T>("output", {3, 3, 3}, result);

    test.Run();
  }

  {
    OpTester test{kOpName, kOpVersion};

    test.AddInput<bool>("condition", {3, 1, 1}, condition_values);
    test.AddInput<T>("X", {1, 1, 3}, X_values);
    test.AddInput<T>("Y", {1, 3, 1}, Y_values);

    std::vector<T> result{};
    result.reserve(3 * 3 * 3);
    for (int i = 0; i < 3; ++i) {
      result.insert(
          result.end(), 3 * 3,
          gsl::make_span(condition_values.begin(), condition_values.size())[i] ? x_value : y_value);
    }
    test.AddOutput<T>("output", {3, 3, 3}, result);

    test.Run();
  }
}
}  // namespace

TEST(WhereOpTest, BasicNumeric) {
  WhereBasicNumericTest<float>();
}

TEST(WhereOpTest, BasicString) {
  OpTester test{kOpName, kOpVersion};

  test.AddInput<bool>("condition", {2}, {false, true});
  const std::vector<std::string> X{"small0", "small1"};
  test.AddInput<std::string>("X", {2}, X);
  const std::vector<std::string> Y{std::string(1024, 'a'), std::string(1024, 'b')};
  test.AddInput<std::string>("Y", {2}, Y);

  test.AddOutput<std::string>("output", {2}, {Y[0], X[1]});

  test.Run();
}

TEST(WhereOpTest, Broadcast) {
  WhereBroadcastTest<float>(1.0f, 0.0f);
  WhereBroadcastTest<std::string>("true", "false");
}

TEST(WhereOpTest, BroadcastDimWithZero) {
  // test where broadcast is possible, and dim of 0 should be selected
  OpTester test{kOpName, kOpVersion};

  test.AddInput<bool>("condition", {3}, {true, false, true});
  test.AddInput<int64_t>("X", {1, 3}, {1, 2, 3});
  test.AddInput<int64_t>("Y", {0, 1}, {});

  test.AddOutput<int64_t>("output", {0, 3}, {});

  // exclude NGraph as this isn't handled by that EP
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {kNGraphExecutionProvider});
}
}  // namespace test
}  // namespace onnxruntime

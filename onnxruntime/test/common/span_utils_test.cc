// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "test/util/include/asserts.h"
#include "core/common/span_utils.h"
#include "core/common/inlined_containers.h"

namespace onnxruntime {
namespace test {

namespace {
void f(gsl::span<const int64_t> s) {
  std::copy(s.begin(), s.end(), std::ostream_iterator<int64_t>(std::cout, ", "));
  std::cout << std::endl;
}
}

TEST(Common, SpanUtilsTests) {
  {
    // list by var
    auto list = {1, 2, 3};
    auto span = AsSpan(list);
    ASSERT_TRUE(SpanEq(gsl::make_span(list.begin(), list.size()), span));
    // no type conversion int -> int64_t
    // use std::array
  }

  {
    AsSpan({1, 2, 3}); // -> gsl::span<const int>
    f(AsSpan<int64_t>({1, 2, 3})); //  -> gsl::span<const int64_t>
  }

  {
    auto list = {1, 2, 3};
    auto arr = std::array<int64_t, 3>{1, 2, 3};
    ASSERT_EQ(std::size(list), arr.size());
    f(arr);
  }

  {
    std::vector<int64_t> vec = {1, 2, 3};
    auto span = AsSpan(vec);
    ASSERT_EQ(vec.size(), span.size());
    ASSERT_TRUE(SpanEq(gsl::make_span(vec), span));
    f(span);
  }

  {
    InlinedVector<int64_t> vec = {1, 2, 3};
    auto span = AsSpan(vec);
    ASSERT_EQ(vec.size(), span.size());
    ASSERT_TRUE(SpanEq(gsl::make_span(vec), span));
    f(span);
  }

  {
    // Array
    int64_t arr[] = {1, 2, 3};
    auto span = AsSpan(arr);
    ASSERT_EQ(std::size(arr), span.size());
    ASSERT_TRUE(SpanEq(gsl::make_span(arr), span));
    f(span);
  }
}
}  // namespace test
}  // namespace onnxruntime
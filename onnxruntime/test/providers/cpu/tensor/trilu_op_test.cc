// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "core/util/math.h"

namespace onnxruntime {
namespace test {

TEST(TriluOpTest, two_by_two_float_upper) {
  OpTester test("Trilu", 14, kOnnxDomain);
  int64_t up = 1;
  test.AddAttribute("upper", up);
  test.AddInput<float>("X", {2, 2}, {4.f, 7.f, 2.f, 6.f});
  test.AddOutput<float>("Y", {2, 2}, {4.f, 7.f, 0.f, 6.f});
  test.Run();
}

TEST(TriluOpTest, two_by_two_float_lower) {
  OpTester test("Trilu", 14, kOnnxDomain);
  int64_t up = 0;
  test.AddAttribute("upper", up);
  test.AddInput<float>("X", {2, 2}, {4.f, 7.f, 2.f, 6.f});
  test.AddOutput<float>("Y", {2, 2}, {4.f, 0.f, 2.f, 6.f});
  test.Run();
}

TEST(TriluOpTest, two_by_two_double_upper) {
  OpTester test("Trilu", 14, kOnnxDomain);
  test.AddInput<double>("X", {2, 2}, {4, 7, 2, 6});
  test.AddInput<int64_t>("k", {1}, {1});
  test.AddOutput<double>("Y", {2, 2}, {0, 7, 0, 0});
  test.Run();
}

TEST(TriluOpTest, two_by_two_double_lower) {
  OpTester test("Trilu", 14, kOnnxDomain);
  int64_t up = 0;
  test.AddAttribute("upper", up);
  test.AddInput<double>("X", {2, 2}, {4, 7, 2, 6});
  test.AddInput<int64_t>("k", {1}, {1});
  test.AddOutput<double>("Y", {2, 2}, {4, 7, 2, 6});
  test.Run();
}

TEST(TriluOpTest, two_by_two_long_upper) {
  OpTester test("Trilu", 14, kOnnxDomain);
  int64_t up = 1;
  test.AddAttribute("upper", up);
  test.AddInput<int64_t>("X", {2, 2}, {4, 7, 2, 6});
  test.AddOutput<int64_t>("Y", {2, 2}, {4, 7, 0, 6});
  test.Run();
}

TEST(TriluOpTest, two_by_two_long_lower) {
  OpTester test("Trilu", 14, kOnnxDomain);
  int64_t up = 0;
  test.AddAttribute("upper", up);
  test.AddInput<int64_t>("X", {2, 2}, {4, 7, 2, 6});
  test.AddOutput<int64_t>("Y", {2, 2}, {4, 0, 2, 6});
  test.Run();
}

TEST(TriluOpTest, three_dim_float_upper) {
  OpTester test("Trilu", 14, kOnnxDomain);
  test.AddInput<float>("X", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.AddInput<int64_t>("k", {1}, {1});
  test.AddOutput<float>("Y", {2, 3, 4},
    {0.f, 1.f, 5.f, 8.f,
     0.f, 0.f, 2.f, 4.f,
     0.f, 0.f, 0.f, 3.f,
     0.f, 6.f, 2.f, 1.f,
     0.f, 0.f, 5.f, 8.f,
     0.f, 0.f, 0.f, 4.f,
    });
  test.Run();
}

TEST(TriluOpTest, three_dim_float_lower) {
  OpTester test("Trilu", 14, kOnnxDomain);
  int64_t up = 0;
  test.AddAttribute("upper", up);
  test.AddInput<float>("X", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.AddInput<int64_t>("k", {1}, {1});
  test.AddOutput<float>("Y", {2, 3, 4},
    {4.f, 1.f, 0.f, 0.f,
     4.f, 3.f, 2.f, 0.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 0.f, 0.f,
     4.f, 1.f, 5.f, 0.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.Run();
}

TEST(TriluOpTest, neg_k_float_upper) {
  OpTester test("Trilu", 14, kOnnxDomain);
  int64_t up = 1;
  test.AddAttribute("upper", up);
  test.AddInput<float>("X", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.AddInput<int64_t>("k", {1}, {-1});
  test.AddOutput<float>("Y", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     0.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     0.f, 3.f, 2.f, 4.f,
    });
  test.Run();
}

TEST(TriluOpTest, neg_k_float_lower) {
  OpTester test("Trilu", 14, kOnnxDomain);
  int64_t up = 0;
  test.AddAttribute("upper", up);
  test.AddInput<float>("X", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.AddInput<int64_t>("k", {1}, {-1});
  test.AddOutput<float>("Y", {2, 3, 4},
    {0.f, 0.f, 0.f, 0.f,
     4.f, 0.f, 0.f, 0.f,
     6.f, 1.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f,
     4.f, 0.f, 0.f, 0.f,
     4.f, 3.f, 0.f, 0.f,
    });
  test.Run();
}

TEST(TriluTest, small_k_float_upper) {
  OpTester test("Trilu", 14, kOnnxDomain);
  test.AddInput<float>("X", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.AddInput<int64_t>("k", {1}, {-5});
  test.AddOutput<float>("Y", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.Run();
}

TEST(TriluOpTest, small_k_float_lower) {
  OpTester test("Trilu", 14, kOnnxDomain);
  int64_t up = 0;
  test.AddAttribute("upper", up);
  test.AddInput<float>("X", {2, 3, 4},
    {4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
     6.f, 1.f, 2.f, 3.f,
     1.f, 6.f, 2.f, 1.f,
     4.f, 1.f, 5.f, 8.f,
     4.f, 3.f, 2.f, 4.f,
    });
  test.AddInput<int64_t>("k", {1}, {-5});
  test.AddOutput<float>("Y", {2, 3, 4},
    {0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f,
     0.f, 0.f, 0.f, 0.f,
    });
  test.Run();  
}

TEST(TriluOpTest, zero_dim_upper) {
  OpTester test("Trilu", 14, kOnnxDomain);
  test.AddInput<float>("X", {2, 3, 0}, {});
  test.AddInput<int64_t>("k", {1}, {0});
  test.AddOutput<float>("Y", {2, 3, 0}, {});
  test.Run();
}

TEST(TriluOpTest, zero_dim_lower) {
  OpTester test("Trilu", 14, kOnnxDomain);
  int64_t up = 0;
  test.AddAttribute("upper", up);
  test.AddInput<float>("X", {2, 3, 0}, {});
  test.AddInput<int64_t>("k", {1}, {0});
  test.AddOutput<float>("Y", {2, 3, 0}, {});
  test.Run();
}

TEST(TriluOpTest, zero_dim_2_upper) {
  OpTester test("Trilu", 14, kOnnxDomain);
  test.AddInput<float>("X", {2, 0, 0}, {});
  test.AddInput<int64_t>("k", {1}, {-5});
  test.AddOutput<float>("Y", {2, 0, 0}, {});
  test.Run();
}

TEST(TriluOpTest, zero_dim_2_lower) {
  OpTester test("Trilu", 14, kOnnxDomain);
  int64_t up = 0;
  test.AddAttribute("upper", up);
  test.AddInput<float>("X", {2, 0, 0}, {});
  test.AddInput<int64_t>("k", {1}, {-5});
  test.AddOutput<float>("Y", {2, 0, 0}, {});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

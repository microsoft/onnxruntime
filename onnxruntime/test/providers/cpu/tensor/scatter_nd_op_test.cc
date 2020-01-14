// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(ScatterNDOpTest, ScatterND_scaler_string_int64) {
  OpTester test1("ScatterND", 11);
  test1.AddInput<std::string>("data", {2,2}, {"h","h","o","z"});
  test1.AddInput<int64_t>("indices", {2}, {0,1});
  test1.AddInput<std::string>("updates", {}, {"k"});
  test1.AddOutput<std::string>("output", {2,2}, {"h","k","o","z"});
  test1.Run();

  OpTester test2("ScatterND", 11);
  test2.AddInput<std::string>("data", {6}, {"h","k","o","o","l","t"});
  test2.AddInput<int64_t>("indices", {1}, {3});
  test2.AddInput<std::string>("updates", {}, {"z"});
  test2.AddOutput<std::string>("output", {6}, {"h","k","o","z","l","t"});
  test2.Run();

  OpTester test3("ScatterND", 11);
  test3.AddInput<std::string>("data", {3,2}, {"h","k","o","z","l","z"});
  test3.AddInput<int64_t>("indices", {2}, {2,1});
  test3.AddInput<std::string>("updates", {}, {"t"});
  test3.AddOutput<std::string>("output", {3,2}, {"h","k","o","z","l","t"});
  test3.Run();
}

TEST(ScatterNDOpTest, ScatterND_matrice_int64_int64) {
  OpTester test("ScatterND", 11);
  test.AddInput<int64_t> ("data", {2,2}, {1LL,1LL,2LL,2LL});
  test.AddInput<int64_t> ("indices", {2,2}, {0LL,0LL,1LL,1LL});
  test.AddInput<int64_t>("updates", {2}, {0LL,3LL});
  test.AddOutput<int64_t>("output", {2,2}, {0LL,1LL,2LL,3LL});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_matrice_string_int64) {
  OpTester test1("ScatterND", 11);
  test1.AddInput<std::string>("data", {2,2,2}, {"egg","dance","bob","air","smart","terry","laugh","kite"});
  test1.AddInput<int64_t>("indices", {2,1,2}, {0,1,1,0});
  test1.AddInput<std::string>("updates", {2,1,2}, {"air","bob","terry","smart"});
  test1.AddOutput<std::string>("output", {2,2,2}, {"egg","dance","air","bob","terry","smart","laugh","kite"});
  test1.Run();

  OpTester test2("ScatterND", 11);
  test2.AddInput<std::string>("data", {3,3}, {"egg","","air","","terry","smart","laugh","","hop"});
  test2.AddInput<int64_t>("indices", {3,2}, {2,1,1,0,0,1});
  test2.AddInput<std::string>("updates", {3}, {"kite","bob","dance"});
  test2.AddOutput<std::string>("output", {3,3}, {"egg","dance","air","bob","terry","smart","laugh","kite","hop"});
  test2.Run();
}

TEST(ScatterNDOpTest, ScatterND_slice_float_int64_t) {
  OpTester test("ScatterND", 11);
  test.AddInput<float>("data", {2,2}, {0.0f,0.1f,0.1f,0.1f});
  test.AddInput<int64_t>("indices", {2,1}, {1LL,0LL});
  test.AddInput<float>("updates", {2,2}, {0.2f,0.3f,0.0f,0.1f});
  test.AddOutput<float>("output", {2,2}, {0.0f,0.1f,0.2f,0.3f});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_slice_double_int64_t) {
  OpTester test("ScatterND", 11);
  test.AddInput<double>("data", {2,2}, {0.0f,0.1f,0.1f,0.1f});
  test.AddInput<int64_t>("indices", {2,1}, {1LL,0LL});
  test.AddInput<double>("updates", {2,2}, {0.2f,0.3f,0.0f,0.1f});
  test.AddOutput<double>("output", {2,2}, {0.0f,0.1f,0.2f,0.3f});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_3tensor_int64) {
  OpTester test1("ScatterND", 11);
  test1.AddInput<int64_t>("data", {2,2,2}, {0LL,1LL,1LL,1LL,1LL,1LL,6LL,7LL});
  test1.AddInput<int64_t>("indices", {2,2}, {0LL,1LL,1LL,0LL});
  test1.AddInput<int64_t>("updates", {2,2}, {2LL,3LL,4LL,5LL});
  test1.AddOutput<int64_t>("output", {2,2,2}, {0LL,1LL,2LL,3LL,4LL,5LL,6LL,7LL});
  test1.Run();

  OpTester test2("ScatterND", 11);
  test2.AddInput<int8_t>("data", {2,2,2}, {0,0,2,3,4,0,6,7});
  test2.AddInput<int64_t>("indices", {2,3}, {0,0,1,1,0,1});
  test2.AddInput<int8_t>("updates", {2}, {1,5});
  test2.AddOutput<int8_t>("output", {2,2,2}, {0,1,2,3,4,5,6,7});
  test2.Run();

  OpTester test3("ScatterND", 11);
  test3.AddInput<int16_t>("data", {2,2,2}, {0,1,2,3,0,1,2,3});
  test3.AddInput<int64_t>("indices", {1,1}, {1LL});
  test3.AddInput<int16_t>("updates", {1,2,2}, {4,5,6,7});
  test3.AddOutput<int16_t>("output", {2,2,2}, {0,1,2,3,4,5,6,7});
  test3.Run();
}

TEST(ScatterNDOpTest, ScatterND_batched_index_int64) {
  OpTester test("ScatterND", 11);
  test.AddInput<int64_t>("data", {2,2}, {2LL,3LL,2LL,3LL});
  test.AddInput<int64_t>("indices", {2,1,2}, {0LL,0LL,0LL,1LL});
  test.AddInput<int64_t>("updates", {2,1}, {0LL,1LL});
  test.AddOutput<int64_t>("output", {2,2}, {0LL,1LL,2LL,3LL});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_batched_index_bool_int64) {
  OpTester test("ScatterND", 11);
  test.AddInput<bool>("data", {2,2}, {false,true,false,true});
  test.AddInput<int64_t>("indices", {2,1,2}, {0LL,0LL,0LL,1LL});
  test.AddInput<bool>("updates", {2,1}, {true,false});
  test.AddOutput<bool>("output", {2,2}, {true,false,false,true});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_sliced_index_int64) {
  OpTester test("ScatterND", 11);
  test.AddInput<int64_t>("data", {2,2}, {0LL,0LL,0LL,0LL});
  test.AddInput<int64_t>("indices", {2,1,1}, {1LL,0LL});
  test.AddInput<int64_t>("updates", {2,1,2}, {2LL,3LL,0LL,1LL});
  test.AddOutput<int64_t>("output", {2,2}, {0LL,1LL,2LL,3LL});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_sliced_index_string_int64) {
  OpTester test("ScatterND", 11);
  test.AddInput<std::string>("data", {2,2}, {"","","",""});
  test.AddInput<int64_t>("indices", {2,1,1}, {1LL,0LL});
  test.AddInput<std::string>("updates", {2,1,2}, {"f","ghi","ab","cde"});
  test.AddOutput<std::string>("output", {2,2}, {"ab","cde","f","ghi"});
  test.Run();
}

TEST(ScatterNDOpTest, ScatterND_batched_3tensor_int64) {
  OpTester test1("ScatterND", 11);
  test1.AddInput<uint32_t>("data", {2,2,2}, {0,0,0,0,0,0,0,0});
  test1.AddInput<int64_t>("indices", {2,2,2}, {0LL,1LL,1LL,0LL,0LL,0LL,1LL,1LL});
  test1.AddInput<uint32_t>("updates", {2,2,2}, {2,3,4,5,0,1,6,7});
  test1.AddOutput<uint32_t>("output", {2,2,2}, {0,1,2,3,4,5,6,7});
  test1.Run();

  OpTester test2("ScatterND", 11);
  test2.AddInput<uint32_t>("data", {2,2,2}, {0,0,2,0,4,0,0,7});
  test2.AddInput<int64_t>("indices", {2,2,3}, {0,0,1,1,0,1,0,1,1,1,1,0});
  test2.AddInput<uint32_t>("updates", {2,2}, {1,5,3,6});
  test2.AddOutput<uint32_t>("output", {2,2,2}, {0,1,2,3,4,5,6,7});
  test2.Run();

  OpTester test3("ScatterND", 11);
  test3.AddInput<int64_t>("data", {2,2,2}, {1LL,0LL,0LL,0LL,0LL,0LL,0LL,0LL});
  test3.AddInput<int64_t>("indices", {2,1,1}, {1,0});
  test3.AddInput<int64_t>("updates", {2,1,2,2}, {4LL,5LL,6LL,7LL,0LL,1LL,2LL,3LL});
  test3.AddOutput<int64_t>("output", {2,2,2}, {0LL,1LL,2LL,3LL,4LL,5LL,6LL,7LL});
  test3.Run();
}

}  // namespace test
}  // namespace onnxruntime

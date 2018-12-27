// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(DynamicSliceTest, dynamic_full_slice_varied_types) {
  OpTester test1 ("DynamicSlice", 1);
  test1.AddInput  <int32_t> ("data",   {3,3}, {1,2,3,4,5,6,7,8,9});
  test1.AddInput  <int32_t> ("starts", {2},   {1,1});
  test1.AddInput  <int32_t> ("ends",   {2},   {3,3});
  test1.AddOutput <int32_t> ("output", {2,2}, {5,6,8,9});
  test1.Run();

  OpTester test2("DynamicSlice", 1);
  test2.AddInput  <int64_t> ("data",   {3,3}, {1LL,2LL,3LL,4LL,5LL,6LL,7LL,8LL,9LL});
  test2.AddInput  <int32_t> ("starts", {2},   {1,1});
  test2.AddInput  <int32_t> ("ends",   {2},   {3,3});
  test2.AddOutput <int64_t> ("output", {2,2}, {5LL,6LL,8LL,9LL});
  test2.Run();

  OpTester test3("DynamicSlice", 1);
  test3.AddInput  <std::string> ("data",   {3,3}, {"a","b","c","d","e","f","g","h","i"});
  test3.AddInput  <int64_t>     ("starts", {2},   {1,1});
  test3.AddInput  <int64_t>     ("ends",   {2},   {3,3});
  test3.AddOutput <std::string> ("output", {2,2}, {"e","f","h","i"});
  test3.Run();

  OpTester test4("DynamicSlice", 1);
  test4.AddInput  <float>    ("data",   {3,3}, {1.1f,2.2f,3.3f,4.4f,5.5f,6.6f,7.7f,8.8f,9.9f});
  test4.AddInput  <int32_t>  ("starts", {2},   {1,1});
  test4.AddInput  <int32_t>  ("ends",   {2},   {3,3});
  test4.AddOutput <float>    ("output", {2,2}, {5.5f,6.6f,8.8f,9.9f});
  test4.Run();

  OpTester test5("DynamicSlice", 1);
  test5.AddInput  <bool>    ("data",   {3,3}, {false,true,false,false,false,false,true,false,true});
  test5.AddInput  <int32_t> ("starts", {2},   {1,1});
  test5.AddInput  <int32_t> ("ends",   {2},   {3,3});
  test5.AddOutput <bool>    ("output", {2,2}, {false,false,false,true});
  test5.Run();
}

TEST(DynamicSliceTest, dynamic_slice_with_axes) {
  OpTester test1 ("DynamicSlice", 1);
  test1.AddInput  <int32_t> ("data",   {3,3}, {1,2,3,4,5,6,7,8,9});
  test1.AddInput  <int32_t> ("starts", {1},   {1});
  test1.AddInput  <int32_t> ("ends",   {1},   {3});
  test1.AddInput  <int32_t> ("axes",   {1},   {1});
  test1.AddOutput <int32_t> ("output", {3,2}, {2,3,5,6,8,9});
  test1.Run();

  OpTester test2 ("DynamicSlice", 1);
  test2.AddInput  <int32_t> ("data",   {3,3,3}, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                 10,11,12,13,14,15,16,17,18,
                                                 19,20,21,22,23,24,25,26,27});
  test2.AddInput  <int32_t> ("starts", {1},     {1});
  test2.AddInput  <int32_t> ("ends",   {1},     {-1});
  test2.AddInput  <int32_t> ("axes",   {1},     {1});
  test2.AddOutput <int32_t> ("output", {3,1,3}, {4,5,6,13,14,15,22,23,24});
  test2.Run();

  OpTester test3 ("DynamicSlice", 1);
  test3.AddInput  <int32_t> ("data",   {3,3,3}, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                 10,11,12,13,14,15,16,17,18,
                                                 19,20,21,22,23,24,25,26,27});
  test3.AddInput  <int32_t> ("starts", {1},     {1});
  test3.AddInput  <int32_t> ("ends",   {1},     {2});
  test3.AddInput  <int32_t> ("axes",   {1},     {2});
  test3.AddOutput <int32_t> ("output", {3,3,1}, {2,5,8,11,14,17,20,23,26});
  test3.Run();

  OpTester test4 ("DynamicSlice", 1);
  test4.AddInput  <int32_t> ("data",   {3,3,3}, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                 10,11,12,13,14,15,16,17,18,
                                                 19,20,21,22,23,24,25,26,27});
  test4.AddInput  <int32_t> ("starts", {2},     {0,-2});
  test4.AddInput  <int32_t> ("ends",   {2},     {2,1000});
  test4.AddInput  <int32_t> ("axes",   {2},     {1,2});
  test4.AddOutput <int32_t> ("output", {3,2,2}, {2,3,5,6,11,12,14,15,20,21,23,24});
  test4.Run();

  OpTester test5 ("DynamicSlice", 1);
  test5.AddInput  <int32_t> ("data",   {3,3,3}, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                 10,11,12,13,14,15,16,17,18,
                                                 19,20,21,22,23,24,25,26,27});
  test5.AddInput  <int32_t> ("starts", {2},     {-3,0});
  test5.AddInput  <int32_t> ("ends",   {2},     {-1,2});
  test5.AddInput  <int32_t> ("axes",   {2},     {0,2});
  test5.AddOutput <int32_t> ("output", {2,3,2}, {1,2,4,5,7,8,10,11,13,14,16,17});
  test5.Run();

  OpTester test6 ("DynamicSlice", 1);
  test6.AddInput  <int32_t> ("data",   {3,3,3}, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                 10,11,12,13,14,15,16,17,18,
                                                 19,20,21,22,23,24,25,26,27});
  test6.AddInput  <int32_t> ("starts", {3},     {0,1,1});
  test6.AddInput  <int32_t> ("ends",   {3},     {1,3,2});
  test6.AddInput  <int32_t> ("axes",   {3},     {0,1,2});
  test6.AddOutput <int32_t> ("output", {1,2,1}, {5,8});
  test6.Run();
}

TEST(DynamicSliceTest, dynamic_slice_with_axes_7) {
  OpTester test7 ("DynamicSlice", 1);
  test7.AddInput  <int32_t> ("data",   {3,3,3}, {1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                 10,11,12,13,14,15,16,17,18,
                                                 19,20,21,22,23,24,25,26,27});
  test7.AddInput  <int32_t> ("starts", {3},     {1,0,1});
  test7.AddInput  <int32_t> ("ends",   {3},     {2,1,3});
  test7.AddInput  <int32_t> ("axes",   {3},     {2,0,1});
  test7.AddOutput <int32_t> ("output", {1,2,1}, {5,8});
  test7.Run();
}

}  // namespace Test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

namespace onnxruntime {
namespace test {

TEST(CropAndResizeTest, CropAndResize_1122) {
  OpTester test1 ("CropAndResize", 1, onnxruntime::kMSDomain);
  test1.AddInput  <float> ("X",   {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test1.AddInput  <float> ("rois", {3, 4},   {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test1.AddInput  <int32_t> ("batch_indices",    {3},    {0, 0, 0});
  test1.AddInput  <int32_t> ("crop_size",    {2},    {1, 1});
  test1.AddOutput <float> ("output", {3, 1, 1, 1}, {2.75f, 1.925f, 2.2f});
  test1.Run();

  OpTester test2 ("CropAndResize", 1, onnxruntime::kMSDomain);
  test2.AddInput  <float> ("X",   {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test2.AddInput  <float> ("rois", {3, 4},   {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test2.AddInput  <int32_t> ("batch_indices",   {3},   {0, 0, 0});
  test2.AddInput  <int32_t> ("crop_size",    {2},    {2, 2});
  test2.AddOutput <float> ("output", {3, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 1.1f, 1.65f, 2.2f, 2.75f, 1.1f, 2.2f, 2.2f, 3.3f});
  test2.Run();

  OpTester test3 ("CropAndResize", 1, onnxruntime::kMSDomain);
  test3.AddInput  <float> ("X",   {1, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f});
  test3.AddInput  <float> ("rois", {2, 4},   {0.0f, 0.0f, 1.5f, 1.5f, 0.25f, 0.25f, 0.75f, 0.5f});
  test3.AddInput  <int32_t> ("batch_indices",   {2},   {0, 0});
  test3.AddInput  <int32_t> ("crop_size",    {2},    {2, 2});
  test3.AddAttribute("extrapolation_value", (float)5.5);
  test3.AddOutput <float> ("output", {2, 1, 2, 2}, {1.1f, 5.5f, 5.5f, 5.5f, 1.925f, 2.2f, 3.025f, 3.3f});
  test3.Run();
}

TEST(CropAndResizeTest, CropAndResize_2122) {
  OpTester test1 ("CropAndResize", 1, onnxruntime::kMSDomain);
  test1.AddInput  <float> ("X",   {2, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f});
  test1.AddInput  <float> ("rois", {3, 4},   {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test1.AddInput  <int32_t> ("batch_indices",   {3},   {0, 1, 1});
  test1.AddInput  <int32_t> ("crop_size",    {2},    {1, 1});
  test1.AddOutput <float> ("output",  {3, 1, 1, 1}, {2.75f, 6.325f, 6.6f});
  test1.Run();

  OpTester test2 ("CropAndResize", 1, onnxruntime::kMSDomain);
  test2.AddInput  <float> ("X",   {2, 1, 2, 2},  {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f});
  test2.AddInput  <float> ("rois",  {3, 4},   {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test2.AddInput  <int32_t> ("batch_indices",   {3},   {0, 1, 1});
  test2.AddInput  <int32_t> ("crop_size",    {2},    {2, 2});
  test2.AddOutput <float> ("output", {3, 1, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.05f, 6.6f, 7.15f, 5.5f, 6.6f, 6.6f, 7.7f});
  test2.Run();
}

TEST(CropAndResizeTest, CropAndResize_1222) {
  OpTester test1 ("CropAndResize", 1, onnxruntime::kMSDomain);
  test1.AddInput  <float> ("X",   {1, 2, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f});
  test1.AddInput  <float> ("rois", {3, 4},   {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test1.AddInput  <int32_t> ("batch_indices",   {3},   {0, 0, 0});
  test1.AddInput  <int32_t> ("crop_size",    {2},    {1, 1});
  test1.AddOutput <float> ("output", {3, 2, 1, 1},  {2.75f, 7.15f, 1.925f, 6.325f, 2.2f, 6.6f});
  test1.Run();

  OpTester test2 ("CropAndResize", 1, onnxruntime::kMSDomain);
  test2.AddInput  <float> ("X",   {1, 2, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f});
  test2.AddInput  <float> ("rois", {3, 4},   {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
  test2.AddInput  <int32_t> ("batch_indices",   {3},   {0, 0, 0});
  test2.AddInput  <int32_t> ("crop_size",    {2},    {2, 2});
  test2.AddOutput <float> ("output", {3, 2, 2, 2}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 1.1f, 1.65f, 2.2f, 2.75f, \
                                                 5.5f, 6.05f, 6.6f, 7.15f, 1.1f, 2.2f, 2.2f, 3.3f, 5.5f, 6.6f, 6.6f, 7.7f});
  test2.Run();
}

TEST(CropAndResizeTest, CropAndResize_1133) {
	OpTester test1("CropAndResize", 1, onnxruntime::kMSDomain);
	test1.AddInput  <float>("X", {1, 1, 3, 3}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f});
	test1.AddInput  <float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
	test1.AddInput  <int32_t>("batch_indices", {3}, {0, 0, 0});
	test1.AddInput  <int32_t> ("crop_size",    {2},    {1, 1});
	test1.AddOutput <float>("output", {3, 1, 1, 1}, {5.5f, 3.3f, 3.85f});
	test1.Run();

	OpTester test2("CropAndResize", 1, onnxruntime::kMSDomain);
	test2.AddInput  <float>("X", {1, 1, 3, 3}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f});
	test2.AddInput  <float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
	test2.AddInput  <int32_t>("batch_indices", {3}, {0, 0, 0});
	test2.AddInput  <int32_t>("crop_size",    {2},    {2, 2});
	test2.AddOutput <float>("output", {3, 1, 2, 2}, {1.1f, 3.3f, 7.7f, 9.9f, 1.1f, 2.2f, 4.4f, 5.5f, 1.1f, 3.3f, 4.4f, 6.6f});
	test2.Run();

	OpTester test3("CropAndResize", 1, onnxruntime::kMSDomain);
	test3.AddInput  <float>("X", {1, 1, 3, 3}, {1.1f, 2.2f, 3.3f, 4.4f, 5.5f, 6.6f, 7.7f, 8.8f, 9.9f});
	test3.AddInput  <float>("rois", {3, 4}, {0.0f, 0.0f, 1.0f, 1.0f, 0.0f, 0.0f, 0.5f, 0.5f, 0.0f, 0.0f, 0.5f, 1.0f});
	test3.AddInput  <int32_t>("batch_indices", {3}, {0, 0, 0});
	test3.AddInput  <int32_t>("crop_size",    {2},    {2, 2});
	test3.AddAttribute("mode", "nearest");
	test3.AddOutput <float>("output", {3, 1, 2, 2}, {1.1f, 3.3f, 7.7f, 9.9f, 1.1f, 2.2f, 4.4f, 5.5f, 1.1f, 3.3f, 4.4f, 6.6f});
	test3.Run();
}



}  // namespace Test
}  // namespace onnxruntime

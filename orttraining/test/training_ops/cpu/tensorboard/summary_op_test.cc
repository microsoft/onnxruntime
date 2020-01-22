// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"

#include "tensorboard/compat/proto/summary.pb.h"

namespace onnxruntime {
namespace test {

static tensorboard::Summary CreateSummary(const std::vector<std::string>& tags, const std::vector<float>& values) {
  tensorboard::Summary summary;
  EXPECT_EQ(tags.size(), values.size());
  for (size_t i = 0; i < tags.size() && i < values.size(); i++) {
    auto* summary_value = summary.add_value();
    summary_value->set_tag(tags[i]);
    summary_value->set_simple_value(values[i]);
  }
  return summary;
}

TEST(SummaryOpTest, SummaryScalarOp_Tags_Missing) {
  OpTester test("SummaryScalar", 1, onnxruntime::kMSDomain);

  std::vector<float> X = {0.85f, 0.13f};
  const int64_t N = static_cast<int64_t>(X.size());
  tensorboard::Summary summary;

  // Attribute 'tags' is missing
  test.AddInput("input", {N}, X);
  test.AddOutput("summary", {}, {summary.SerializeAsString()});
  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(SummaryOpTest, SummaryScalarOp_Input_Incorrect_Shape) {
  OpTester test("SummaryScalar", 1, onnxruntime::kMSDomain);

  std::vector<std::string> tags = {"precision", "loss"};
  std::vector<float> X = {0.85f, 0.13f, 1.0f}; // input size doesn't match tags size
  const int64_t N = static_cast<int64_t>(X.size());
  tensorboard::Summary summary;

  test.AddAttribute("tags", tags);
  test.AddInput("input", {N}, X);
  test.AddOutput("summary", {}, {summary.SerializeAsString()});
  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(SummaryOpTest, SummaryScalarOp_Valid) {
  OpTester test("SummaryScalar", 1, onnxruntime::kMSDomain);

  std::vector<std::string> tags = {"precision", "loss"};
  std::vector<float> X = {0.85f, 0.13f};
  const int64_t N = static_cast<int64_t>(X.size());
  tensorboard::Summary summary = CreateSummary(tags, X);

  test.AddAttribute("tags", tags);
  test.AddInput("input", {N}, X);
  test.AddOutput("summary", {}, {summary.SerializeAsString()});
  test.Run();
}

TEST(SummaryOpTest, SummaryHistogramOp_Tag_Missing) {
  OpTester test("SummaryHistogram", 1, onnxruntime::kMSDomain);

  std::vector<double> X = {-9.0, -2.0, 0.0, 2.0, 3.0, 9.0};
  const int64_t N = static_cast<int64_t>(X.size());
  tensorboard::Summary summary;

  // Attribute 'tag' is missing
  test.AddInput("input", {N}, X);
  test.AddOutput("summary", {}, {summary.SerializeAsString()});
  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(SummaryOpTest, SummaryHistogramOp_Valid) {
  OpTester test("SummaryHistogram", 1, onnxruntime::kMSDomain, /*verify_output:*/ false);

  const std::string tag = "histogram";
  std::vector<double> X = {-9.0, -2.0, 0.0, 2.0, 3.0, 9.0};
  const int64_t N = static_cast<int64_t>(X.size());
  tensorboard::Summary summary;

  test.AddAttribute("tag", tag);
  test.AddInput("input", {N}, X);
  test.AddOutput<std::string>("summary", {}, {summary.SerializeAsString()});
  test.Run();
}

TEST(SummaryOpTest, SummaryMergeOp_SingleInput) {
  OpTester test("SummaryMerge", 1, onnxruntime::kMSDomain);

  tensorboard::Summary summary = CreateSummary({"tag"}, {1.0f});

  test.AddInput("input_0", {}, {summary.SerializeAsString()});
  test.AddOutput<std::string>("summary", {}, {summary.SerializeAsString()});
  test.Run();
}

TEST(SummaryOpTest, SummaryMergeOp_MultipleInput) {
  OpTester test("SummaryMerge", 1, onnxruntime::kMSDomain);

  tensorboard::Summary summary0 = CreateSummary({"tag0"}, {0.0f});
  tensorboard::Summary summary1 = CreateSummary({"tag1"}, {1.0f});
  tensorboard::Summary summary2 = CreateSummary({"tag2"}, {2.0f});
  tensorboard::Summary expected = CreateSummary({"tag0", "tag1", "tag2"}, {0.0f, 1.0f, 2.0f});

  test.AddInput("input_0", {}, {summary0.SerializeAsString()});
  test.AddInput("input_1", {}, {summary1.SerializeAsString()});
  test.AddInput("input_2", {}, {summary2.SerializeAsString()});
  test.AddOutput<std::string>("summary", {}, {expected.SerializeAsString()});
  test.Run();
}

TEST(SummaryOpTest, SummaryMergeOp_DuplicateTag) {
  OpTester test("SummaryMerge", 1, onnxruntime::kMSDomain);

  tensorboard::Summary summary0 = CreateSummary({"tag"}, {0.0f});
  tensorboard::Summary summary1 = CreateSummary({"tag"}, {1.0f});
  tensorboard::Summary expected = CreateSummary({"tag", "tag"}, {0.0f, 1.0f});

  test.AddInput("input_0", {}, {summary0.SerializeAsString()});
  test.AddInput("input_1", {}, {summary1.SerializeAsString()});
  test.AddOutput<std::string>("summary", {}, {expected.SerializeAsString()});
  test.Run(OpTester::ExpectResult::kExpectFailure, "duplicate tag");
}

TEST(SummaryOpTest, SummaryTextOp_Tag_Missing) {
  OpTester test("SummaryText", 1, onnxruntime::kMSDomain);

  std::vector<std::string> X = {"text 0", "text 1", "text 2", "text 3"};
  const int64_t N = static_cast<int64_t>(X.size());
  tensorboard::Summary summary;

  // Attribute 'tag' is missing
  test.AddInput("input", {N}, X);
  test.AddOutput("summary", {}, {summary.SerializeAsString()});
  test.Run(OpTester::ExpectResult::kExpectFailure);
}

TEST(SummaryOpTest, SummaryTextOp_1D_Tensor) {
  OpTester test("SummaryText", 1, onnxruntime::kMSDomain);

  const std::string tag = "text";
  std::vector<std::string> X = {"text 0", "text 1", "text 2", "text 3"};
  const int64_t N = static_cast<int64_t>(X.size());
  tensorboard::Summary summary;
  auto* summary_value = summary.add_value();
  summary_value->set_tag(tag);
  summary_value->mutable_metadata()->mutable_plugin_data()->set_plugin_name("text");
  auto* summary_tensor = summary_value->mutable_tensor();
  summary_tensor->mutable_tensor_shape()->add_dim()->set_size(N);
  summary_tensor->set_dtype(tensorboard::DataType::DT_STRING);
  for (const std::string& s : X)
    summary_tensor->add_string_val(s);

  test.AddAttribute("tag", tag);
  test.AddInput("input", {N}, X);
  test.AddOutput<std::string>("summary", {}, {summary.SerializeAsString()});
  test.Run();
}

TEST(SummaryOpTest, SummaryTextOp_2D_Tensor) {
  OpTester test("SummaryText", 1, onnxruntime::kMSDomain);

  const std::string tag = "text";
  std::vector<std::string> X = {"text 0", "text 1", "text 2", "text 3"};
  const int64_t N = static_cast<int64_t>(X.size());
  tensorboard::Summary summary;
  auto* summary_value = summary.add_value();
  summary_value->set_tag(tag);
  summary_value->mutable_metadata()->mutable_plugin_data()->set_plugin_name("text");
  auto* summary_tensor = summary_value->mutable_tensor();
  summary_tensor->mutable_tensor_shape()->add_dim()->set_size(N/2);
  summary_tensor->mutable_tensor_shape()->add_dim()->set_size(2);
  summary_tensor->set_dtype(tensorboard::DataType::DT_STRING);
  for (const std::string& s : X)
    summary_tensor->add_string_val(s);

  test.AddAttribute("tag", tag);
  test.AddInput("input", {N/2, 2}, X);
  test.AddOutput<std::string>("summary", {}, {summary.SerializeAsString()});
  test.Run();
}

}  // namespace test
}  // namespace onnxruntime

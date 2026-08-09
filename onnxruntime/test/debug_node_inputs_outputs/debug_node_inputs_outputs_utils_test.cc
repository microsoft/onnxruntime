// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/debug_node_inputs_outputs_utils.h"

#include <fstream>

#include "gtest/gtest.h"

#include "core/framework/tensorprotoutils.h"
#include "core/platform/env.h"
#include "core/platform/path_lib.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/scoped_env_vars.h"
#include "test/util/include/temp_dir.h"

namespace onnxruntime {
namespace test {

namespace {
template <typename T>
void VerifyTensorProtoFileData(const PathString& tensor_proto_path, gsl::span<const T> expected_data) {
  std::ifstream tensor_proto_stream{tensor_proto_path};

  ONNX_NAMESPACE::TensorProto tensor_proto{};
  ASSERT_TRUE(tensor_proto.ParseFromIstream(&tensor_proto_stream));

  std::vector<T> actual_data{};
  actual_data.resize(expected_data.size());
  ASSERT_STATUS_OK(utils::UnpackTensor(tensor_proto, tensor_proto_path, actual_data.data(), actual_data.size()));

  ASSERT_EQ(gsl::span<const T>(actual_data), expected_data);
}

template <bool Signed>
void VerifyTensorProtoFileDataInt4(const PathString& tensor_proto_path,
                                   gsl::span<const Int4x2Base<Signed>> expected_data,
                                   gsl::span<const int64_t> shape) {
  size_t num_elems = 1;
  for (auto dim_val : shape) {
    num_elems *= static_cast<size_t>(dim_val);
  }

  std::ifstream tensor_proto_stream{tensor_proto_path};

  ONNX_NAMESPACE::TensorProto tensor_proto{};
  ASSERT_TRUE(tensor_proto.ParseFromIstream(&tensor_proto_stream));

  std::vector<Int4x2Base<Signed>> actual_data{};
  actual_data.resize(expected_data.size());
  ASSERT_STATUS_OK(utils::UnpackTensor(tensor_proto, tensor_proto_path, actual_data.data(), num_elems));

  ASSERT_EQ(actual_data.size(), expected_data.size());

  for (size_t i = 0; i < num_elems; i++) {
    auto indices = Int4x2Base<Signed>::GetTensorElemIndices(i);
    auto actual_val = actual_data[indices.first].GetElem(indices.second);
    auto expected_val = expected_data[indices.first].GetElem(indices.second);
    ASSERT_EQ(actual_val, expected_val);
  }
}
}  // namespace

namespace env_vars = utils::debug_node_inputs_outputs_env_vars;

TEST(DebugNodeInputsOutputs, BasicFileOutput) {
  TemporaryDirectory temp_dir{ORT_TSTR("debug_node_inputs_outputs_utils_test")};
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {env_vars::kDumpInputData, "1"},
          {env_vars::kDumpOutputData, "1"},
          {env_vars::kNameFilter, nullopt},
          {env_vars::kOpTypeFilter, nullopt},
          {env_vars::kDumpDataDestination, "files"},
          {env_vars::kAppendRankToFileName, nullopt},
          {env_vars::kOutputDir, ToUTF8String(temp_dir.Path())},
          {env_vars::kDumpingDataToFilesForAllNodesIsOk, "1"},
      }};

  OpTester tester{"Round", 11, kOnnxDomain};
  const std::vector<float> input{0.9f, 1.8f};
  tester.AddInput<float>("x", {static_cast<int64_t>(input.size())}, input);
  const std::vector<float> output{1.0f, 2.0f};
  tester.AddOutput<float>("y", {static_cast<int64_t>(output.size())}, output);

  auto verify_file_data =
      [&temp_dir, &input, &output](
          const std::vector<OrtValue>& fetches,
          const std::string& /*provider_type*/) {
        ASSERT_EQ(fetches.size(), 1u);
        // check it contains a tensor
        fetches[0].Get<Tensor>();
        VerifyTensorProtoFileData(temp_dir.Path() + ORT_TSTR("/x.tensorproto"), gsl::make_span(input));
        VerifyTensorProtoFileData(temp_dir.Path() + ORT_TSTR("/y.tensorproto"),
                                  gsl::make_span(output));
      };

  tester.SetCustomOutputVerifier(verify_file_data);

  tester.Run();
}

// Test dumping input and output INT4 tensors to file.
TEST(DebugNodeInputsOutputs, FileOutput_Int4) {
  TemporaryDirectory temp_dir{ORT_TSTR("debug_node_inputs_outputs_utils_test")};
  ScopedEnvironmentVariables scoped_env_vars{
      EnvVarMap{
          {env_vars::kDumpInputData, "1"},
          {env_vars::kDumpOutputData, "1"},
          {env_vars::kNameFilter, nullopt},
          {env_vars::kOpTypeFilter, nullopt},
          {env_vars::kDumpDataDestination, "files"},
          {env_vars::kAppendRankToFileName, nullopt},
          {env_vars::kOutputDir, ToUTF8String(temp_dir.Path())},
          {env_vars::kDumpingDataToFilesForAllNodesIsOk, "1"},
      }};

  constexpr int8_t unused_val = 0;
  const std::vector<int64_t> input_shape({5, 3});
  const std::vector<Int4x2> input_vals = {Int4x2(1, 2), Int4x2(3, 4), Int4x2(5, 6), Int4x2(7, 8),
                                          Int4x2(9, 10), Int4x2(11, 12), Int4x2(13, 14), Int4x2(15, unused_val)};

  const std::vector<int64_t> perm = {1, 0};
  const std::vector<int64_t> expected_shape({3, 5});
  const std::vector<Int4x2> expected_vals = {Int4x2(1, 4), Int4x2(7, 10), Int4x2(13, 2), Int4x2(5, 8),
                                             Int4x2(11, 14), Int4x2(3, 6), Int4x2(9, 12), Int4x2(15, unused_val)};

  OpTester tester{"Transpose", 21, kOnnxDomain};
  tester.AddAttribute("perm", perm);
  tester.AddInput<Int4x2>("x", input_shape, input_vals);
  tester.AddOutput<Int4x2>("y", expected_shape, expected_vals);

  auto verify_file_data =
      [&temp_dir, &input_vals, &expected_vals, &input_shape, &expected_shape](
          const std::vector<OrtValue>& fetches,
          const std::string& /*provider_type*/) {
        ASSERT_EQ(fetches.size(), 1u);
        // check it contains a tensor
        fetches[0].Get<Tensor>();
        VerifyTensorProtoFileDataInt4(temp_dir.Path() + ORT_TSTR("/x.tensorproto"), gsl::make_span(input_vals),
                                      gsl::make_span(input_shape));
        VerifyTensorProtoFileDataInt4(temp_dir.Path() + ORT_TSTR("/y.tensorproto"),
                                      gsl::make_span(expected_vals), gsl::make_span(expected_shape));
      };

  tester.SetCustomOutputVerifier(verify_file_data);

  tester.Run();
}

}  // namespace test
}  // namespace onnxruntime

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
  ASSERT_STATUS_OK(utils::UnpackTensor(tensor_proto, Path{}, actual_data.data(), actual_data.size()));

  ASSERT_EQ(gsl::span<const T>(actual_data), expected_data);
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

}  // namespace test
}  // namespace onnxruntime

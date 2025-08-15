// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <filesystem>
#include <string>

#include "core/framework/provider_options.h"
#include "core/framework/tensor_shape.h"
#include "core/framework/float16.h"

#include "test/util/include/test_utils.h"
#include "test/util/include/test/test_environment.h"
#include "test/util/include/default_providers.h"
#include "test/util/include/qdq_test_utils.h"

#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/inference_session.h"
#include "core/graph/model_saving_options.h"

#include "gtest/gtest.h"
#include "gmock/gmock.h"

using namespace ONNX_NAMESPACE;
using namespace onnxruntime::logging;

extern std::unique_ptr<Ort::Env> ort_env;

class OVEPEPContextTests : public ::testing::Test {
};

namespace onnxruntime {
namespace test {

// Test if folder path given to ep_context_file_path throws an error
TEST_F(OVEPEPContextTests, OVEPEPContextFolderPath) {
  Ort::SessionOptions sessionOptions;
  std::unordered_map<std::string, std::string> ov_options;

  // The below line could fail the test in non NPU platforms.Commenting it out so that the device used for building OVEP will be used.
  // ov_options["device_type"] = "NPU";

  const std::unordered_map<std::string, int> domain_to_version = {{"", 13}, {kMSDomain, 1}};

  auto& logging_manager = DefaultLoggingManager();
  logging_manager.SetDefaultLoggerSeverity(logging::Severity::kERROR);

  onnxruntime::Model model("OVEP_Test_Model", false, ModelMetaData(), PathString(),
                           IOnnxRuntimeOpSchemaRegistryList(), domain_to_version, {},
                           logging_manager.DefaultLogger());

  ASSERT_STATUS_OK(model.MainGraph().Resolve());

  // Serialize the model to a string.
  std::string model_data;
  model.ToProto().SerializeToString(&model_data);

  const auto model_data_span = AsByteSpan(model_data.data(), model_data.size());

  const std::string ep_context_file_path = "./ep_context_folder_path/";

  sessionOptions.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
  sessionOptions.AddConfigEntry(kOrtSessionOptionEpContextFilePath, ep_context_file_path.c_str());
  sessionOptions.AppendExecutionProvider_OpenVINO_V2(ov_options);

  try {
    Ort::Session session(*ort_env, model_data_span.data(), model_data_span.size(), sessionOptions);
    FAIL();  // Should not get here!
  } catch (const Ort::Exception& excpt) {
    ASSERT_EQ(excpt.GetOrtErrorCode(), ORT_INVALID_ARGUMENT);
    ASSERT_THAT(excpt.what(), testing::HasSubstr("context_file_path should not point to a folder."));
  }
}

}  // namespace test
}  // namespace onnxruntime

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include "test_configuration.h"
#include <core/session/onnxruntime_c_api.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <random>
#include "command_args_parser.h"
#include <google/protobuf/stubs/common.h>

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/inference_session.h"
#include "core/session/ort_env.h"
#include "core/providers/provider_factory_creators.h"
#include "core/common/logging/sinks/clog_sink.h"

using namespace onnxruntime;
const OrtApi* g_ort = NULL;
std::unique_ptr<Ort::Env> ort_env;

void CheckStatus(const Status& status) {
  if (status.Code() != common::StatusCode::OK) {
    std::string msg = status.ErrorMessage();
    throw Ort::Exception(std::move(msg), OrtErrorCode::ORT_EP_FAIL);
  }
}

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[]) {
#else
int real_main(int argc, char* argv[]) {
#endif
  g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  qnnctxgen::PerformanceTestConfig test_config;
  if (!qnnctxgen::CommandLineParser::ParseArguments(test_config, argc, argv)) {
    qnnctxgen::CommandLineParser::ShowUsage();
    return -1;
  }

  {
    bool failed = false;
    ORT_TRY {
      OrtLoggingLevel logging_level = test_config.run_config.f_verbose
                                          ? ORT_LOGGING_LEVEL_VERBOSE
                                          : ORT_LOGGING_LEVEL_WARNING;

      ort_env = std::make_unique<Ort::Env>(logging_level, "Default");
    }
    ORT_CATCH(const Ort::Exception& e) {
      ORT_HANDLE_EXCEPTION([&]() {
        fprintf(stderr, "Error creating environment: %s \n", e.what());
        failed = true;
      });
    }

    if (failed)
      return -1;
  }

  SessionOptions so;
  so.session_logid = "qnn_ctx_gen_session_logger";
  // Set default session option to dump QNN context model with non-embed mode
  CheckStatus(so.config_options.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1"));
  CheckStatus(so.config_options.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0"));
  RunOptions run_options;
  run_options.run_tag = so.session_logid;

  ProviderOptions provider_options;
#if defined(_WIN32)
  provider_options["backend_path"] = "QnnHtp.dll";
#else
  provider_options["backend_path"] = "libQnnHtp.so";
#endif
  // set default QNN EP option to enable weight sharing
  provider_options["enable_htp_weight_sharing"] = "1";

  for (auto it : test_config.run_config.qnn_options) {
    provider_options[it.first] = it.second;
  }

  for (auto it : test_config.run_config.session_config_entries) {
    CheckStatus(so.config_options.AddConfigEntry(it.first.c_str(), it.second.c_str()));
  }

  {
    std::shared_ptr<IExecutionProvider> qnn_ep(QNNProviderFactoryCreator::Create(provider_options, &so)->CreateProvider());

    for (auto model_path : test_config.model_file_paths) {
      InferenceSession session_object1{so, ((OrtEnv*)*ort_env.get())->GetEnvironment()};
      CheckStatus(session_object1.RegisterExecutionProvider(qnn_ep));
      CheckStatus(session_object1.Load(ToPathString(model_path)));
      CheckStatus(session_object1.Initialize());
    }
  }
  ort_env.reset();

  return 0;
}

#ifdef _WIN32
int wmain(int argc, wchar_t* argv[]) {
#else
int main(int argc, char* argv[]) {
#endif
  int retval = -1;
  ORT_TRY {
    retval = real_main(argc, argv);
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      fprintf(stderr, "%s\n", ex.what());
      retval = -1;
    });
  }

  ::google::protobuf::ShutdownProtobufLibrary();

  return retval;
}

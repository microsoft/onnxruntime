// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// onnxruntime dependencies
#include "test_configuration.h"
#include <core/session/onnxruntime_cxx_api.h>
#include "command_args_parser.h"
#include <google/protobuf/stubs/common.h>

#include "core/session/onnxruntime_session_options_config_keys.h"
#include "core/session/ort_env.h"

#include "core/graph/model.h"
#include "core/session/environment.h"

using namespace onnxruntime;
using ProviderOptions = std::unordered_map<std::string, std::string>;

std::unique_ptr<Ort::Env> ort_env;

static void CheckStatus(const Status& status) {
  if (status.Code() != common::StatusCode::OK) {
    std::string msg = status.ErrorMessage();
    throw Ort::Exception(std::move(msg), OrtErrorCode::ORT_FAIL);
  }
}

static int64_t GetNodeAttr(const Node& node, const std::string& attr_name, int64_t default_val) {
  const auto& attributes = node.GetAttributes();
  if (auto entry = attributes.find(attr_name); entry != attributes.end()) {
    return entry->second.i();
  }

  return default_val;
}

static const std::string& GetNodeAttr(const Node& node, const std::string& attr_name, const std::string& default_val) {
  const auto& attributes = node.GetAttributes();
  if (auto entry = attributes.find(attr_name); entry != attributes.end()) {
    return entry->second.s();
  }

  return default_val;
}

// from the last context cache Onnx model, find the EPContext node with main_context=1,
// and get the QNN context binary file name, this context binary contains all graphs from all Onnx models
// get the max spill fill buffer size
static void GetLastContextBinaryFileName(const std::basic_string<ORTCHAR_T> last_onnx_ctx_file,
                                         std::string& last_ctx_bin_file,
                                         int64_t& max_size) {
  max_size = 0;
  std::shared_ptr<Model> ctx_model;
  CheckStatus(Model::Load(ToPathString(last_onnx_ctx_file), ctx_model, nullptr,
                          (*((OrtEnv*)*ort_env.get())->GetEnvironment().GetLoggingManager()).DefaultLogger()));
  auto& ctx_graph = ctx_model->MainGraph();
  for (auto& node : ctx_graph.Nodes()) {
    if (node.OpType() == "EPContext") {
      int64_t is_main_context = GetNodeAttr(node, "main_context", static_cast<int64_t>(0));
      max_size = GetNodeAttr(node, "max_size", static_cast<int64_t>(0));
      if (1 == is_main_context) {
        last_ctx_bin_file = GetNodeAttr(node, "ep_cache_context", "");
        return;
      }
    }
  }
}

// Update generated context cache Onnx model to make the main EPContext node point to
// the last QNN context binary file
// Remove not used QNN context binary file, only keep the last one which contains all graphs
static void UpdateEpContextModel(const std::vector<std::basic_string<ORTCHAR_T>>& ep_ctx_files,
                                 const std::string& last_qnn_ctx_binary_file_name,
                                 int64_t max_size) {
  for (auto ep_ctx_file : ep_ctx_files) {
    std::shared_ptr<Model> ctx_model;
    auto path_str = ToPathString(ep_ctx_file);
    CheckStatus(Model::Load(path_str, ctx_model, nullptr,
                            (*((OrtEnv*)*ort_env.get())->GetEnvironment().GetLoggingManager()).DefaultLogger()));
    auto& ctx_graph = ctx_model->MainGraph();
    GraphViewer graph_viewer(ctx_graph);
    auto path = std::filesystem::path(path_str);

    for (auto& node : ctx_graph.Nodes()) {
      if (node.OpType() == "EPContext") {
        int64_t is_main_context = GetNodeAttr(node, "main_context", static_cast<int64_t>(0));
        if (1 == is_main_context) {
          std::string old_qnn_ctx_binary_file_name = GetNodeAttr(node, "ep_cache_context", "");
          auto file_path = path.replace_filename(old_qnn_ctx_binary_file_name);
          std::remove(file_path.string().c_str());
          node.ClearAttribute("ep_cache_context");
          node.AddAttribute("ep_cache_context", last_qnn_ctx_binary_file_name);
          node.ClearAttribute("max_size");
          node.AddAttribute("max_size", max_size);
        }
      }
    }
    std::remove(ToUTF8String(ep_ctx_file).c_str());
    CheckStatus(Model::Save(*ctx_model.get(), ToPathString(ep_ctx_file)));
  }
}

#ifdef _WIN32
int real_main(int argc, wchar_t* argv[]) {
#else
int real_main(int argc, char* argv[]) {
#endif
  qnnctxgen::TestConfig test_config;
  if (!qnnctxgen::CommandLineParser::ParseArguments(test_config, argc, argv)) {
    qnnctxgen::CommandLineParser::ShowUsage();
    return -1;
  }

  OrtLoggingLevel logging_level = test_config.run_config.f_verbose
                                      ? ORT_LOGGING_LEVEL_VERBOSE
                                      : ORT_LOGGING_LEVEL_WARNING;
  Ort::Env env(logging_level, "ep_weight_sharing");

  ORT_TRY {
    Ort::SessionOptions so;
    so.SetLogId("ep_weight_sharing_ctx_gen_session_logger");
    // Set default session option to dump QNN context model with non-embed mode
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");
    so.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

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
      if (it.first == kOrtSessionOptionEpContextEnable && it.second != "1") {
        std::cerr << "Need to enable ep context cache." << std::endl;
        continue;
      }
      if (it.first == kOrtSessionOptionEpContextEmbedMode && it.second != "0") {
        std::cerr << "Only support non-embed model for weight sharing." << std::endl;
        continue;
      }
      if (it.first == kOrtSessionOptionEpContextFilePath) {
        std::cout << "Not support to specify the generated Onnx context cache file name." << std::endl;
        continue;
      }
      so.AddConfigEntry(it.first.c_str(), it.second.c_str());
    }

    for (auto model_path : test_config.model_file_paths) {
      std::cout << "Model file path: " << ToUTF8String(model_path) << std::endl;
    }

    // Generate context cache model files with QNN context binary files
    // The context binary file generated later includes all graphs from previous models
    {
      so.AppendExecutionProvider("QNN", provider_options);

      for (auto model_path : test_config.model_file_paths) {
        std::cout << "Generate context cache model for: " << ToUTF8String(model_path) << std::endl;
        Ort::Session session(env, model_path.c_str(), so);
      }
    }

    std::cout << "Start to update the generated Onnx model." << std::endl;
    std::vector<std::basic_string<ORTCHAR_T>> ep_ctx_files;
    ep_ctx_files.reserve(test_config.model_file_paths.size());
    for (auto model_path : test_config.model_file_paths) {
      ep_ctx_files.push_back(model_path + ORT_TSTR("_ctx.onnx"));
    }

    // Get the last context binary file name
    std::string last_qnn_ctx_binary_file_name;
    int64_t max_size = 0;
    GetLastContextBinaryFileName(ep_ctx_files.back(), last_qnn_ctx_binary_file_name, max_size);
    std::cout << "The last context binary file: " << last_qnn_ctx_binary_file_name << std::endl;
    if (last_qnn_ctx_binary_file_name.empty()) {
      throw Ort::Exception("Can't find QNN context binary file from the Onnx model.", OrtErrorCode::ORT_FAIL);
    }
    ep_ctx_files.pop_back();

    // Update generated context cache Onnx model to make the main EPContext node point to
    // the last QNN context binary file
    // Remove not used QNN context binary file, only keep the last one which contains all graphs
    UpdateEpContextModel(ep_ctx_files, last_qnn_ctx_binary_file_name, max_size);
  }
  ORT_CATCH(const Ort::Exception& e) {
    fprintf(stderr, "Failed to generate context cache file: %s \n", e.what());

    ort_env.reset();
    return -1;
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

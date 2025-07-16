// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "test_configuration.h"
#include "command_args_parser.h"

// onnxruntime dependencies
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/onnxruntime_session_options_config_keys.h"

// onnx dependencies
#include "onnx/onnx_pb.h"
#include <fstream>

using namespace onnxruntime;
using ProviderOptions = std::unordered_map<std::string, std::string>;

// from the last context cache Onnx model, find the EPContext node with main_context=1,
// get the max spill fill buffer size
static void GetEpContextInfoFromLastContextModel(const std::basic_string<ORTCHAR_T> last_onnx_ctx_file,
                                                 int64_t& max_size) {
  max_size = 0;

  onnx::ModelProto model;
  std::ifstream onnx_file_stream(last_onnx_ctx_file, std::ios_base::binary);
  model.ParseFromIstream(&onnx_file_stream);

  for (auto& node : model.graph().node()) {
    if (node.op_type() == "EPContext") {
      int64_t is_main_context = 0;
      for (auto& attr : node.attribute()) {
        if (attr.name() == "main_context") {
          is_main_context = attr.i();
        }
        if (attr.name() == "max_size") {
          max_size = attr.i();
        }
      }
      if (is_main_context) {
        return;
      }
    }
  }

  onnx_file_stream.close();
}

// Update generated context cache Onnx model to have the same max_size (align with the last generated model)
static void UpdateEpContextModel(const std::vector<std::basic_string<ORTCHAR_T>>& ep_ctx_files,
                                 int64_t max_size) {
  for (auto ep_ctx_file : ep_ctx_files) {
    onnx::ModelProto model;
    std::ifstream onnx_file_stream(ep_ctx_file, std::ios_base::binary);
    model.ParseFromIstream(&onnx_file_stream);
    onnx_file_stream.close();

    for (auto& node : *(model.mutable_graph()->mutable_node())) {
      if (node.op_type() == "EPContext") {
        int64_t is_main_context = 0;
        int max_size_index = 0;
        for (auto i = 0; i < node.attribute_size(); ++i) {
          auto& attr = node.attribute()[i];
          if (attr.name() == "main_context") {
            is_main_context = attr.i();
          }
          if (attr.name() == "max_size") {
            max_size = attr.i();
            max_size_index = i;
          }
        }
        if (is_main_context) {
          node.mutable_attribute(max_size_index)->set_i(max_size);
        }
      }
    }

    // re-write the onnx ctx file
    std::ofstream onnx_file_ostream(ep_ctx_file, std::ios_base::binary);
    model.SerializeToOstream(&onnx_file_ostream);
    onnx_file_ostream.close();
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
                                      : ORT_LOGGING_LEVEL_ERROR;
  Ort::Env env(logging_level, "ep_weight_sharing");

  ORT_TRY {
    Ort::SessionOptions so;
    so.SetLogId("ep_weight_sharing_ctx_gen_session_logger");
    // Set default session option to dump EPContext model with non-embed mode
    so.AddConfigEntry(kOrtSessionOptionEpContextEnable, "1");
    so.AddConfigEntry(kOrtSessionOptionEpContextEmbedMode, "0");
    // enable ep.share_ep_contexts
    so.AddConfigEntry(kOrtSessionOptionShareEpContexts, "1");

    ProviderOptions provider_options;

    for (auto it : test_config.run_config.provider_options) {
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
      std::string provider_name_ = test_config.machine_config.provider_type_name;
      if (provider_name_ == onnxruntime::kQnnExecutionProvider) {
#ifdef USE_QNN
        so.AppendExecutionProvider("QNN", provider_options);
#else
        ORT_THROW("QNN is not supported in this build\n");
#endif
      } else if (!provider_name_.empty()) {
        ORT_THROW("This execution provider is not included in this tool.\n");
      }

      size_t total_file_count = test_config.model_file_paths.size();
      for (size_t i = 0; i < total_file_count; ++i) {
        auto model_path = test_config.model_file_paths[i];
        std::cout << "Generating context cache model for: " << ToUTF8String(model_path) << std::endl;
        if (i == total_file_count - 1) {
          so.AddConfigEntry(kOrtSessionOptionStopShareEpContexts, "1");
        }
        Ort::Session session(env, model_path.c_str(), so);
      }
    }

    // Only with enable_htp_spill_fill_buffer enabled:
    // Update generated context cache Onnx model to have the same max_size (align with the last generated model)
    // so that the inference session can be created with any order of the ctx.onnx models
    const std::string enable_htp_spill_fill_buffer = "enable_htp_spill_fill_buffer";
    auto pos = provider_options.find(enable_htp_spill_fill_buffer);
    if (pos != provider_options.end() && pos->second == "1") {
      std::cout << "Start to update the generated Onnx model to reflect the max_size." << std::endl;

      // The steps below only used for spill fill buffer enabled
      std::vector<std::basic_string<ORTCHAR_T>> ep_ctx_files;
      ep_ctx_files.reserve(test_config.model_file_paths.size());
      for (auto model_path : test_config.model_file_paths) {
        auto dot_pos = model_path.find_last_of(ORT_TSTR("."));
        if (dot_pos != std::string::npos) {
          model_path = model_path.substr(0, dot_pos) + ORT_TSTR("_ctx.onnx");
        } else {
          model_path = model_path + ORT_TSTR("_ctx.onnx");
        }
        ep_ctx_files.push_back(model_path);
      }

      int64_t max_size = 0;
      GetEpContextInfoFromLastContextModel(ep_ctx_files.back(), max_size);
      ep_ctx_files.pop_back();

      UpdateEpContextModel(ep_ctx_files, max_size);
    }
  }
  ORT_CATCH(const Ort::Exception& e) {
    std::cerr << "Failed to generate context cache file: " << e.what();
    return -1;
  }

  std::cout << "Generation done!";
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

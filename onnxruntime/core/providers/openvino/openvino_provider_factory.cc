// Copyright(C) 2019 Intel Corporation
// Licensed under the MIT License

#include "core/providers/openvino/openvino_provider_factory.h"
#include "openvino_execution_provider.h"
#include "core/session/abi_session_options_impl.h"

namespace onnxruntime {
struct OpenVINOProviderFactory : IExecutionProviderFactory {
  OpenVINOProviderFactory(const char* device_type, bool enable_vpu_fast_compile,
                          const char* device_id, size_t num_of_threads)
    : enable_vpu_fast_compile_(enable_vpu_fast_compile), num_of_threads_(num_of_threads) {
    device_type_ = (device_type == nullptr) ? "" : device_type;
    device_id_ = (device_id == nullptr) ? "" : device_id;
  }
  ~OpenVINOProviderFactory() override {
  }

  std::unique_ptr<IExecutionProvider> CreateProvider() override;

 private:
  std::string device_type_;
  bool enable_vpu_fast_compile_;
  std::string device_id_;
  size_t num_of_threads_;
};

std::unique_ptr<IExecutionProvider> OpenVINOProviderFactory::CreateProvider() {
  OpenVINOExecutionProviderInfo info(device_type_, enable_vpu_fast_compile_, device_id_, num_of_threads_);
  return std::make_unique<OpenVINOExecutionProvider>(info);
}

std::shared_ptr<IExecutionProviderFactory> CreateExecutionProviderFactory_OpenVINO(
    const char* device_type, bool enable_vpu_fast_compile, const char* device_id, size_t num_of_threads) {
  return std::make_shared<onnxruntime::OpenVINOProviderFactory>(device_type, enable_vpu_fast_compile, device_id, num_of_threads);
}

}  // namespace onnxruntime

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProvider_OpenVINO,
                    _In_ OrtSessionOptions* options, _In_ const char* device_type) {
  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_OpenVINO(device_type, false, "", 8));
  return nullptr;
}

ORT_API_STATUS_IMPL(OrtSessionOptionsAppendExecutionProviderEx_OpenVINO,
                    _In_ OrtSessionOptions* options, _In_ const char* settings_str) {

  std::string device_type = "";
  bool enable_vpu_fast_compile = false;
  std::string device_id = "";
  size_t num_of_threads = 8;

  // Parse settings string
  std::stringstream iss;
  iss << settings_str;
  std::string token;
  while (std::getline(iss, token)) {
    if(token == "") {
      continue;
    }
    auto pos = token.find("|");
    if(pos == std::string::npos || pos == 0 || pos == token.length()) {
      continue;
    }

    auto key = token.substr(0,pos);
    auto value = token.substr(pos+1);

    if ( key == "device_type") {
      device_type = value;
    } else if (key == "enable_vpu_fast_compile") {
      if(value == "true" || value == "True"){
        enable_vpu_fast_compile = true;
      }
    } else if(key == "device_id") {
      device_id = value;
    } else if (key == "num_of_threads") {
      size_t n_t = std::stoi(value);
      if((int)n_t <= 0) {
        num_of_threads = 8;
      }
      else {
        num_of_threads = n_t;
      }
    }

  }

  options->provider_factories.push_back(
      onnxruntime::CreateExecutionProviderFactory_OpenVINO(device_type.c_str(),
                                                           enable_vpu_fast_compile,
                                                           device_id.c_str(),
                                                           num_of_threads));
  return nullptr;
}

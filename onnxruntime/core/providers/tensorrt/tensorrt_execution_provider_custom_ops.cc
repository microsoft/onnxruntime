// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "core/framework/provider_options.h"
#include "tensorrt_execution_provider_custom_ops.h"
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <iostream>
#include <unordered_set>

namespace onnxruntime {
class TensorrtLogger : public nvinfer1::ILogger {
  nvinfer1::ILogger::Severity verbosity_;

 public:
  TensorrtLogger(Severity verbosity = Severity::kWARNING)
      : verbosity_(verbosity) {}
  void log(Severity severity, const char* msg) noexcept override {
    if (severity <= verbosity_) {
      time_t rawtime = std::time(0);
      struct tm stm;
#ifdef _MSC_VER
      gmtime_s(&stm, &rawtime);
#else
      gmtime_r(&rawtime, &stm);
#endif
      char buf[256];
      strftime(&buf[0], 256,
               "%Y-%m-%d %H:%M:%S",
               &stm);
      const char* sevstr = (severity == Severity::kINTERNAL_ERROR ? "    BUG" : severity == Severity::kERROR ? "  ERROR"
                                                                            : severity == Severity::kWARNING ? "WARNING"
                                                                            : severity == Severity::kINFO    ? "   INFO"
                                                                                                             : "UNKNOWN");
      if (severity <= Severity::kERROR) {
        std::cout << "[" << buf << " " << sevstr << "] " << msg << std::endl;
      } else {
        std::cout << "[" << buf << " " << sevstr << "] " << msg;
      }
    }
  }
};
TensorrtLogger& GetTensorrtLogger2() {
  static TensorrtLogger trt_logger2(nvinfer1::ILogger::Severity::kWARNING);
  return trt_logger2;
}


void IterateTensorRTPluginFields(const nvinfer1::PluginFieldCollection* plugin_field_collection) {
  if (plugin_field_collection == nullptr) {
    return;
  }
  std::cout << "plugin fields:" << std::endl;
  for (int i = 0; i < plugin_field_collection->nbFields; ++i) {
    auto plugin_field = plugin_field_collection->fields[i];
    std::string plugin_field_name(plugin_field.name);
    std::cout << "\t" << plugin_field_name << std::endl;
  }  
}

int CreateTensorRTCustomOpDomain(OrtProviderCustomOpDomain** domain) {
  std::unique_ptr<OrtProviderCustomOpDomain> custom_op_domain = std::make_unique<OrtProviderCustomOpDomain>();
  custom_op_domain->domain_ = "";

  TensorrtLogger trt_logger = GetTensorrtLogger2();
  initLibNvInferPlugins(&trt_logger, "");

  int num_plugin_creator = 0;
  auto plugin_creators = getPluginRegistry()->getPluginCreatorList(&num_plugin_creator);
  std::unordered_set<std::string> registered_plugin_name; 
  
  for (int i = 0; i < num_plugin_creator; ++i) {
    auto plugin_creator = plugin_creators[i];
    std::string plugin_name(plugin_creator->getPluginName());
    auto plugin_field_collection = plugin_creator->getFieldNames();

    std::cout << plugin_name << ", version: " << plugin_creator->getPluginVersion() << std::endl;
    IterateTensorRTPluginFields(plugin_field_collection);

    if (registered_plugin_name.find(plugin_name) != registered_plugin_name.end()) {
      continue;
    }
    registered_plugin_name.insert(plugin_name);

    std::unique_ptr<TensorRTCustomOp> trt_custom_op = std::make_unique<TensorRTCustomOp>(onnxruntime::kTensorrtExecutionProvider, nullptr);
    trt_custom_op->SetName(plugin_creator->getPluginName());
    custom_op_domain->custom_ops_.push_back(trt_custom_op.release());
  }

  *domain = custom_op_domain.release();
  return 0;
}

}  // namespace onnxruntime
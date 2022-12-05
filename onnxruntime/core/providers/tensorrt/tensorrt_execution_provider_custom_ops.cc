// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "core/framework/provider_options.h"
#include "tensorrt_execution_provider_custom_ops.h"
#include "tensorrt_execution_provider.h"
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <iostream>
#include <unordered_set>

namespace onnxruntime {
extern TensorrtLogger& GetTensorrtLogger();

// This is the helper function to get the plugin fields that is currently not being used
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

common::Status CreateTensorRTCustomOpDomain(OrtProviderCustomOpDomain** domain) {
  //std::unordered_set legacy_custom_ops = {"EfficientNMS_TRT", "MultilevelCropAndResize_TRT", "PyramidROIAlign_TRT", "DisentangledAttention_TRT"};
  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Get all registered TRT plugins from registry.";
  std::unique_ptr<OrtProviderCustomOpDomain> custom_op_domain = std::make_unique<OrtProviderCustomOpDomain>();
  custom_op_domain->domain_ = "";

  TensorrtLogger trt_logger = GetTensorrtLogger();
  initLibNvInferPlugins(&trt_logger, "");

  int num_plugin_creator = 0;
  auto plugin_creators = getPluginRegistry()->getPluginCreatorList(&num_plugin_creator);
  std::unordered_set<std::string> registered_plugin_names; 
  
  for (int i = 0; i < num_plugin_creator; ++i) {
    auto plugin_creator = plugin_creators[i];
    std::string plugin_name(plugin_creator->getPluginName());
    LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] " << plugin_name << ", version : " << plugin_creator->getPluginVersion();

    //auto plugin_field_collection = plugin_creator->getFieldNames();
    //IterateTensorRTPluginFields(plugin_field_collection);

    // plugin has different versions and we only register once
    if (registered_plugin_names.find(plugin_name) != registered_plugin_names.end()) {
      continue;
    }
    registered_plugin_names.insert(plugin_name);

    std::unique_ptr<TensorRTCustomOp> trt_custom_op = std::make_unique<TensorRTCustomOp>(onnxruntime::kTensorrtExecutionProvider, nullptr);
    trt_custom_op->SetName(plugin_creator->getPluginName());
    custom_op_domain->custom_ops_.push_back(trt_custom_op.release());
  }

  *domain = custom_op_domain.release();

  return common::Status::OK();
}

}  // namespace onnxruntime
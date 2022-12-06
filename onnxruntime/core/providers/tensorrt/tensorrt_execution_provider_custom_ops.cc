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

/*
 * Create custom op domain list for TRT plugins.
 *
 * There are several TRT plugins registered as onnx schema op through contrib op with onnx domain.
 * In order not to break the old models using those TRT plugins and maintain backward compatibility, we need to keep
 * the old/legacy TRT plugins with onnx domain. Moving forward, all newly added TRT plugins should be registered with
 * "trt.plugins" domain.
 *
 * Please note that current TRT plugin doesn't have APIs to get number of inputs/outputs of the plugin.
 * So, the TensorRTCustomOp currently hardcodes number of inputs/outputs of the plugin/custom op.
 */
common::Status CreateTensorRTCustomOpDomainList(std::vector<OrtProviderCustomOpDomain*>& custom_op_domain_list) {
  std::unique_ptr<OrtProviderCustomOpDomain> legacy_custom_op_domain = std::make_unique<OrtProviderCustomOpDomain>();
  legacy_custom_op_domain->domain_ = kOnnxDomain;
  std::unique_ptr<OrtProviderCustomOpDomain> custom_op_domain = std::make_unique<OrtProviderCustomOpDomain>();
  custom_op_domain->domain_ = "trt.plugins";

  LOGS_DEFAULT(VERBOSE) << "[TensorRT EP] Get all registered TRT plugins from registry.";
  TensorrtLogger trt_logger = GetTensorrtLogger();
  initLibNvInferPlugins(&trt_logger, "");

  int num_plugin_creator = 0;
  auto plugin_creators = getPluginRegistry()->getPluginCreatorList(&num_plugin_creator);
  std::unordered_set<std::string> registered_plugin_names;
  std::unordered_set<std::string> legacy_custom_ops = {"EfficientNMS_TRT", "MultilevelCropAndResize_TRT", "PyramidROIAlign_TRT", "DisentangledAttention_TRT"};

  for (int i = 0; i < num_plugin_creator; i++) {
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

    if (legacy_custom_ops.find(plugin_name) != legacy_custom_ops.end()) {
      std::unique_ptr<TensorRTCustomOp> legacy_trt_custom_op = std::make_unique<TensorRTCustomOp>(onnxruntime::kTensorrtExecutionProvider, nullptr);
      legacy_trt_custom_op->SetName(plugin_creator->getPluginName());
      legacy_custom_op_domain->custom_ops_.push_back(legacy_trt_custom_op.release());
    } else {
      std::unique_ptr<TensorRTCustomOp> trt_custom_op = std::make_unique<TensorRTCustomOp>(onnxruntime::kTensorrtExecutionProvider, nullptr);
      trt_custom_op->SetName(plugin_creator->getPluginName());
      custom_op_domain->custom_ops_.push_back(trt_custom_op.release());
    }
  }
  custom_op_domain_list.push_back(legacy_custom_op_domain.release());
  custom_op_domain_list.push_back(custom_op_domain.release());

  return common::Status::OK();
}

/*
 * Create custom op domain list for TRT plugins.
 */
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

void ReleaseTensorRTCustomOpDomain(OrtProviderCustomOpDomain* domain) {
  if (domain != nullptr) {
    for (auto ptr : domain->custom_ops_) {
      if (ptr != nullptr) {
        delete ptr;
      }
    }
    delete domain;
  }
}

void ReleaseTensorRTCustomOpDomainList(std::vector<OrtProviderCustomOpDomain*>& custom_op_domain_list) {
  for (auto ptr : custom_op_domain_list) {
    ReleaseTensorRTCustomOpDomain(ptr);
  }
}

}  // namespace onnxruntime
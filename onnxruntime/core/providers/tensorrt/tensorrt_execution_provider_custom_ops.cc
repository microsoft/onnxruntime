// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


#include "core/framework/provider_options.h"
#include "tensorrt_execution_provider_custom_ops.h"
#include <NvInferRuntime.h>
#include <iostream>

namespace onnxruntime {
int CreateTensorRTCustomOpDomain_new(OrtProviderCustomOpDomain** domain) {
  std::unique_ptr<OrtProviderCustomOpDomain> custom_op_domain = std::make_unique<OrtProviderCustomOpDomain>();
  custom_op_domain->domain_ = "";

  std::unique_ptr<OrtCustomOp> disentangled_attention_custom_op = std::make_unique<DisentangledAttentionCustomOp>("TensorrtExecutionProvider", nullptr);
  custom_op_domain->custom_ops_.push_back(disentangled_attention_custom_op.release());
  //custom_ops.push_back(disentangled_attention_custom_op.release());

  *domain = custom_op_domain.release();

  int num_plugin_creator = 0;
  auto creators = getPluginRegistry()->getPluginCreatorList(&num_plugin_creator);

  for (int i = 0; i < num_plugin_creator; ++i) {
    auto plugin_creator = creators[i];
    std::string str(plugin_creator->getPluginName());
    std::cout << "\n" << str << std::endl;
    std::cout << "version: " << plugin_creator->getPluginVersion();

    auto plugin_field_names = plugin_creator->getFieldNames();
    if (plugin_field_names != nullptr) {
      int num_field = plugin_field_names->nbFields;
      auto plugin_field = plugin_field_names->fields;
      std::cout << num_field << std::endl;
      std::string field_name_str(plugin_field->name);
      std::cout << field_name_str << std::endl;
    }
    

  }

  return 0;
}
}  // namespace onnxruntime
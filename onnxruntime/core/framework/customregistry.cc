// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/customregistry.h"
namespace onnxruntime {

common::Status CustomRegistry::RegisterCustomKernel(KernelDefBuilder& kernel_def_builder, const KernelCreateFn& kernel_creator) {
  return kernel_registry_->Register(kernel_def_builder, kernel_creator);
}

common::Status CustomRegistry::RegisterCustomKernel(KernelCreateInfo& create_info) {
  return kernel_registry_->Register(std::move(create_info));
}

const std::shared_ptr<KernelRegistry>& CustomRegistry::GetKernelRegistry() {
  return kernel_registry_;
}

#if !defined(ORT_MINIMAL_BUILD)
common::Status CustomRegistry::RegisterOpSet(
    std::vector<ONNX_NAMESPACE::OpSchema>& schemas,
    const std::string& domain,
    int baseline_opset_version,
    int opset_version) {
  return opschema_registry_->RegisterOpSet(schemas, domain, baseline_opset_version, opset_version);
}

const std::shared_ptr<onnxruntime::OnnxRuntimeOpSchemaRegistry>& CustomRegistry::GetOpschemaRegistry() {
  return opschema_registry_;
}
#endif

}  // namespace onnxruntime

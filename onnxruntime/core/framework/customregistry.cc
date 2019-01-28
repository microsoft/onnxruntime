// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/customregistry.h"
namespace onnxruntime {

common::Status CustomRegistry::RegisterCustomKernel(KernelDefBuilder& kernel_def_builder, const KernelCreateFn& kernel_creator) {
  return Register(kernel_def_builder, kernel_creator);
}

common::Status CustomRegistry::RegisterCustomKernel(KernelCreateInfo& create_info) {
  return Register(std::move(create_info));
}

}  // namespace onnxruntime

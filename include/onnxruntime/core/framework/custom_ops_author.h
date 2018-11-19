// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

/**
   this header should include all the headers that are required to build a custom op so that
   custom op developers don't have to worry about which headers to include, etc.
*/
#include "core/framework/op_kernel.h"

struct KernelsContainer {
  std::vector<::onnxruntime::KernelCreateInfo> kernels_list;
};

struct SchemasContainer {
  std::vector<ONNX_NAMESPACE::OpSchema> schemas_list;
  std::string domain;
  int baseline_opset_version;
  int opset_version;
};

extern "C" {
  KernelsContainer* GetAllKernels();
  SchemasContainer* GetAllSchemas();
  void FreeKernelsContainer(KernelsContainer*);
  void FreeSchemasContainer(SchemasContainer*);
}

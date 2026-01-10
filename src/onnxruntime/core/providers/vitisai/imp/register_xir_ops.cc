// Copyright (c) 2023 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.

#include "./register_xir_ops.h"
#include "./vai_assert.h"
#include "core/providers/shared_library/provider_api.h"
#include "core/session/onnxruntime_c_api.h"

using namespace onnxruntime;

namespace vaip {
void register_xir_ops(const std::vector<OrtCustomOpDomain*>& domains) {
  for (auto domain : domains) {
    for (auto op : domain->custom_ops_) {
      // skip dequant quant schema, but register kernel
      if (Provider_GetHost()->GetSchema(op->GetName(op), op->GetStartVersion(op), domain->domain_) == nullptr) {
        Provider_GetHost()->RegisterSchema(domain->domain_, op);
      }
    }
  }
}

void deregister_xir_ops(const std::vector<OrtCustomOpDomain*>& domains) {
  for (auto domain : domains) {
    if (domain->domain_ != "com.xilinx") continue;  // skip dequant quant schema
    for (auto op : domain->custom_ops_) {
      if (Provider_GetHost()->GetSchema(op->GetName(op), op->GetStartVersion(op), domain->domain_) != nullptr) {
        Provider_GetHost()->DeregisterSchema(domain->domain_, op->GetName(op), op->GetStartVersion(op));
      }
    }
  }
}

}  // namespace vaip

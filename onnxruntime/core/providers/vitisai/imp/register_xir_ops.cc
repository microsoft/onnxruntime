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
      if (Provider_GetHost()->GetSchema(op->GetName(op), op->GetStartVersion(op), domain->domain_) == nullptr) {
        Provider_GetHost()->RegisterSchema(domain->domain_, op);
      }
    }
  }
}

}  // namespace vaip

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
      auto name = op->GetName(op);
      if ((std::string)name == "super_layer") {
        Provider_GetHost()->RegisterSchema(domain->domain_, op, 1);
      } else if ((std::string)name == "FixNeuron") {
        Provider_GetHost()->RegisterSchema(domain->domain_, op, 2);
      } else {
        Provider_GetHost()->RegisterSchema(domain->domain_, op, 3);
      }
    }
  }
}

}  // namespace vaip

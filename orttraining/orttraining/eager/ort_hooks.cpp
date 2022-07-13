// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <ATen/detail/ORTHooksInterface.h>

namespace torch_ort {
namespace eager {

struct ORTHooks : public at::ORTHooksInterface {
  ORTHooks(at::ORTHooksArgs) {}
  std::string showConfig() const override {
    return "  - ORT is enabled\n";
  }
};

using at::ORTHooksRegistry;
using at::RegistererORTHooksRegistry;
REGISTER_ORT_HOOKS(ORTHooks);

} // namespace eager
} // namespace torch_ort
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/codegen/common/target_info.h"
#include <functional>
#include <limits.h>

namespace onnxruntime {
namespace codegen {

using DomainVersionLookupFunc = std::function<int(const std::string&)>;

struct CodeGenHandle {
  CodeGenTarget* codegen_target;
  DomainVersionLookupFunc domain_version_lookup_func =
      // by default, always uses the latest opset implemented
      [](const std::string&) { return INT_MAX; };
};

}  // namespace codegen
}  // namespace onnxruntime

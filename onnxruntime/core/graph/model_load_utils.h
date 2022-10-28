// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <list>
#include <unordered_map>
#include <memory>
#include <string>
#include "core/platform/env.h"
#include "core/common/common.h"

namespace onnxruntime {

namespace model_load_utils {

static constexpr const char* kAllowReleasedONNXOpsetOnly = "ALLOW_RELEASED_ONNX_OPSET_ONLY";

inline bool IsAllowReleasedONNXOpsetsOnlySet() {
  // Get the value of env variable kAllowReleasedONNXOpsetOnly
  const std::string allow_official_onnx_release_only_str =
      Env::Default().GetEnvironmentVar(model_load_utils::kAllowReleasedONNXOpsetOnly);

  if (!allow_official_onnx_release_only_str.empty()) {
    // Check if the env var contains an unsupported value
    if (allow_official_onnx_release_only_str.length() > 1 ||
        (allow_official_onnx_release_only_str[0] != '0' && allow_official_onnx_release_only_str[0] != '1')) {
      ORT_THROW("The only supported values for the environment variable ",
                model_load_utils::kAllowReleasedONNXOpsetOnly,
                " are '0' and '1'. The environment variable contained the value: ",
                allow_official_onnx_release_only_str);
    }

    return allow_official_onnx_release_only_str[0] == '1';
  }

  return true;
}

inline void ValidateOpsetForDomain(const std::unordered_map<std::string, int>& onnx_released_versions, const logging::Logger& logger,
                                   bool allow_official_onnx_release_only,
                                   const std::string& domain, int version) {
  auto it = onnx_released_versions.find(domain);
  if (it != onnx_released_versions.end() && version > it->second) {
    auto current_domain = domain.empty() ? kOnnxDomainAlias : domain;
    if (allow_official_onnx_release_only) {
      ORT_THROW(
          "ONNX Runtime only *guarantees* support for models stamped "
          "with official released onnx opset versions. "
          "Opset ",
          version,
          " is under development and support for this is limited. "
          "The operator schemas and or other functionality may change before next ONNX release "
          "and in this case ONNX Runtime will not guarantee backward compatibility. "
          "Current official support for domain ",
          current_domain, " is till opset ",
          it->second, ".");
    } else {
      LOGS(logger, WARNING) << "ONNX Runtime only *guarantees* support for models stamped "
                               "with official released onnx opset versions. "
                               "Opset "
                            << version
                            << " is under development and support for this is limited. "
                               "The operator schemas and or other functionality "
                               "could possibly change before next ONNX release and "
                               "in this case ONNX Runtime will not guarantee backward compatibility. "
                               "Current official support for domain "
                            << current_domain
                            << " is till opset "
                            << it->second
                            << ".";
    }
  }
}

/** Generates a unique identifier for the given FunctionProto using the function proto domain and name.
*/
inline std::string GetModelLocalFuncId(const ONNX_NAMESPACE::FunctionProto& function_proto) {
  return function_proto.domain() + ":" + function_proto.name();
}
}  //namespace model_load_utils
}  // namespace onnxruntime

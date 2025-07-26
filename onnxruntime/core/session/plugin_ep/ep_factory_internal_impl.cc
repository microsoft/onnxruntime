// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/session/plugin_ep/ep_factory_internal_impl.h"

#include "core/framework/error_code_helper.h"
#include "core/session/abi_devices.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/plugin_ep/ep_factory_internal.h"

namespace onnxruntime {

// Prior to addition to SessionOptions the EP options do not have a prefix.
// They are prefixed with 'ep.<ep_name>.' when added to SessionOptions.
//
// Use this function to get the options without the prefix from SessionOptions.
// Required by the option parsing for multiple existing EPs.
ProviderOptions EpFactoryInternalImpl::GetOptionsFromSessionOptions(const SessionOptions& session_options) const {
  const std::string option_prefix = OrtSessionOptions::GetProviderOptionPrefix(GetName());
  ProviderOptions ep_options;

  for (const auto& [key, value] : session_options.config_options.configurations) {
    if (key.find(option_prefix) == 0) {
      // remove the prefix and add
      ep_options[key.substr(option_prefix.length())] = value;
    }
  }

  return ep_options;
}

}  // namespace onnxruntime

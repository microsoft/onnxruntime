// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/optimizer/selectors_actions/actions.h"

namespace onnxruntime {

class FusedConvActivationActionBase : public ReplaceWithNew {
private:
  std::string OpType(const RuntimeState& runtime_state) const override {
    const auto& domain = runtime_state.selected_nodes.Target().Domain();
    const auto& op_type = runtime_state.selected_nodes.Target().OpType();
    if (domain == kOnnxDomain) {
      if (op_type == "Conv") {
        return "FusedConv";
      }
    } else if (domain == kMSDomain) {
      if (op_type == "NhwcConv") {
        return "NhwcFusedConv";
      }
    } else if (domain == kMSInternalNHWCDomain) {
      if (op_type == "Conv") {
        return "Conv";
      }
    }
    ORT_THROW("Unsupported operator: ", op_type, " and domain: ", domain);
  }

  std::string Domain(const RuntimeState& runtime_state) const override {
    auto domain = runtime_state.selected_nodes.Target().Domain();
    return domain == kOnnxDomain ? kMSDomain : domain;
  }};

}  // namespace onnxruntime

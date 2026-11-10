// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "stateful/session/session_chain.h"

namespace onnxruntime {

struct ModelPackageChainDefinition {
  SessionChainOptions options;
  InlinedVector<std::string> component_names;
  InlinedVector<std::optional<SessionLoopOptions>, 1> loops;
};

Status ParseModelPackageChain(std::string_view json, ModelPackageChainDefinition& out);

}  // namespace onnxruntime

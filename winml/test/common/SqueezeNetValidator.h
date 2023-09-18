// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "std.h"
#include "winrt_headers.h"

enum OutputBindingStrategy {
  Bound,
  Unbound,
  Empty
};

namespace WinML::Engine::Test::ModelValidator {
void FnsCandy16(
  const std::string& instance,
  winml::LearningModelDeviceKind deviceKind,
  OutputBindingStrategy outputBindingStrategy,
  bool bindInputsAsIInspectable,
  float dataTolerance = false
);

void SqueezeNet(
  const std::string& instance,
  winml::LearningModelDeviceKind deviceKind,
  float dataTolerance,
  bool bindAsImage = false,
  OutputBindingStrategy outputBindingStrategy = OutputBindingStrategy::Bound,
  bool bindInputsAsIInspectable = false
);
}  // namespace WinML::Engine::Test::ModelValidator

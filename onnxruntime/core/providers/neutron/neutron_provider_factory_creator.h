// Copyright 2024-2026 NXP
// SPDX-License-Identifier: MIT

#pragma once

#include <memory>

#include "core/providers/providers.h"
#include "core/providers/neutron/neutron_provider_factory.h"

namespace onnxruntime {
struct NeutronProviderFactoryCreator {
  static std::shared_ptr<IExecutionProviderFactory> Create(NeutronProviderOptions neutron_options);
};
}  // namespace onnxruntime

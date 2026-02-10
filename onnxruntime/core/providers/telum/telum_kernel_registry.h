// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <memory>

namespace onnxruntime {
class KernelRegistry;

namespace telum {

// Shared kernel registry for the Telum EP. Intended to be shared across sessions.
std::shared_ptr<KernelRegistry> GetTelumKernelRegistry();

}  // namespace telum
}  // namespace onnxruntime


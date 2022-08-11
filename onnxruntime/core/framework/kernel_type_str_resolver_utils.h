// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"

#include "core/common/status.h"
#include "core/framework/kernel_type_str_resolver.h"

namespace onnxruntime::kernel_type_str_resolver_utils {

using common::Status;

gsl::span<const OpIdentifier> GetRequiredOpIdentifiers();

Status AddOptimizationOpsToKernelTypeStrResolver(KernelTypeStrResolver& kernel_type_str_resolver);

}  // namespace onnxruntime::utils

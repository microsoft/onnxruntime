// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "gsl/gsl"

#include "core/common/status.h"
#include "core/framework/kernel_type_str_resolver.h"

namespace flatbuffers {
class DetachedBuffer;
}

namespace onnxruntime::kernel_type_str_resolver_utils {

using common::Status;

// serialization and deserialization

Status SaveKernelTypeStrResolverToBuffer(const KernelTypeStrResolver& kernel_type_str_resolver,
                                         flatbuffers::DetachedBuffer& buffer, gsl::span<const uint8_t>& buffer_span);

Status LoadKernelTypeStrResolverFromBuffer(KernelTypeStrResolver& kernel_type_str_resolver,
                                           gsl::span<const uint8_t> buffer_span);

// add required ops to resolver
// TODO find better name than "required ops"
gsl::span<const OpIdentifier> GetRequiredOpIdentifiers();

Status AddRequiredOpsToKernelTypeStrResolver(KernelTypeStrResolver& kernel_type_str_resolver);

}  // namespace onnxruntime::utils

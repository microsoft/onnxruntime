// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#if !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

#include "core/common/gsl.h"
#include "core/common/status.h"
#include "core/framework/kernel_type_str_resolver.h"
#include "core/graph/op_identifier.h"

namespace flatbuffers {
class DetachedBuffer;
}

namespace onnxruntime::kernel_type_str_resolver_utils {

#if !defined(ORT_MINIMAL_BUILD)

/**
 * Gets the ops that the layout transformation may potentially add.
 */
gsl::span<const OpIdentifierWithStringViews> GetLayoutTransformationRequiredOpIdentifiers();

/**
 * Saves `kernel_type_str_resolver` to a byte buffer owned by `buffer` and referenced by `buffer_span`.
 */
Status SaveKernelTypeStrResolverToBuffer(const KernelTypeStrResolver& kernel_type_str_resolver,
                                         flatbuffers::DetachedBuffer& buffer, gsl::span<const uint8_t>& buffer_span);

#endif  // !defined(ORT_MINIMAL_BUILD)

/**
 * Loads `kernel_type_str_resolver` from the byte buffer referenced by `buffer_span`.
 */
Status LoadKernelTypeStrResolverFromBuffer(KernelTypeStrResolver& kernel_type_str_resolver,
                                           gsl::span<const uint8_t> buffer_span);

/**
 * Adds the ops that the layout transformation may potentially add to `kernel_type_str_resolver`.
 * This is needed when loading an ORT format model in a build where layout transformation is enabled.
 */
Status AddLayoutTransformationRequiredOpsToKernelTypeStrResolver(KernelTypeStrResolver& kernel_type_str_resolver);

}  // namespace onnxruntime::kernel_type_str_resolver_utils

#endif  // !defined(ORT_MINIMAL_BUILD) || defined(ORT_EXTENDED_MINIMAL_BUILD)

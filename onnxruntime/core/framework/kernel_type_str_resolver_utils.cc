// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/framework/kernel_type_str_resolver_utils.h"

#include <array>

#include "flatbuffers/flatbuffers.h"

#include "core/common/common.h"
#include "core/graph/op_identifier_utils.h"
#include "core/flatbuffers/schema/ort.fbs.h"

namespace onnxruntime::kernel_type_str_resolver_utils {

gsl::span<const OpIdentifier> GetRequiredOpIdentifiers() {
  static const std::array op_identifiers{
      MakeOpId(kOnnxDomain, "Transpose", 1),
      MakeOpId(kOnnxDomain, "Transpose", 13),
      MakeOpId(kOnnxDomain, "Squeeze", 1),
      MakeOpId(kOnnxDomain, "Squeeze", 11),
      MakeOpId(kOnnxDomain, "Squeeze", 13),
      MakeOpId(kOnnxDomain, "Unqueeze", 1),
      MakeOpId(kOnnxDomain, "Unqueeze", 11),
      MakeOpId(kOnnxDomain, "Unqueeze", 13),
      MakeOpId(kOnnxDomain, "Gather", 1),
      MakeOpId(kOnnxDomain, "Gather", 11),
      MakeOpId(kOnnxDomain, "Gather", 13),
      MakeOpId(kOnnxDomain, "Identity", 1),
      MakeOpId(kOnnxDomain, "Identity", 13),
      MakeOpId(kOnnxDomain, "Identity", 14),
      MakeOpId(kOnnxDomain, "Identity", 16),

      MakeOpId(kMSDomain, "QLinearConv", 1),
      MakeOpId(kMSDomain, "NhwcMaxPool", 1),
  };
  return op_identifiers;
}

Status AddOptimizationOpsToKernelTypeStrResolver(KernelTypeStrResolver& kernel_type_str_resolver) {
  KernelTypeStrResolver resolver_with_required_types{};
  // TODO load from bytes
  // resolver_with_required_types.LoadFromOrtFormat(...)
  kernel_type_str_resolver.Merge(resolver_with_required_types);
  return Status::OK();
}

}  // namespace onnxruntime::kernel_type_str_resolver_utils

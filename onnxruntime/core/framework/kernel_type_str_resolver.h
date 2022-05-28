// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <utility>

#include "gsl/gsl"

#include "onnx/defs/schema.h"

#include "core/common/inlined_containers.h"
#include "core/graph/basic_types.h"
#include "core/graph/graph.h"

namespace flatbuffers {
class FlatBufferBuilder;
template <typename T>
struct Offset;
}  // namespace flatbuffers

namespace onnxruntime {

namespace fbs {
struct KernelTypeStrResolver;
}  // namespace fbs

using ArgTypeAndIndex = std::pair<ArgType, size_t>;

class KernelTypeStrResolver {
 public:
  using KernelTypeStrToArgsMap = InlinedHashMap<std::string, InlinedVector<ArgTypeAndIndex>>;

  gsl::span<const ArgTypeAndIndex> ResolveKernelTypeStr(const OpIdentifier& op_id,
                                                        const std::string& kernel_type_str) const;

  bool RegisterKernelTypeStrToArgsMap(OpIdentifier op_id, KernelTypeStrToArgsMap kernel_type_str_to_args);

#if !defined(ORT_MINIMAL_BUILD)
  bool RegisterOpSchema(const ONNX_NAMESPACE::OpSchema& op_schema);

  bool RegisterNodeOpSchema(const Node& node);

  Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                         flatbuffers::Offset<fbs::KernelTypeStrResolver>& fbs_kernel_type_str_resolver) const;
#endif  // !defined(ORT_MINIMAL_BUILD)

  Status LoadFromOrtFormat(const fbs::KernelTypeStrResolver& fbs_kernel_type_str_resolver);

 private:
  using OpKernelTypeStrMap = InlinedHashMap<OpIdentifier, KernelTypeStrToArgsMap>;
  OpKernelTypeStrMap op_kernel_type_str_map_;
};

}  // namespace onnxruntime

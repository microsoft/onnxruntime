// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <string>
#include <string_view>
#include <utility>

#include "flatbuffers/flatbuffers.h"

#if !defined(ORT_MINIMAL_BUILD)
#include "core/graph/onnx_protobuf.h"
#endif  // !defined(ORT_MINIMAL_BUILD)

#include "core/common/gsl.h"
#include "core/common/inlined_containers.h"
#include "core/common/status.h"
#include "core/graph/op_identifier.h"
#include "core/graph/graph.h"
#include "core/platform/ort_mutex.h"

namespace onnxruntime {

namespace fbs {
struct KernelTypeStrResolver;
}  // namespace fbs

using ArgTypeAndIndex = std::pair<ArgType, size_t>;
using KernelTypeStrToArgsMap = InlinedHashMap<std::string, InlinedVector<ArgTypeAndIndex>>;
using OpKernelTypeStrMap = InlinedHashMap<OpIdentifier, KernelTypeStrToArgsMap>;

/**
 * This class interface provides a way to resolve an op's kernel type string to its associated arguments.
 *
 * A 'kernel type string' is a string that is used in kernel def type constraints. In particular, it can be a named
 * type parameter (such as 'T') specified in the op schema or it can be the name of an input or output parameter.
 */
class IKernelTypeStrResolver {
 public:
  /**
   * Resolves an op's kernel type string to its associated arguments.
   * @param node The op's node.
   * @param kernel_type_str The op kernel type string.
   * @param[out] resolved_args The op arguments associated with kernel_type_str.
   */
  virtual Status ResolveKernelTypeStr(const Node& node, std::string_view kernel_type_str,
                                      gsl::span<const ArgTypeAndIndex>& resolved_args) const = 0;

 protected:
  ~IKernelTypeStrResolver() = default;
};

/**
 * A basic implementation of IKernelTypeStrResolver.
 *
 * Supports loading information from op schemas in a full build and saving to/loading from an ORT format model
 * representation.
 */
class KernelTypeStrResolver final : public IKernelTypeStrResolver {
 public:
  Status ResolveKernelTypeStr(const Node& node, std::string_view kernel_type_str,
                              gsl::span<const ArgTypeAndIndex>& resolved_args) const override;

#if !defined(ORT_MINIMAL_BUILD)

  /**
   * Registers kernel type string matching info from an op schema.
   * This will not overwrite an existing registration for the same op.
   * @param op_schema The op schema to register.
   * @param[out] registered Whether the op schema was registered or there was already an existing registration.
   */
  Status RegisterOpSchema(const ONNX_NAMESPACE::OpSchema& op_schema, bool* registered = nullptr);

  /**
   * Registers kernel type string matching info from an op schema from a node.
   * @param node The node to register.
   */
  Status RegisterNodeOpSchema(const Node& node);

  /**
   * Registers kernel type string matching info from op schemas from nodes in a graph.
   * @param graph The graph to register.
   */
  Status RegisterGraphNodeOpSchemas(const Graph& graph);

  /**
   * Saves to an ORT format model representation.
   * @param builder The flatbuffers builder.
   * @param[out] fbs_kernel_type_str_resolver The saved flatbuffers representation offset.
   */
  Status SaveToOrtFormat(flatbuffers::FlatBufferBuilder& builder,
                         flatbuffers::Offset<fbs::KernelTypeStrResolver>& fbs_kernel_type_str_resolver) const;

#endif  // !defined(ORT_MINIMAL_BUILD)

  /**
   * Loads from an ORT format model representation.
   * @param fbs_kernel_type_str_resolver The flatbuffers representation to load.
   */
  Status LoadFromOrtFormat(const fbs::KernelTypeStrResolver& fbs_kernel_type_str_resolver);

  /**
   * Merges kernel type string matching info from another KernelTypeStrResolver.
   * @param src The KernelTypeStrResolver to merge from.
   */
  void Merge(KernelTypeStrResolver src);

  const OpKernelTypeStrMap& GetOpKernelTypeStrMap() const { return op_kernel_type_str_map_; }

 private:
  OpKernelTypeStrMap op_kernel_type_str_map_;
};

#if !defined(ORT_MINIMAL_BUILD)

/**
 * An implementation of IKernelTypeStrResolver which loads kernel type string matching info from node op schemas.
 *
 * As this requires node op schemas, it is only enabled in a full build.
 */
class OpSchemaKernelTypeStrResolver final : public IKernelTypeStrResolver {
 public:
  // Note: `node`'s op schema must be populated.
  Status ResolveKernelTypeStr(const Node& node, std::string_view kernel_type_str,
                              gsl::span<const ArgTypeAndIndex>& resolved_args) const override;

 private:
  // used as a cache when resolving
  // since the cache may be modified with a const instance, ensure that access to the cache is thread-safe
  mutable KernelTypeStrResolver resolver_;
  mutable OrtMutex resolver_mutex_;
};

#endif  // !defined(ORT_MINIMAL_BUILD)

}  // namespace onnxruntime

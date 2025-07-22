// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <unordered_map>
#include <unordered_set>
#include <string>
#include <cstdint>
#include <memory>
#include <functional>

#include "core/common/basic_types.h"
#include "core/common/transform_iterator.h"
#include "core/common/status.h"

namespace ONNX_NAMESPACE {
class ValueInfoProto;
class TensorProto;
class SparseTensorProto;
class TypeProto;
class AttributeProto;
class FunctionProto;
class OperatorSetIdProto;
// define types that would come from the ONNX library if we were building against it.
#if defined(ORT_MINIMAL_BUILD)
using OperatorSetVersion = int;
#endif

}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
using NodeIndex = size_t;
using Version = int64_t;
using NodeArgInfo = ONNX_NAMESPACE::ValueInfoProto;
using ArgNameToTypeMap = std::unordered_map<std::string, ONNX_NAMESPACE::TypeProto>;
using ProviderType = const std::string&;

// Create a shallow class to act as a proxy for whatever container is currently used for the initialized tensors in the graph.
// exposing begin()/end(), size(), empty methods. Begin()/end() method pairs must return a onnxruntime::transform_iterator template
// that returns references to the map keys only. That transform iterator must use std::select1st() as a transformation function.
namespace details {
constexpr auto select1st = [](auto&& x) noexcept -> decltype(auto) {
  return std::get<0>(std::forward<decltype(x)>(x));
};

using TensorSetIterator = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>::const_iterator;
using ProjectedTensorSetIterator =
    onnxruntime::transform_iterator<TensorSetIterator, decltype(select1st)>;

}  // namespace details

/**
 * @class InitializedTensorSetProxy
 * @brief A read-only proxy for a set of initialized tensors.
 *
 * This class provides a lightweight, non-owning view over a container of initialized tensors
 * (specifically, a `std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>`).
 *
 * It exposes an iterator-based interface (`begin()`, `end()`) that allows iterating directly
 * over the `const ONNX_NAMESPACE::TensorProto*` values, hiding the underlying map structure.
 * This is useful for algorithms that only need to access the tensor protos without their names.
 *
 * @remarks The lifetime of an InitializedTensorSetProxy instance must not exceed the lifetime
 * of the underlying container it references.
 */
class InitializersNames {
 public:
  using const_iterator = details::ProjectedTensorSetIterator;
  using container = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>;

  InitializersNames(const container& initialized_tensors) noexcept
      : initialized_tensors_(initialized_tensors) {}

  const_iterator begin() const {
    return const_iterator(initialized_tensors_.get().begin(), details::select1st);
  }
  const_iterator cbegin() const {
    return const_iterator(initialized_tensors_.get().begin(), details::select1st);
  }
  const_iterator end() const {
    return const_iterator(initialized_tensors_.get().end(), details::select1st);
  }
  const_iterator cend() const {
    return const_iterator(initialized_tensors_.get().end(), details::select1st);
  }
  size_t size() const {
    return initialized_tensors_.get().size();
  }
  bool empty() const {
    return initialized_tensors_.get().empty();
  }
  bool count(const std::string& name) const {
    return initialized_tensors_.get().count(name);
  }
  bool contains(const std::string& name) const {
    return initialized_tensors_.get().count(name) > 0;
  }

 private:
  std::reference_wrapper<const container> initialized_tensors_;
};
;

using InitializedTensorSet = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>;

// TODO - Evaluate switching the types below to support transparent comparators and enable
// lookups based on gsl::cstring_span<> and std::string_view.  This would reduces allocations
// converting to std::string, but requires conversion to std::map<std::string, foo, std::less<>>
// instead of std::unordered_map<std::string, foo, [std::less<foo>]>.

using NodeAttributes = std::unordered_map<std::string, ONNX_NAMESPACE::AttributeProto>;
class IOnnxRuntimeOpSchemaCollection;
using IOnnxRuntimeOpSchemaCollectionPtr = std::shared_ptr<IOnnxRuntimeOpSchemaCollection>;

class OpKernel;
class OpKernelInfo;
class FuncManager;
using KernelCreateFn = std::function<onnxruntime::common::Status(FuncManager& func_mgr, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out)>;
}  // namespace onnxruntime

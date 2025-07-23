// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <functional>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "core/common/transform_iterator.h"

namespace ONNX_NAMESPACE {
#ifndef SHARED_PROVIDER
class TensorProto;
#else
struct TensorProto;
#endif
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {
// Create a shallow class to act as a proxy for whatever container is currently used for the initialized tensors
// in the graph. Exposing begin()/end(), size(), empty methods. Begin()/end() method pairs must return a
// onnxruntime::transform_iterator template that returns references to the map keys only. That transform
// iterator must use std::select1st() as a transformation function.
namespace initializers_names_details {
constexpr auto select1st = [](auto&& x) noexcept -> decltype(auto) {
  return std::get<0>(std::forward<decltype(x)>(x));
};

using TensorSetIterator = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>::const_iterator;
using ProjectedTensorSetIterator =
    onnxruntime::transform_iterator<TensorSetIterator, decltype(select1st)>;

}  // namespace initializers_names_details

/**
 * @class InitializersNames
 * @brief A read-only proxy for a set of initialized tensors.
 *
 * This class provides a lightweight, non-owning view over a container of initialized tensors
 * (specifically, at this time it is a `std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>`).
 *
 * It exposes an iterator-based interface (`begin()`, `end()`) that allows iterating directly
 * over the `const ONNX_NAMESPACE::TensorProto*` values, hiding the underlying map structure.
 * This is useful for hiding the underlying data structure allowing for more flexibility.
 *
 * @remarks The lifetime of an InitializersNames instance must not exceed the lifetime
 * of the underlying container it references.
 */
class InitializersNames {
 public:
  using const_iterator = initializers_names_details::ProjectedTensorSetIterator;
  using container = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>;

  InitializersNames(const container& initialized_tensors) noexcept
      : initialized_tensors_(initialized_tensors) {}

  const_iterator begin() const {
    return const_iterator(initialized_tensors_.get().begin(), initializers_names_details::select1st);
  }
  const_iterator cbegin() const {
    return const_iterator(initialized_tensors_.get().begin(), initializers_names_details::select1st);
  }
  const_iterator end() const {
    return const_iterator(initialized_tensors_.get().end(), initializers_names_details::select1st);
  }
  const_iterator cend() const {
    return const_iterator(initialized_tensors_.get().end(), initializers_names_details::select1st);
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

}  // namespace onnxruntime

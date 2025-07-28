// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <iterator>
#include <functional>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "boost/iterator/transform_iterator.hpp"

namespace ONNX_NAMESPACE {
// This header is being used in the provider bridge as is
// make sure it does not pull anything that is not compatible with the provider bridge.
#ifndef SHARED_PROVIDER
class TensorProto;
#else
struct TensorProto;
#endif
}  // namespace ONNX_NAMESPACE

namespace onnxruntime {


// Create a shallow class to act as a proxy for whatever container is currently used for the initialized tensors
// in the graph. Exposing begin()/end(), size(), empty methods. Begin()/end() method pairs must return a
// boost::transform_iterator template that returns references to the map keys only.
namespace initializers_names_details {

using tensor_container = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>;
using tensor_container_const_iterator = tensor_container::const_iterator;
using tensor_container_value_type = tensor_container::value_type;

struct select1st_t {
  using argument_type = tensor_container_value_type;
  using result_type = const std::string&;
  constexpr result_type operator()(const argument_type& v) const noexcept {
    return std::get<0>(v);
  }
};

using KeyProjectionIterator =
    boost::transform_iterator<select1st_t, tensor_container_const_iterator>;

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
  using const_iterator = initializers_names_details::KeyProjectionIterator;
  using container = std::unordered_map<std::string, const ONNX_NAMESPACE::TensorProto*>;

  InitializersNames(const container& initialized_tensors) noexcept
      : initialized_tensors_(initialized_tensors) {}

  const_iterator begin() const {
    return cbegin();
  }
  const_iterator cbegin() const {
    return boost::make_transform_iterator(initialized_tensors_.get().begin(), initializers_names_details::select1st_t());
  }
  const_iterator end() const {
    return cend();
  }
  const_iterator cend() const {
    return boost::make_transform_iterator(initialized_tensors_.get().end(), initializers_names_details::select1st_t());
  }
  size_t size() const noexcept {
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

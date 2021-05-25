// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#ifndef SHARED_PROVIDER
#include "core/common/status.h"
#include "core/graph/graph_viewer.h"
#include "gsl/gsl"
#endif

#ifdef __has_attribute
#define ORT_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define ORT_HAVE_ATTRIBUTE(x) 0
#endif

#if ORT_HAVE_ATTRIBUTE(nodiscard)
#define MUST_USE_RESULT [[nodiscard]]
#elif defined(__clang__) && ORT_HAVE_ATTRIBUTE(warn_unused_result)
#define MUST_USE_RESULT __attribute__((warn_unused_result))
#else
#define MUST_USE_RESULT
#endif

class IMLOpKernel;

namespace onnxruntime {

/**
   A set of wrappers with common signatures for use with both OpKernelInfo
   (as its base class) and InferenceContext.  Used by ABI kernels for both
   shape / type inference and kernel construction
*/
template <class Impl_t>
class OpNodeProtoHelper {
 public:
  explicit OpNodeProtoHelper(const Impl_t* impl) : impl_(impl) {}

  /**
     Get a single attribute
     Call this function for a required attribute or when a default value for an optional attribute is specified in the op schema
  */
  template <typename T>
  MUST_USE_RESULT Status GetAttr(const std::string& name, T* value) const;

  /**
     Get a single attribute
     Call this function only when a default value for an optional attribute isn't specified in the op schema
  */
  template <typename T>
  T GetAttrOrDefault(const std::string& name, const T& default_value) const {
    T tmp;
    return GetAttr<T>(name, &tmp).IsOK() ? tmp : default_value;
  }

  /**
     Get a single attribute
     Call this function only when a default value for an optional attribute isn't specified in the op schema
  */
  template <typename T>
  void GetAttrOrDefault(const std::string& name, T* value, const T& default_value) const {
    if (!GetAttr<T>(name, value).IsOK())
      *value = default_value;
  }

  /**
     Get repeated attributes
     Call this function only when a default value for an optional attribute isn't specified in the op schema
  */
  template <typename T>
  MUST_USE_RESULT std::vector<T> GetAttrsOrDefault(const std::string& name, const std::vector<T>& default_value = std::vector<T>{}) const {
    std::vector<T> tmp;
    return GetAttrs<T>(name, tmp).IsOK() ? tmp : default_value;
  }

  /**
     Get repeated attributes
  */
  template <typename T>
  MUST_USE_RESULT Status GetAttrs(const std::string& name, std::vector<T>& values) const;

  template <typename T>
  MUST_USE_RESULT Status GetAttrs(const std::string& name, gsl::span<T> values) const;

  /// <summary>
  /// Return a gsl::span that points to an array of primitive types held by AttributeProto
  /// This function allows to avoid copying big attributes locally into a kernel and operate on
  /// AttributeProto data directly.
  ///
  ///  Does not apply to strings, Tensors and Sparse Tensors that require special treatment.
  /// </summary>
  /// <typeparam name="T">Primitive type contained in the array</typeparam>
  /// <param name="name">Attribute name</param>
  /// <param name="values">Attribute data in a span, out parameter</param>
  /// <returns>Status</returns>
  template <typename T>
  MUST_USE_RESULT Status GetAttrsAsSpan(const std::string& name, gsl::span<const T>& values) const;

  MUST_USE_RESULT Status GetAttrsStringRefs(const std::string& name,
                                            std::vector<std::reference_wrapper<const std::string>>& refs) const;

  uint32_t GetPrimitiveAttrElementCount(ONNX_NAMESPACE::AttributeProto_AttributeType type,
                                        const std::string& name) const noexcept;

  bool HasPrimitiveAttribute(ONNX_NAMESPACE::AttributeProto_AttributeType type,
                             const std::string& name) const noexcept;

  uint32_t GetInputCount() const {
    return gsl::narrow_cast<uint32_t>(impl_->getNumInputs());
  }

  uint32_t GetOutputCount() const {
    return gsl::narrow_cast<uint32_t>(impl_->getNumOutputs());
  }

  const ONNX_NAMESPACE::TypeProto* GetInputType(size_t index) const {
    return impl_->getInputType(index);
  }

  const ONNX_NAMESPACE::TypeProto* GetOutputType(size_t index) const {
    // Work around lack of a const method from the onnx InferenceContext interface
    return const_cast<Impl_t*>(impl_)->getOutputType(index);
  }

  // Try to query an attribute, returning nullptr if it doesn't exist
  const ONNX_NAMESPACE::AttributeProto* TryGetAttribute(const std::string& name) const {
    return impl_->getAttribute(name);
  }

  const ONNX_NAMESPACE::AttributeProto* GetAttribute(const std::string& name) const {
    const ONNX_NAMESPACE::AttributeProto* attr = TryGetAttribute(name);
    ORT_ENFORCE(attr != nullptr);
    return attr;
  }

 private:
  OpNodeProtoHelper() = delete;
  const Impl_t* impl_ = nullptr;
};

// The methods on the following class are called by OpNodeProtoHelper, implementing
// the same signatures as InferenceContext other than const-ness.
class ProtoHelperNodeContext {
 public:
  explicit ProtoHelperNodeContext(const onnxruntime::Node& node) : node_(node) {}
  ProtoHelperNodeContext() = delete;

  const ONNX_NAMESPACE::AttributeProto* getAttribute(const std::string& name) const;
  size_t getNumInputs() const;
  const ONNX_NAMESPACE::TypeProto* getInputType(size_t index) const;
  size_t getNumOutputs() const;
  const ONNX_NAMESPACE::TypeProto* getOutputType(size_t index) const;

 private:
  const onnxruntime::Node& node_;
};

}  // namespace onnxruntime

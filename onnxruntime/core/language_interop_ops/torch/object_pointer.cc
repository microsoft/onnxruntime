// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/language_interop_ops/torch/object_pointer.h"

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

template <class T>
ObjectPointer<T>::ObjectPointer(ObjectPointer<T>&& p) noexcept {
  free();
  ptr = p.ptr;
  p.ptr = nullptr;
};

template <class T>
ObjectPointer<T>& ObjectPointer<T>::operator=(ObjectPointer<T>&& p) noexcept {
  free();
  ptr = p.ptr;
  p.ptr = nullptr;
  return *this;
};

}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

// RAII wrapper for any object
template <class T>
class ObjectPointer {
 public:
  ObjectPointer() : ptr(nullptr){};
  explicit ObjectPointer(T* ptr) noexcept : ptr(ptr){};

  ObjectPointer(ObjectPointer&& p) noexcept;
  ObjectPointer& operator=(ObjectPointer&& p) noexcept;

  ~ObjectPointer() { free(); };

  T* get() { return ptr; }
  const T* get() const { return ptr; }

  // Disable copy constructor/assignment
  ObjectPointer(const ObjectPointer&) = delete;
  ObjectPointer operator=(const ObjectPointer&) = delete;

 private:
  void free();
  T* ptr = nullptr;
};

}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime
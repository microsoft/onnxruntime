// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include "core/common/common.h"
#include "core/session/onnxruntime_c_api.h"
#include <atomic>

namespace onnxruntime {

/**
 * Even it's designed to be inherited, this class doesn't have a virtual destructor.
 * No vtable is allowed in this class and its subclasses.
 * \tparam T subclass type name
 */
template <typename T>
class ObjectBase {
 private:
  static ONNXObject static_cls;

 protected:
  const ONNXObject* const ONNXRUNTIME_ATTRIBUTE_UNUSED cls_;
  std::atomic_int ref_count;
  ObjectBase() : cls_(&static_cls), ref_count(1) {
  }

  static uint32_t ONNXRUNTIME_API_STATUSCALL ONNXRuntimeReleaseImpl(void* this_) {
    T* this_ptr = reinterpret_cast<T*>(this_);
    if (--this_ptr->ref_count == 0)
      delete this_ptr;
    return 0;
  }

  static uint32_t ONNXRUNTIME_API_STATUSCALL ONNXRuntimeAddRefImpl(void* this_) {
    T* this_ptr = reinterpret_cast<T*>(this_);
    ++this_ptr->ref_count;
    return 0;
  }
};

template <typename T>
ONNXObject ObjectBase<T>::static_cls = {ObjectBase<T>::ONNXRuntimeAddRefImpl, ObjectBase<T>::ONNXRuntimeReleaseImpl};

}  // namespace onnxruntime

#define ONNXRUNTIME_CHECK_C_OBJECT_LAYOUT \
  { assert((char*)&ref_count == (char*)this + sizeof(this)); }

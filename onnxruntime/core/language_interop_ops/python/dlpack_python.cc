// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef ENABLE_PYTHON

#include "core/language_interop_ops/python/dlpack_python.h"

namespace onnxruntime {
namespace dlpack {

static void DlpackCapsuleDestructor(PyObject* data) {
  DLManagedTensor* dlmanged_tensor = reinterpret_cast<DLManagedTensor*>(
      PyCapsule_GetPointer(data, "dltensor"));
  if (dlmanged_tensor) {
    // The dlmanged_tensor has not been consumed, call deleter ourselves.
    dlmanged_tensor->deleter(const_cast<DLManagedTensor*>(dlmanged_tensor));
  } else {
    // The dlmanged_tensor has been consumed,
    // PyCapsule_GetPointer has set an error indicator.
    PyErr_Clear();
  }
}

PyObject* ToDlpack(OrtValue ort_value) {
  DLManagedTensor* dlmanaged_tensor = dlpack::OrtValueToDlpack(ort_value);
  return PyCapsule_New(dlmanaged_tensor, "dltensor", DlpackCapsuleDestructor);
}

OrtValue FromDlpack(PyObject* dlpack_tensor, const bool is_bool_tensor) {
  // Extract DLPack tensor pointer from the capsule carrier.
  DLManagedTensor* dlmanaged_tensor = (DLManagedTensor*)PyCapsule_GetPointer(dlpack_tensor, "dltensor");
  OrtValue ort_value = dlpack::DlpackToOrtValue(dlmanaged_tensor, is_bool_tensor);
  // Make sure this capsule will never be used again.
  PyCapsule_SetName(dlpack_tensor, "used_dltensor");
  return ort_value;
}

}  // namespace dlpack
}  // namespace onnxruntime

#endif

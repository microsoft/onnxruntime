// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "error_code.h"
#ifdef __cplusplus
extern "C" {
#endif
/**
 * Just like the IUnknown interface in COM
 * Every type inherented from ONNXObject should be deleted by ONNXRuntimeReleaseObject(...).
 */
typedef struct ONNXObject {
  ///returns the new reference count.
  uint32_t(ONNXRUNTIME_API_STATUSCALL* AddRef)(void* this_);
  ///returns the new reference count.
  uint32_t(ONNXRUNTIME_API_STATUSCALL* Release)(void* this_);
  //TODO: implement QueryInterface?
} ONNXObject;

/**
 * This function is a wrapper to "(*(ONNXObject**)ptr)->AddRef(ptr)"
 * WARNING: There is NO type checking in this function.
 * Before calling this function, caller should make sure current ref count > 0
 * \return the new reference count
 */
ONNXRUNTIME_API(uint32_t, ONNXRuntimeAddRefToObject, _In_ void* ptr);

/**
 * 
 * A wrapper to "(*(ONNXObject**)ptr)->Release(ptr)"
 * WARNING: There is NO type checking in this function.
 * \param ptr Can be NULL. If it's NULL, this function will return zero.
 * \return the new reference count.
 */
ONNXRUNTIME_API(uint32_t, ONNXRuntimeReleaseObject, _Inout_opt_ void* ptr);
#ifdef __cplusplus
}
#endif

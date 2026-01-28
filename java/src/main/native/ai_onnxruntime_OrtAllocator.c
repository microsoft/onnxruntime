/*
 * Copyright (c) 2019, 2026, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtAllocator.h"

/*
 * Class:     ai_onnxruntime_OrtAllocator
 * Method:    getStats
 * Signature: (JJ)[[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtAllocator_getStats
  (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong nativeHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtAllocator* allocator = (OrtAllocator*) nativeHandle;
  if (allocator->GetStats == NULL) {
    // Throw an OrtException with an ORT_INVALID_ARGUMENT error code
    throwOrtException(jniEnv, 2, "This allocator does not support GetStats.");
    return NULL;
  }
  OrtKeyValuePairs* kvp;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, allocator->GetStats(allocator, &kvp));
  if (code != ORT_OK) {
    return NULL;
  } else {
    jobjectArray pair = convertOrtKeyValuePairsToArrays(jniEnv, api, kvp);
    return pair;
  }
}

/*
 * Class:     ai_onnxruntime_OrtAllocator
 * Method:    closeAllocator
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtAllocator_closeAllocator
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong allocatorHandle) {
  (void) jniEnv; (void) jobj; (void) apiHandle; (void) allocatorHandle;
}



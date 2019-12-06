/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtAllocator.h"
/*
 * Class:     ai_onnxruntime_OrtAllocator
 * Method:    closeAllocator
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtAllocator_closeAllocator
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong allocatorHandle) {
  (void) jniEnv; (void) jobj; (void) apiHandle; (void) allocatorHandle;
}



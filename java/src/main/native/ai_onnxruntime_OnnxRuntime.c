/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "ai_onnxruntime_OnnxRuntime.h"

/*
 * Class:     ai_onnxruntime_OnnxRuntime
 * Method:    initialiseAPIBase
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxRuntime_initialiseAPIBase
  (JNIEnv * jniEnv, jclass clazz, jint apiVersion) {
    (void) jniEnv; (void) clazz; // required JNI parameters not needed by functions which don't call back into Java.
    const OrtApi* ortPtr = OrtGetApiBase()->GetApi((uint32_t) apiVersion);
    return (jlong) ortPtr;
}


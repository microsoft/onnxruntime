/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "com_microsoft_onnxruntime_ONNX.h"

/*
 * Class:     com_microsoft_onnxruntime_ONNX
 * Method:    initialiseAPIBase
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_com_microsoft_onnxruntime_ONNX_initialiseAPIBase
  (JNIEnv * jniEnv, jclass clazz, jint apiVersion) {
    const OrtApi* ortPtr = OrtGetApiBase()->GetApi((uint32_t) apiVersion);
    return (jlong) ortPtr;
}


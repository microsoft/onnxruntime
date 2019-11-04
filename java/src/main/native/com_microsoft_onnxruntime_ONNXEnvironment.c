/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "ONNXUtil.h"
#include "ai_onnxruntime_ONNXEnvironment.h"

/*
 * Class:     ai_onnxruntime_ONNXEnvironment
 * Method:    createHandle
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_ONNXEnvironment_createHandle(JNIEnv * jniEnv, jobject obj, jlong apiHandle, jint loggingLevel, jstring name) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtEnv* env;
    jboolean copy;
    const char* cName = (*jniEnv)->GetStringUTFChars(jniEnv, name, &copy);
    checkONNXStatus(jniEnv,api,api->CreateEnv(convertLoggingLevel(loggingLevel), cName, &env));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,name,cName);
    return (jlong) env;
}

/*
 * Class:     ai_onnxruntime_ONNXEnvironment
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXEnvironment_close(JNIEnv * jniEnv, jobject obj, jlong apiHandle, jlong handle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseEnv((OrtEnv*)handle);
}

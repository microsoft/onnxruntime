/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "ONNXUtil.h"
#include "ai_onnxruntime_OrtEnvironment.h"

/*
 * Class:     ai_onnxruntime_OrtEnvironment
 * Method:    createHandle
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtEnvironment_createHandle(JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jint loggingLevel, jstring name) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtEnv* env;
    jboolean copy;
    const char* cName = (*jniEnv)->GetStringUTFChars(jniEnv, name, &copy);
    checkOrtStatus(jniEnv,api,api->CreateEnv(convertLoggingLevel(loggingLevel), cName, &env));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,name,cName);
    return (jlong) env;
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtEnvironment_close(JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseEnv((OrtEnv*)handle);
}

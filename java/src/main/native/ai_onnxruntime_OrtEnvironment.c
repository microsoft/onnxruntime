/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtEnvironment.h"

/*
 * Class:     ai_onnxruntime_OrtEnvironment
 * Method:    createHandle
 * Signature: (Ljava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtEnvironment_createHandle__JILjava_lang_String_2
  (JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jint loggingLevel, jstring name) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtEnv* env;
    jboolean copy;
    const char* cName = (*jniEnv)->GetStringUTFChars(jniEnv, name, &copy);
    checkOrtStatus(jniEnv,api,api->CreateEnv(convertLoggingLevel(loggingLevel), cName, &env));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,name,cName);
    checkOrtStatus(jniEnv, api, api->SetLanguageProjection(env, ORT_PROJECTION_JAVA));
    return (jlong) env;
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment
 * Method:    createHandle
 * Signature: (JILjava/lang/String;J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtEnvironment_createHandle__JILjava_lang_String_2J
  (JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jint loggingLevel, jstring name, jlong threadOptionsHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtEnv* env;
    jboolean copy;
    const char* cName = (*jniEnv)->GetStringUTFChars(jniEnv, name, &copy);
    checkOrtStatus(jniEnv,api,api->CreateEnvWithGlobalThreadPools(convertLoggingLevel(loggingLevel),
                                                                  cName,
                                                                  (OrtThreadingOptions*) threadOptionsHandle,
                                                                  &env));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,name,cName);
    checkOrtStatus(jniEnv, api, api->SetLanguageProjection(env, ORT_PROJECTION_JAVA));
    return (jlong) env;
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment
 * Method:    getDefaultAllocator
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtEnvironment_getDefaultAllocator
  (JNIEnv * jniEnv, jclass jobj, jlong apiHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator;
    checkOrtStatus(jniEnv,api,api->GetAllocatorWithDefaultOptions(&allocator));
    return (jlong)allocator;
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment
 * Method:    close
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtEnvironment_close(JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseEnv((OrtEnv*)handle);
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment
 * Method:    setTelemetry
 * Signature: (JJZ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtEnvironment_setTelemetry
        (JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jlong nativeHandle, jboolean sendTelemetry) {
    (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtEnv* env = (OrtEnv*) nativeHandle;
    if (sendTelemetry) {
        checkOrtStatus(jniEnv,api,api->EnableTelemetryEvents(env));
    } else {
        checkOrtStatus(jniEnv,api,api->DisableTelemetryEvents(env));
    }
}

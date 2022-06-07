/*
 * Copyright (c) 2021 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtEnvironment_ThreadingOptions.h"

/*
 * Class:     ai_onnxruntime_OrtEnvironment_ThreadingOptions
 * Method:    createThreadingOptions
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtEnvironment_00024ThreadingOptions_createThreadingOptions
  (JNIEnv * jniEnv, jclass clazz, jlong apiHandle) {
    (void) clazz; // Required JNI parameter not needed by functions which don't need to access their host class.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtThreadingOptions* opts;
    checkOrtStatus(jniEnv,api,api->CreateThreadingOptions(&opts));
    return (jlong) opts;
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment_ThreadingOptions
 * Method:    setGlobalIntraOpNumThreads
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtEnvironment_00024ThreadingOptions_setGlobalIntraOpNumThreads
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint numThreads) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkOrtStatus(jniEnv,api,api->SetGlobalIntraOpNumThreads((OrtThreadingOptions*) handle, numThreads));
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment_ThreadingOptions
 * Method:    setGlobalInterOpNumThreads
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtEnvironment_00024ThreadingOptions_setGlobalInterOpNumThreads
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint numThreads) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkOrtStatus(jniEnv,api,api->SetGlobalInterOpNumThreads((OrtThreadingOptions*) handle, numThreads));
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment_ThreadingOptions
 * Method:    setGlobalSpinControl
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtEnvironment_00024ThreadingOptions_setGlobalSpinControl
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint allowSpinning) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkOrtStatus(jniEnv,api,api->SetGlobalSpinControl((OrtThreadingOptions*) handle, allowSpinning));
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment_ThreadingOptions
 * Method:    setGlobalDenormalAsZero
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtEnvironment_00024ThreadingOptions_setGlobalDenormalAsZero
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkOrtStatus(jniEnv,api,api->SetGlobalDenormalAsZero((OrtThreadingOptions*) handle));
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment_ThreadingOptions
 * Method:    closeThreadingOptions
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtEnvironment_00024ThreadingOptions_closeThreadingOptions
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
  (void)jniEnv; (void)jobj;  // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  api->ReleaseThreadingOptions((OrtThreadingOptions*)handle);
}

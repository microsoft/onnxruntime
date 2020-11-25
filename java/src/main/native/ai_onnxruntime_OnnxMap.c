/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OnnxMap.h"

/*
 * Class:     ai_onnxruntime_OnnxMap
 * Method:    getStringKeys
 * Signature: (J)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OnnxMap_getStringKeys
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract key
    OrtValue* keys;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,0,allocator,&keys));

    // Convert to Java String array
    jobjectArray output = createStringArrayFromTensor(jniEnv, api, allocator, keys);

    api->ReleaseValue(keys);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxMap
 * Method:    getLongKeys
 * Signature: (JJ)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_OnnxMap_getLongKeys
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract key
    OrtValue* keys;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,0,allocator,&keys));

    jlongArray output = createLongArrayFromTensor(jniEnv, api, keys);

    api->ReleaseValue(keys);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxMap
 * Method:    getStringValues
 * Signature: (JJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OnnxMap_getStringValues
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract value
    OrtValue* values;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,1,allocator,&values));

    // Convert to Java String array
    jobjectArray output = createStringArrayFromTensor(jniEnv, api, allocator, values);

    api->ReleaseValue(values);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxMap
 * Method:    getLongValues
 * Signature: (JJ)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_OnnxMap_getLongValues
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract value
    OrtValue* values;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,1,allocator,&values));

    jlongArray output = createLongArrayFromTensor(jniEnv, api, values);

    api->ReleaseValue(values);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxMap
 * Method:    getFloatValues
 * Signature: (JJ)[F
 */
JNIEXPORT jfloatArray JNICALL Java_ai_onnxruntime_OnnxMap_getFloatValues
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract value
    OrtValue* values;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,1,allocator,&values));

    jfloatArray output = createFloatArrayFromTensor(jniEnv, api, values);

    api->ReleaseValue(values);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxMap
 * Method:    getDoubleValues
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_onnxruntime_OnnxMap_getDoubleValues
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract value
    OrtValue* values;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,1,allocator,&values));

    jdoubleArray output = createDoubleArrayFromTensor(jniEnv, api, values);

    api->ReleaseValue(values);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxMap
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OnnxMap_close
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseValue((OrtValue*)handle);
}

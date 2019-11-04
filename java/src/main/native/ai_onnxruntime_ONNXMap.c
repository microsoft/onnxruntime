/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "ONNXUtil.h"
#include "ai_onnxruntime_ONNXMap.h"

/*
 * Class:     ai_onnxruntime_ONNXMap
 * Method:    getStringKeys
 * Signature: (J)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_ONNXMap_getStringKeys
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract key
    OrtValue* keys;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,0,allocator,&keys));

    // Convert to Java String array
    jobjectArray output = createStringArrayFromTensor(jniEnv, api, allocator, keys);

    api->ReleaseValue(keys);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXMap
 * Method:    getLongKeys
 * Signature: (JJ)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_ONNXMap_getLongKeys
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract key
    OrtValue* keys;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,0,allocator,&keys));

    jlongArray output = createLongArrayFromTensor(jniEnv, api, keys);

    api->ReleaseValue(keys);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXMap
 * Method:    getStringValues
 * Signature: (JJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_ONNXMap_getStringValues
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract value
    OrtValue* values;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,1,allocator,&values));

    // Convert to Java String array
    jobjectArray output = createStringArrayFromTensor(jniEnv, api, allocator, values);

    api->ReleaseValue(values);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXMap
 * Method:    getLongValues
 * Signature: (JJ)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_ONNXMap_getLongValues
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract value
    OrtValue* values;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,1,allocator,&values));

    jlongArray output = createLongArrayFromTensor(jniEnv, api, values);

    api->ReleaseValue(values);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXMap
 * Method:    getFloatValues
 * Signature: (JJ)[F
 */
JNIEXPORT jfloatArray JNICALL Java_ai_onnxruntime_ONNXMap_getFloatValues
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract value
    OrtValue* values;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,1,allocator,&values));

    jfloatArray output = createFloatArrayFromTensor(jniEnv, api, values);

    api->ReleaseValue(values);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXMap
 * Method:    getDoubleValues
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_onnxruntime_ONNXMap_getDoubleValues
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract value
    OrtValue* values;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,1,allocator,&values));

    jdoubleArray output = createDoubleArrayFromTensor(jniEnv, api, values);

    api->ReleaseValue(values);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXMap
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXMap_close
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseValue((OrtValue*)handle);
}

/*
 * Copyright (c) 2019, 2021, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OnnxSequence.h"
/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getStringKeys
 * Signature: (JJI)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OnnxSequence_getStringKeys
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract keys from element
    OrtValue* keys;
    checkOrtStatus(jniEnv,api,api->GetValue(element,0,allocator,&keys));

    // Convert to Java String array
    jobjectArray output = createStringArrayFromTensor(jniEnv, api, allocator, keys);

    api->ReleaseValue(keys);
    api->ReleaseValue(element);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getLongKeys
 * Signature: (JJI)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_OnnxSequence_getLongKeys
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract keys from element
    OrtValue* keys;
    checkOrtStatus(jniEnv,api,api->GetValue(element,0,allocator,&keys));

    jlongArray output = createLongArrayFromTensor(jniEnv, api, keys);

    api->ReleaseValue(keys);
    api->ReleaseValue(element);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getStringValues
 * Signature: (JJI)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OnnxSequence_getStringValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract values from element
    OrtValue* values;
    checkOrtStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

    // Convert to Java String array
    jobjectArray output = createStringArrayFromTensor(jniEnv, api, allocator, values);

    api->ReleaseValue(values);
    api->ReleaseValue(element);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getLongValues
 * Signature: (JJI)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_OnnxSequence_getLongValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract values from element
    OrtValue* values;
    checkOrtStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

    jlongArray output = createLongArrayFromTensor(jniEnv, api, values);

    api->ReleaseValue(values);
    api->ReleaseValue(element);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getFloatValues
 * Signature: (JJI)[F
 */
JNIEXPORT jfloatArray JNICALL Java_ai_onnxruntime_OnnxSequence_getFloatValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract values from element
    OrtValue* values;
    checkOrtStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

    jfloatArray output = createFloatArrayFromTensor(jniEnv, api,values);

    api->ReleaseValue(values);
    api->ReleaseValue(element);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getDoubleValues
 * Signature: (JJI)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_onnxruntime_OnnxSequence_getDoubleValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkOrtStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract values from element
    OrtValue* values;
    checkOrtStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

    jdoubleArray output = createDoubleArrayFromTensor(jniEnv, api,values);

    api->ReleaseValue(values);
    api->ReleaseValue(element);

    return output;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getStrings
 * Signature: (JJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OnnxSequence_getStrings
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Get the element count of this sequence
    size_t count;
    checkOrtStatus(jniEnv,api,api->GetValueCount(sequence,&count));

    jclass stringClazz = (*jniEnv)->FindClass(jniEnv,"java/lang/String");
    jobjectArray outputArray = (*jniEnv)->NewObjectArray(jniEnv,safecast_size_t_to_jsize(count),stringClazz, NULL);
    for (size_t i = 0; i < count; i++) {
        // Extract element
        OrtValue* element;
        checkOrtStatus(jniEnv,api,api->GetValue(sequence,(int)i,allocator,&element));

        createStringFromStringTensor(jniEnv,api,allocator,element);

        api->ReleaseValue(element);
    }

    return outputArray;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getLongs
 * Signature: (JJ)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_OnnxSequence_getLongs
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Get the element count of this sequence
    size_t count;
    checkOrtStatus(jniEnv,api,api->GetValueCount(sequence,&count));

    int64_t* values;
    checkOrtStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(int64_t)*count,(void**)&values));

    for (size_t i = 0; i < count; i++) {
        // Extract element
        OrtValue* element;
        checkOrtStatus(jniEnv,api,api->GetValue(sequence,(int)i,allocator,&element));

        // Extract the values
        int64_t* arr;
        checkOrtStatus(jniEnv,api,api->GetTensorMutableData(element,(void**)&arr));
        values[i] = arr[0];

        api->ReleaseValue(element);
    }

    jlongArray outputArray = (*jniEnv)->NewLongArray(jniEnv,safecast_size_t_to_jsize(count));
    (*jniEnv)->SetLongArrayRegion(jniEnv, outputArray,0,safecast_size_t_to_jsize(count),(jlong*)values);

    checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));

    return outputArray;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getFloats
 * Signature: (JJ)[F
 */
JNIEXPORT jfloatArray JNICALL Java_ai_onnxruntime_OnnxSequence_getFloats
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Get the element count of this sequence
    size_t count;
    checkOrtStatus(jniEnv,api,api->GetValueCount(sequence,&count));

    float* values;
    checkOrtStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(float)*count,(void**)&values));

    for (size_t i = 0; i < count; i++) {
        // Extract element
        OrtValue* element;
        checkOrtStatus(jniEnv,api,api->GetValue(sequence,(int)i,allocator,&element));

        // Extract the values
        float* arr;
        checkOrtStatus(jniEnv,api,api->GetTensorMutableData(element,(void**)&arr));
        values[i] = arr[0];

        api->ReleaseValue(element);
    }

    jfloatArray outputArray = (*jniEnv)->NewFloatArray(jniEnv,safecast_size_t_to_jsize(count));
    (*jniEnv)->SetFloatArrayRegion(jniEnv,outputArray,0,safecast_size_t_to_jsize(count),values);

    checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));

    return outputArray;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getDoubles
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_onnxruntime_OnnxSequence_getDoubles
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Get the element count of this sequence
    size_t count;
    checkOrtStatus(jniEnv,api,api->GetValueCount(sequence,&count));

    double* values;
    checkOrtStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(double)*count,(void**)&values));

    for (size_t i = 0; i < count; i++) {
        // Extract element
        OrtValue* element;
        checkOrtStatus(jniEnv,api,api->GetValue(sequence,(int)i,allocator,&element));

        // Extract the values
        double* arr;
        checkOrtStatus(jniEnv,api,api->GetTensorMutableData(element,(void**)&arr));
        values[i] = arr[0];

        api->ReleaseValue(element);
    }

    jdoubleArray outputArray = (*jniEnv)->NewDoubleArray(jniEnv,safecast_size_t_to_jsize(count));
    (*jniEnv)->SetDoubleArrayRegion(jniEnv,outputArray,0,safecast_size_t_to_jsize(count),values);

    checkOrtStatus(jniEnv,api,api->AllocatorFree(allocator,values));

    return outputArray;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OnnxSequence_close(JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseValue((OrtValue*)handle);
}

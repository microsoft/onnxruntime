/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "ONNXUtil.h"
#include "ai_onnxruntime_ONNXSequence.h"
/*
 * Class:     ai_onnxruntime_ONNXSequence
 * Method:    getStringKeys
 * Signature: (JJI)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_ONNXSequence_getStringKeys
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract keys from element
    OrtValue* keys;
    checkONNXStatus(jniEnv,api,api->GetValue(element,0,allocator,&keys));

    // Convert to Java String array
    jobjectArray output = createStringArrayFromTensor(jniEnv, api, allocator, keys);

    api->ReleaseValue(keys);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXSequence
 * Method:    getLongKeys
 * Signature: (JJI)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_ONNXSequence_getLongKeys
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract keys from element
    OrtValue* keys;
    checkONNXStatus(jniEnv,api,api->GetValue(element,0,allocator,&keys));

    jlongArray output = createLongArrayFromTensor(jniEnv, api, keys);

    api->ReleaseValue(keys);
    api->ReleaseValue(element);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXSequence
 * Method:    getStringValues
 * Signature: (JJI)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_ONNXSequence_getStringValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract values from element
    OrtValue* values;
    checkONNXStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

    // Convert to Java String array
    jobjectArray output = createStringArrayFromTensor(jniEnv, api, allocator, values);

    api->ReleaseValue(values);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXSequence
 * Method:    getLongValues
 * Signature: (JJI)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_ONNXSequence_getLongValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract values from element
    OrtValue* values;
    checkONNXStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

    jlongArray output = createLongArrayFromTensor(jniEnv, api, values);

    api->ReleaseValue(values);
    api->ReleaseValue(element);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXSequence
 * Method:    getFloatValues
 * Signature: (JJI)[F
 */
JNIEXPORT jfloatArray JNICALL Java_ai_onnxruntime_ONNXSequence_getFloatValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract values from element
    OrtValue* values;
    checkONNXStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

    jfloatArray output = createFloatArrayFromTensor(jniEnv, api,values);

    api->ReleaseValue(values);
    api->ReleaseValue(element);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXSequence
 * Method:    getDoubleValues
 * Signature: (JJI)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_onnxruntime_ONNXSequence_getDoubleValues
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jint index) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Extract element
    OrtValue* element;
    checkONNXStatus(jniEnv,api,api->GetValue((OrtValue*)handle,index,allocator,&element));

    // Extract values from element
    OrtValue* values;
    checkONNXStatus(jniEnv,api,api->GetValue(element,1,allocator,&values));

    jdoubleArray output = createDoubleArrayFromTensor(jniEnv, api,values);

    api->ReleaseValue(values);
    api->ReleaseValue(element);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXSequence
 * Method:    getStrings
 * Signature: (JJ)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_ONNXSequence_getStrings
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Get the element count of this sequence
    size_t count;
    checkONNXStatus(jniEnv,api,api->GetValueCount(sequence,&count));

    jclass stringClazz = (*jniEnv)->FindClass(jniEnv,"java/lang/String");
    jobjectArray outputArray = (*jniEnv)->NewObjectArray(jniEnv,count,stringClazz,NULL);
    for (int i = 0; i < count; i++) {
        // Extract element
        OrtValue* element;
        checkONNXStatus(jniEnv,api,api->GetValue(sequence,i,allocator,&element));

        createStringFromStringTensor(jniEnv,api,allocator,element);

        api->ReleaseValue(element);
    }

    return outputArray;
}

/*
 * Class:     ai_onnxruntime_ONNXSequence
 * Method:    getLongs
 * Signature: (JJ)[J
 */
JNIEXPORT jlongArray JNICALL Java_ai_onnxruntime_ONNXSequence_getLongs
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Get the element count of this sequence
    size_t count;
    checkONNXStatus(jniEnv,api,api->GetValueCount(sequence,&count));

    int64_t* values;
    checkONNXStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(int64_t)*count,(void**)&values));

    for (int i = 0; i < count; i++) {
        // Extract element
        OrtValue* element;
        checkONNXStatus(jniEnv,api,api->GetValue(sequence,i,allocator,&element));

        // Extract the values
        int64_t* arr;
        checkONNXStatus(jniEnv,api,api->GetTensorMutableData(element,(void**)&arr));
        values[i] = arr[0];

        api->ReleaseValue(element);
    }

    jlongArray outputArray = (*jniEnv)->NewLongArray(jniEnv,count);
    (*jniEnv)->SetLongArrayRegion(jniEnv,outputArray,0,count,(jlong*)values);

    checkONNXStatus(jniEnv,api,api->AllocatorFree(allocator,values));

    return outputArray;
}

/*
 * Class:     ai_onnxruntime_ONNXSequence
 * Method:    getFloats
 * Signature: (JJ)[F
 */
JNIEXPORT jfloatArray JNICALL Java_ai_onnxruntime_ONNXSequence_getFloats
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Get the element count of this sequence
    size_t count;
    checkONNXStatus(jniEnv,api,api->GetValueCount(sequence,&count));

    float* values;
    checkONNXStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(float)*count,(void**)&values));

    for (int i = 0; i < count; i++) {
        // Extract element
        OrtValue* element;
        checkONNXStatus(jniEnv,api,api->GetValue(sequence,i,allocator,&element));

        // Extract the values
        float* arr;
        checkONNXStatus(jniEnv,api,api->GetTensorMutableData(element,(void**)&arr));
        values[i] = arr[0];

        api->ReleaseValue(element);
    }

    jfloatArray outputArray = (*jniEnv)->NewFloatArray(jniEnv,count);
    (*jniEnv)->SetFloatArrayRegion(jniEnv,outputArray,0,count,values);

    checkONNXStatus(jniEnv,api,api->AllocatorFree(allocator,values));

    return outputArray;
}

/*
 * Class:     ai_onnxruntime_ONNXSequence
 * Method:    getDoubles
 * Signature: (JJ)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_ai_onnxruntime_ONNXSequence_getDoubles
  (JNIEnv *jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtValue* sequence = (OrtValue*) handle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Get the element count of this sequence
    size_t count;
    checkONNXStatus(jniEnv,api,api->GetValueCount(sequence,&count));

    double* values;
    checkONNXStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(double)*count,(void**)&values));

    for (int i = 0; i < count; i++) {
        // Extract element
        OrtValue* element;
        checkONNXStatus(jniEnv,api,api->GetValue(sequence,i,allocator,&element));

        // Extract the values
        double* arr;
        checkONNXStatus(jniEnv,api,api->GetTensorMutableData(element,(void**)&arr));
        values[i] = arr[0];

        api->ReleaseValue(element);
    }

    jdoubleArray outputArray = (*jniEnv)->NewDoubleArray(jniEnv,count);
    (*jniEnv)->SetDoubleArrayRegion(jniEnv,outputArray,0,count,values);

    checkONNXStatus(jniEnv,api,api->AllocatorFree(allocator,values));

    return outputArray;
}

/*
 * Class:     ai_onnxruntime_ONNXSequence
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSequence_close(JNIEnv * jniEnv, jobject obj, jlong apiHandle, jlong handle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseValue((OrtValue*)handle);
}

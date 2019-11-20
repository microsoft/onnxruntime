/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <math.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "ONNXUtil.h"
#include "ai_onnxruntime_ONNXTensor.h"

/*
 * Class:     ai_onnxruntime_ONNXTensor
 * Method:    getFloat
 * Signature: (JI)F
 */
JNIEXPORT jfloat JNICALL Java_ai_onnxruntime_ONNXTensor_getFloat
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxType) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    if (onnxType == 9) {
        uint16_t* arr;
        checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
        jfloat floatVal = convertHalfToFloat(*arr);
        return floatVal;
    } else if (onnxType == 10) {
        jfloat* arr;
        checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
        return *arr;
    } else {
        return NAN;
    }
}

/*
 * Class:     ai_onnxruntime_ONNXTensor
 * Method:    getDouble
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_ai_onnxruntime_ONNXTensor_getDouble
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    jdouble* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return *arr;
}

/*
 * Class:     ai_onnxruntime_ONNXTensor
 * Method:    getByte
 * Signature: (JI)B
 */
JNIEXPORT jbyte JNICALL Java_ai_onnxruntime_ONNXTensor_getByte
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxType) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  if (onnxType == 1) {
    uint8_t* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jbyte) *arr;
  } else if (onnxType == 2) {
    int8_t* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jbyte) *arr;
  } else {
    return (jbyte) 0;
  }
}

/*
 * Class:     ai_onnxruntime_ONNXTensor
 * Method:    getShort
 * Signature: (JI)S
 */
JNIEXPORT jshort JNICALL Java_ai_onnxruntime_ONNXTensor_getShort
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxType) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  if (onnxType == 3) {
    uint16_t* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jshort) *arr;
  } else if (onnxType == 4) {
    int16_t* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jshort) *arr;
  } else {
    return (jshort) 0;
  }
}

/*
 * Class:     ai_onnxruntime_ONNXTensor
 * Method:    getInt
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_ai_onnxruntime_ONNXTensor_getInt
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxType) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  if (onnxType == 5) {
    uint32_t* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jint) *arr;
  } else if (onnxType == 6) {
    int32_t* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jint) *arr;
  } else {
    return (jint) 0;
  }
}

/*
 * Class:     ai_onnxruntime_ONNXTensor
 * Method:    getLong
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_ONNXTensor_getLong
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxType) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  if (onnxType == 7) {
    uint64_t* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jlong) *arr;
  } else if (onnxType == 8) {
    int64_t* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jlong) *arr;
  } else {
    return (jlong) 0;
  }
}

/*
 * Class:     ai_onnxruntime_ONNXTensor
 * Method:    getString
 * Signature: (JJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_ONNXTensor_getString
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    // Extract a String array - if this becomes a performance issue we'll refactor later.
    jobjectArray outputArray = createStringArrayFromTensor(jniEnv,api, (OrtAllocator*) allocatorHandle, (OrtValue*) handle);

    // Get reference to the string
    jobject output = (*jniEnv)->GetObjectArrayElement(jniEnv, outputArray, 0);

    // Free array
    (*jniEnv)->DeleteLocalRef(jniEnv,outputArray);

    return output;
}

/*
 * Class:     ai_onnxruntime_ONNXTensor
 * Method:    getBool
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_ai_onnxruntime_ONNXTensor_getBool
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    jboolean* arr;
    checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return *arr;
}

/*
 * Class:     ai_onnxruntime_ONNXTensor
 * Method:    getArray
 * Signature: (JJLjava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXTensor_getArray
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jobject carrier) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtTensorTypeAndShapeInfo* info;
    checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape((OrtValue*) handle, &info));
    size_t dimensions;
    checkONNXStatus(jniEnv,api,api->GetDimensionsCount(info,&dimensions));
    size_t arrSize;
    checkONNXStatus(jniEnv,api,api->GetTensorShapeElementCount(info,&arrSize));
    ONNXTensorElementDataType onnxTypeEnum;
    checkONNXStatus(jniEnv,api,api->GetTensorElementType(info,&onnxTypeEnum));
    api->ReleaseTensorTypeAndShapeInfo(info);

    if (onnxTypeEnum == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        copyStringTensorToArray(jniEnv,api, (OrtAllocator*) allocatorHandle, (OrtValue*)handle, arrSize, carrier);
    } else {
        uint8_t* arr;
        checkONNXStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
        copyTensorToJava(jniEnv,onnxTypeEnum,arr,arrSize,dimensions,(jarray)carrier);
    }
}

/*
 * Class:     ai_onnxruntime_ONNXTensor
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXTensor_close(JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseValue((OrtValue*)handle);
}

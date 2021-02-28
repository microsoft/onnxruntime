/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <math.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OnnxTensor.h"

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    createTensor
 * Signature: (JJLjava/lang/Object;[JI)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxTensor_createTensor
        (JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jlong allocatorHandle, jobject dataObj, jlongArray shape, jint onnxTypeJava) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    // Convert type to ONNX C enum
    ONNXTensorElementDataType onnxType = convertToONNXDataFormat(onnxTypeJava);

    // Extract the shape information
    jboolean mkCopy;
    jlong* shapeArr = (*jniEnv)->GetLongArrayElements(jniEnv,shape,&mkCopy);
    jsize shapeLen = (*jniEnv)->GetArrayLength(jniEnv,shape);

    // Create the OrtValue
    OrtValue* ortValue;
    checkOrtStatus(jniEnv, api, api->CreateTensorAsOrtValue(allocator,(int64_t*)shapeArr,shapeLen,onnxType,&ortValue));
    (*jniEnv)->ReleaseLongArrayElements(jniEnv,shape,shapeArr,JNI_ABORT);

    // Get a reference to the OrtValue's data
    uint8_t* tensorData;
    checkOrtStatus(jniEnv, api, api->GetTensorMutableData(ortValue, (void**) &tensorData));

    // Extract the tensor shape information
    OrtTensorTypeAndShapeInfo* info;
    checkOrtStatus(jniEnv,api,api->GetTensorTypeAndShape(ortValue, &info));
    size_t dimensions;
    checkOrtStatus(jniEnv,api,api->GetDimensionsCount(info,&dimensions));
    size_t arrSize;
    checkOrtStatus(jniEnv,api,api->GetTensorShapeElementCount(info,&arrSize));
    ONNXTensorElementDataType onnxTypeEnum;
    checkOrtStatus(jniEnv,api,api->GetTensorElementType(info,&onnxTypeEnum));
    api->ReleaseTensorTypeAndShapeInfo(info);

    // Copy the java array into the tensor
    copyJavaToTensor(jniEnv, onnxType, tensorData, arrSize, dimensions, dataObj);

    // Return the pointer to the OrtValue
    return (jlong) ortValue;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    createTensorFromBuffer
 * Signature: (JJLjava/nio/Buffer;IJ[JI)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxTensor_createTensorFromBuffer
        (JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jlong allocatorHandle, jobject buffer, jint bufferPos, jlong bufferSize, jlongArray shape, jint onnxTypeJava) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    const OrtMemoryInfo* allocatorInfo;
    checkOrtStatus(jniEnv, api, api->AllocatorGetInfo(allocator,&allocatorInfo));

    // Convert type to ONNX C enum
    ONNXTensorElementDataType onnxType = convertToONNXDataFormat(onnxTypeJava);

    // Extract the buffer
    char* bufferArr = (char*)(*jniEnv)->GetDirectBufferAddress(jniEnv,buffer);
    // Increment by bufferPos bytes
    bufferArr = bufferArr + bufferPos;

    // Extract the shape information
    jboolean mkCopy;
    jlong* shapeArr = (*jniEnv)->GetLongArrayElements(jniEnv,shape,&mkCopy);
    jsize shapeLen = (*jniEnv)->GetArrayLength(jniEnv,shape);

    // Create the OrtValue
    OrtValue* ortValue;
    checkOrtStatus(jniEnv, api, api->CreateTensorWithDataAsOrtValue(allocatorInfo,bufferArr,bufferSize,(int64_t*)shapeArr,shapeLen,onnxType,&ortValue));
    (*jniEnv)->ReleaseLongArrayElements(jniEnv,shape,shapeArr,JNI_ABORT);

    // Return the pointer to the OrtValue
    return (jlong) ortValue;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    createString
 * Signature: (JJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxTensor_createString
        (JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jlong allocatorHandle, jstring input) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Extract the shape information
    int64_t* shapeArr;
    checkOrtStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(int64_t),(void**)&shapeArr));
    shapeArr[0] = 1;

    // Create the OrtValue
    OrtValue* ortValue;
    checkOrtStatus(jniEnv, api, api->CreateTensorAsOrtValue(allocator,shapeArr,1,ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,&ortValue));

    // Release the shape
    checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator,shapeArr));

    // Create the buffer for the Java string
    const char* stringBuffer = (*jniEnv)->GetStringUTFChars(jniEnv,input,NULL);

    // Assign the strings into the Tensor
    checkOrtStatus(jniEnv, api, api->FillStringTensor(ortValue,&stringBuffer,1));

    // Release the Java string
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,input,stringBuffer);

    return (jlong) ortValue;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    createStringTensor
 * Signature: (JJ[Ljava/lang/Object;[J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxTensor_createStringTensor
        (JNIEnv * jniEnv, jclass jobj, jlong apiHandle, jlong allocatorHandle, jobjectArray stringArr, jlongArray shape) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Extract the shape information
    jboolean mkCopy;
    jlong* shapeArr = (*jniEnv)->GetLongArrayElements(jniEnv,shape,&mkCopy);
    jsize shapeLen = (*jniEnv)->GetArrayLength(jniEnv,shape);

    // Array length
    jsize length = (*jniEnv)->GetArrayLength(jniEnv, stringArr);

    // Create the OrtValue
    OrtValue* ortValue;
    checkOrtStatus(jniEnv, api, api->CreateTensorAsOrtValue(allocator,(int64_t*)shapeArr,shapeLen,ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,&ortValue));
    (*jniEnv)->ReleaseLongArrayElements(jniEnv,shape,shapeArr,JNI_ABORT);

    // Create the buffers for the Java strings
    const char** strings;
    checkOrtStatus(jniEnv, api, api->AllocatorAlloc(allocator,sizeof(char*)*length,(void**)&strings));
    jobject* javaStrings;
    checkOrtStatus(jniEnv, api, api->AllocatorAlloc(allocator,sizeof(jobject)*length,(void**)&javaStrings));

    // Copy the java strings into the buffers
    for (jsize i = 0; i < length; i++) {
        javaStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv,stringArr,i);
        strings[i] = (*jniEnv)->GetStringUTFChars(jniEnv,javaStrings[i],NULL);
    }

    // Assign the strings into the Tensor
    checkOrtStatus(jniEnv, api, api->FillStringTensor(ortValue,strings,length));

    // Release the Java strings
    for (int i = 0; i < length; i++) {
        (*jniEnv)->ReleaseStringUTFChars(jniEnv,javaStrings[i],strings[i]);
    }

    // Release the buffers
    checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, (void*)strings));
    checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, javaStrings));

    return (jlong) ortValue;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getBuffer
 * Signature: (JJ)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_ai_onnxruntime_OnnxTensor_getBuffer
        (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtTensorTypeAndShapeInfo* info;
    checkOrtStatus(jniEnv,api,api->GetTensorTypeAndShape((OrtValue*) handle, &info));
    size_t arrSize;
    checkOrtStatus(jniEnv,api,api->GetTensorShapeElementCount(info,&arrSize));
    ONNXTensorElementDataType onnxTypeEnum;
    checkOrtStatus(jniEnv,api,api->GetTensorElementType(info,&onnxTypeEnum));
    api->ReleaseTensorTypeAndShapeInfo(info);

    size_t typeSize = onnxTypeSize(onnxTypeEnum);
    size_t sizeBytes = arrSize*typeSize;

    uint8_t* arr;
    checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));

    return (*jniEnv)->NewDirectByteBuffer(jniEnv, arr, sizeBytes);
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getFloat
 * Signature: (JI)F
 */
JNIEXPORT jfloat JNICALL Java_ai_onnxruntime_OnnxTensor_getFloat
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxType) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    if (onnxType == 9) {
        uint16_t* arr;
        checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
        jfloat floatVal = convertHalfToFloat(*arr);
        return floatVal;
    } else if (onnxType == 10) {
        jfloat* arr;
        checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
        return *arr;
    } else {
        return NAN;
    }
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getDouble
 * Signature: (J)D
 */
JNIEXPORT jdouble JNICALL Java_ai_onnxruntime_OnnxTensor_getDouble
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    jdouble* arr;
    checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return *arr;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getByte
 * Signature: (JI)B
 */
JNIEXPORT jbyte JNICALL Java_ai_onnxruntime_OnnxTensor_getByte
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxType) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  if (onnxType == 1) {
    uint8_t* arr;
    checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jbyte) *arr;
  } else if (onnxType == 2) {
    int8_t* arr;
    checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jbyte) *arr;
  } else {
    return (jbyte) 0;
  }
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getShort
 * Signature: (JI)S
 */
JNIEXPORT jshort JNICALL Java_ai_onnxruntime_OnnxTensor_getShort
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxType) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  if (onnxType == 3) {
    uint16_t* arr;
    checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jshort) *arr;
  } else if (onnxType == 4) {
    int16_t* arr;
    checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jshort) *arr;
  } else {
    return (jshort) 0;
  }
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getInt
 * Signature: (JI)I
 */
JNIEXPORT jint JNICALL Java_ai_onnxruntime_OnnxTensor_getInt
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxType) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  if (onnxType == 5) {
    uint32_t* arr;
    checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jint) *arr;
  } else if (onnxType == 6) {
    int32_t* arr;
    checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jint) *arr;
  } else {
    return (jint) 0;
  }
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getLong
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxTensor_getLong
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint onnxType) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
  if (onnxType == 7) {
    uint64_t* arr;
    checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jlong) *arr;
  } else if (onnxType == 8) {
    int64_t* arr;
    checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return (jlong) *arr;
  } else {
    return (jlong) 0;
  }
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getString
 * Signature: (JJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_OnnxTensor_getString
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
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getBool
 * Signature: (J)Z
 */
JNIEXPORT jboolean JNICALL Java_ai_onnxruntime_OnnxTensor_getBool
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    jboolean* arr;
    checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
    return *arr;
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    getArray
 * Signature: (JJLjava/lang/Object;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OnnxTensor_getArray
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong allocatorHandle, jobject carrier) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtTensorTypeAndShapeInfo* info;
    checkOrtStatus(jniEnv,api,api->GetTensorTypeAndShape((OrtValue*) handle, &info));
    size_t dimensions;
    checkOrtStatus(jniEnv,api,api->GetDimensionsCount(info,&dimensions));
    size_t arrSize;
    checkOrtStatus(jniEnv,api,api->GetTensorShapeElementCount(info,&arrSize));
    ONNXTensorElementDataType onnxTypeEnum;
    checkOrtStatus(jniEnv,api,api->GetTensorElementType(info,&onnxTypeEnum));
    api->ReleaseTensorTypeAndShapeInfo(info);

    if (onnxTypeEnum == ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING) {
        copyStringTensorToArray(jniEnv,api, (OrtAllocator*) allocatorHandle, (OrtValue*)handle, arrSize, carrier);
    } else {
        uint8_t* arr;
        checkOrtStatus(jniEnv,api,api->GetTensorMutableData((OrtValue*)handle,(void**)&arr));
        copyTensorToJava(jniEnv,onnxTypeEnum,arr,arrSize,dimensions,(jarray)carrier);
    }
}

/*
 * Class:     ai_onnxruntime_OnnxTensor
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OnnxTensor_close(JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseValue((OrtValue*)handle);
}

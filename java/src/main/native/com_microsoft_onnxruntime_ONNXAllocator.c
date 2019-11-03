/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "ONNXUtil.h"
#include "com_microsoft_onnxruntime_ONNXAllocator.h"

/*
 * Class:     com_microsoft_onnxruntime_ONNXAllocator
 * Method:    createAllocator
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_com_microsoft_onnxruntime_ONNXAllocator_createAllocator
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator;
    checkONNXStatus(jniEnv,api,api->GetAllocatorWithDefaultOptions(&allocator));
    return (jlong)allocator;
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXAllocator
 * Method:    createTensor
 * Signature: (JJLjava/lang/Object;[JI)J
 */
JNIEXPORT jlong JNICALL Java_com_microsoft_onnxruntime_ONNXAllocator_createTensor
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong allocatorHandle, jobject dataObj, jlongArray shape, jint onnxTypeJava) {
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
    checkONNXStatus(jniEnv, api, api->CreateTensorAsOrtValue(allocator,(int64_t*)shapeArr,shapeLen,onnxType,&ortValue));
    (*jniEnv)->ReleaseLongArrayElements(jniEnv,shape,shapeArr,JNI_ABORT);

    // Get a reference to the OrtValue's data
    uint8_t* tensorData;
    checkONNXStatus(jniEnv, api, api->GetTensorMutableData(ortValue, (void**) &tensorData));

    // Extract the tensor shape information
    OrtTensorTypeAndShapeInfo* info;
    checkONNXStatus(jniEnv,api,api->GetTensorTypeAndShape(ortValue, &info));
    size_t dimensions;
    checkONNXStatus(jniEnv,api,api->GetDimensionsCount(info,&dimensions));
    size_t arrSize;
    checkONNXStatus(jniEnv,api,api->GetTensorShapeElementCount(info,&arrSize));
    ONNXTensorElementDataType onnxTypeEnum;
    checkONNXStatus(jniEnv,api,api->GetTensorElementType(info,&onnxTypeEnum));
    api->ReleaseTensorTypeAndShapeInfo(info);

    // Copy the java array into the tensor
    copyJavaToTensor(jniEnv, onnxType, tensorData, arrSize, dimensions, dataObj);

    // Return the pointer to the OrtValue
    return (jlong) ortValue;
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXAllocator
 * Method:    createTensorFromBuffer
 * Signature: (JJLjava/nio/Buffer;J[JI)J
 */
JNIEXPORT jlong JNICALL Java_com_microsoft_onnxruntime_ONNXAllocator_createTensorFromBuffer
        (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong allocatorHandle, jobject buffer, jlong bufferSize, jlongArray shape, jint onnxTypeJava) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    const OrtMemoryInfo* allocatorInfo;
    checkONNXStatus(jniEnv, api, api->AllocatorGetInfo(allocator,&allocatorInfo));

    // Convert type to ONNX C enum
    ONNXTensorElementDataType onnxType = convertToONNXDataFormat(onnxTypeJava);

    // Extract the buffer
    void* bufferArr = (*jniEnv)->GetDirectBufferAddress(jniEnv,buffer);

    // Extract the shape information
    jboolean mkCopy;
    jlong* shapeArr = (*jniEnv)->GetLongArrayElements(jniEnv,shape,&mkCopy);
    jsize shapeLen = (*jniEnv)->GetArrayLength(jniEnv,shape);

    // Create the OrtValue
    OrtValue* ortValue;
    checkONNXStatus(jniEnv, api, api->CreateTensorWithDataAsOrtValue(allocatorInfo,bufferArr,bufferSize,(int64_t*)shapeArr,shapeLen,onnxType,&ortValue));
    (*jniEnv)->ReleaseLongArrayElements(jniEnv,shape,shapeArr,JNI_ABORT);

    // Return the pointer to the OrtValue
    return (jlong) ortValue;
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXAllocator
 * Method:    createString
 * Signature: (JJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_com_microsoft_onnxruntime_ONNXAllocator_createString
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong allocatorHandle, jstring input) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;

    // Extract the shape information
    int64_t* shapeArr;
    checkONNXStatus(jniEnv,api,api->AllocatorAlloc(allocator,sizeof(int64_t),(void**)&shapeArr));
    shapeArr[0] = 1;

    // Create the OrtValue
    OrtValue* ortValue;
    checkONNXStatus(jniEnv, api, api->CreateTensorAsOrtValue(allocator,shapeArr,1,ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,&ortValue));

    // Release the shape
    checkONNXStatus(jniEnv, api, api->AllocatorFree(allocator,shapeArr));

    // Create the buffer for the Java string
    const char* stringBuffer = (*jniEnv)->GetStringUTFChars(jniEnv,input,NULL);

    // Assign the strings into the Tensor
    checkONNXStatus(jniEnv, api, api->FillStringTensor(ortValue,&stringBuffer,1));

    // Release the Java string
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,input,stringBuffer);

    return (jlong) ortValue;
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXAllocator
 * Method:    createStringTensor
 * Signature: (JJ[Ljava/lang/Object;[J)J
 */
JNIEXPORT jlong JNICALL Java_com_microsoft_onnxruntime_ONNXAllocator_createStringTensor
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong allocatorHandle, jobjectArray stringArr, jlongArray shape) {
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
    checkONNXStatus(jniEnv, api, api->CreateTensorAsOrtValue(allocator,(int64_t*)shapeArr,shapeLen,ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,&ortValue));
    (*jniEnv)->ReleaseLongArrayElements(jniEnv,shape,shapeArr,JNI_ABORT);

    // Create the buffers for the Java strings
    const char** strings;
    checkONNXStatus(jniEnv, api, api->AllocatorAlloc(allocator,sizeof(char*)*length,(void**)&strings));
    jobject* javaStrings;
    checkONNXStatus(jniEnv, api, api->AllocatorAlloc(allocator,sizeof(jobject)*length,(void**)&javaStrings));

    // Copy the java strings into the buffers
    for (int i = 0; i < length; i++) {
        javaStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv,stringArr,i);
        strings[i] = (*jniEnv)->GetStringUTFChars(jniEnv,javaStrings[i],NULL);
    }

    // Assign the strings into the Tensor
    checkONNXStatus(jniEnv, api, api->FillStringTensor(ortValue,strings,length));

    // Release the Java strings
    for (int i = 0; i < length; i++) {
        (*jniEnv)->ReleaseStringUTFChars(jniEnv,javaStrings[i],strings[i]);
    }

    // Release the buffers
    checkONNXStatus(jniEnv, api, api->AllocatorFree(allocator, strings));
    checkONNXStatus(jniEnv, api, api->AllocatorFree(allocator, javaStrings));

    return (jlong) ortValue;
}


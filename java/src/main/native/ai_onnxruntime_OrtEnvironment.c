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
 * Method:    createTensor
 * Signature: (JJLjava/lang/Object;[JI)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtEnvironment_createTensor
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong allocatorHandle, jobject dataObj, jlongArray shape, jint onnxTypeJava) {
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
 * Class:     ai_onnxruntime_OrtEnvironment
 * Method:    createTensorFromBuffer
 * Signature: (JJLjava/nio/Buffer;J[JI)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtEnvironment_createTensorFromBuffer
        (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong allocatorHandle, jobject buffer, jlong bufferSize, jlongArray shape, jint onnxTypeJava) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    const OrtMemoryInfo* allocatorInfo;
    checkOrtStatus(jniEnv, api, api->AllocatorGetInfo(allocator,&allocatorInfo));

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
    checkOrtStatus(jniEnv, api, api->CreateTensorWithDataAsOrtValue(allocatorInfo,bufferArr,bufferSize,(int64_t*)shapeArr,shapeLen,onnxType,&ortValue));
    (*jniEnv)->ReleaseLongArrayElements(jniEnv,shape,shapeArr,JNI_ABORT);

    // Return the pointer to the OrtValue
    return (jlong) ortValue;
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment
 * Method:    createString
 * Signature: (JJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtEnvironment_createString
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong allocatorHandle, jstring input) {
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
 * Class:     ai_onnxruntime_OrtEnvironment
 * Method:    createStringTensor
 * Signature: (JJ[Ljava/lang/Object;[J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtEnvironment_createStringTensor
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong allocatorHandle, jobjectArray stringArr, jlongArray shape) {
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
    for (int i = 0; i < length; i++) {
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
    checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, strings));
    checkOrtStatus(jniEnv, api, api->AllocatorFree(allocator, javaStrings));

    return (jlong) ortValue;
}

/*
 * Class:     ai_onnxruntime_OrtEnvironment
 * Method:    getDefaultAllocator
 * Signature: ()J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtEnvironment_getDefaultAllocator
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator;
    checkOrtStatus(jniEnv,api,api->GetAllocatorWithDefaultOptions(&allocator));
    return (jlong)allocator;
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

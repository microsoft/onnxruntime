/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <math.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OnnxSparseTensor.h"

/*
 * Class:     ai_onnxruntime_OnnxSparseTensor
 * Method:    getIndexBuffer
 * Signature: (JJ)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_ai_onnxruntime_OnnxSparseTensor_getIndexBuffer
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtSparseFormat format;
    checkOrtStatus(jniEnv,api,api->GetSparseTensorFormat((OrtValue*) handle, &format));
    switch (format) {
        case ORT_SPARSE_COO: {
            OrtTensorTypeAndShapeInfo* info;
            checkOrtStatus(jniEnv,api,api->GetSparseTensorIndicesTypeShape((OrtValue*) handle, ORT_SPARSE_COO_INDICES, &info));
            size_t arrSize;
            checkOrtStatus(jniEnv,api,api->GetTensorShapeElementCount(info,&arrSize));
            ONNXTensorElementDataType onnxTypeEnum;
            checkOrtStatus(jniEnv,api,api->GetTensorElementType(info,&onnxTypeEnum));
            api->ReleaseTensorTypeAndShapeInfo(info);

            size_t typeSize = onnxTypeSize(onnxTypeEnum);
            size_t sizeBytes = arrSize*typeSize;

            uint8_t* arr;
            size_t indices_size;
            checkOrtStatus(jniEnv,api,api->GetSparseTensorIndices((OrtValue*)handle,ORT_SPARSE_COO_INDICES,&indices_size,(const void**)&arr));

            if (indices_size != arrSize) {
                throwOrtException(jniEnv,convertErrorCode(ORT_RUNTIME_EXCEPTION),"Unexpected size");
            }

            return (*jniEnv)->NewDirectByteBuffer(jniEnv, arr, sizeBytes);
        }
        case ORT_SPARSE_CSRC:
        case ORT_SPARSE_BLOCK_SPARSE:
        case ORT_SPARSE_UNDEFINED: {
            throwOrtException(jniEnv,convertErrorCode(ORT_NOT_IMPLEMENTED),"These types are unsupported - ORT_SPARSE_CSRC, ORT_SPARSE_BLOCK_SPARSE, ORT_SPARSE_UNDEFINED");
            break;
        }
    }
    return NULL;
}


/*
 * Class:     ai_onnxruntime_OnnxSparseTensor
 * Method:    getDataBuffer
 * Signature: (JJ)Ljava/nio/ByteBuffer;
 */
JNIEXPORT jobject JNICALL Java_ai_onnxruntime_OnnxSparseTensor_getDataBuffer
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtSparseFormat format;
    checkOrtStatus(jniEnv,api,api->GetSparseTensorFormat((OrtValue*) handle, &format));
    switch (format) {
        case ORT_SPARSE_COO: {
            OrtTensorTypeAndShapeInfo* info;
            checkOrtStatus(jniEnv,api,api->GetSparseTensorValuesTypeAndShape((OrtValue*) handle, &info));
            size_t arrSize;
            checkOrtStatus(jniEnv,api,api->GetTensorShapeElementCount(info,&arrSize));
            ONNXTensorElementDataType onnxTypeEnum;
            checkOrtStatus(jniEnv,api,api->GetTensorElementType(info,&onnxTypeEnum));
            api->ReleaseTensorTypeAndShapeInfo(info);

            size_t typeSize = onnxTypeSize(onnxTypeEnum);
            size_t sizeBytes = arrSize*typeSize;

            uint8_t* arr;
            checkOrtStatus(jniEnv,api,api->GetSparseTensorValues((OrtValue*)handle,(const void**)&arr));

            return (*jniEnv)->NewDirectByteBuffer(jniEnv, arr, sizeBytes);
        }
        case ORT_SPARSE_CSRC:
        case ORT_SPARSE_BLOCK_SPARSE:
        case ORT_SPARSE_UNDEFINED: {
            throwOrtException(jniEnv,convertErrorCode(ORT_NOT_IMPLEMENTED),"These types are unsupported - ORT_SPARSE_CSRC, ORT_SPARSE_BLOCK_SPARSE, ORT_SPARSE_UNDEFINED");
            break;
        }
    }

    return NULL;
}

/*
 * Class:     ai_onnxruntime_OnnxSparseTensor
 * Method:    close
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OnnxSparseTensor_close(JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseValue((OrtValue*)handle);
}

/*
 * Class:     ai_onnxruntime_OnnxSparseTensor
 * Method:    createSparseTensorFromBuffer
 * Signature: (JJLjava/nio/Buffer;IJLjava/nio/Buffer;IJ[J[JII)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxSparseTensor_createSparseTensorFromBuffer
  (JNIEnv * jniEnv, jclass cls, jlong apiHandle, jlong allocatorHandle,
  jobject indicesBuffer, jint indicesBufferPos, jlong indicesBufferSize,
  jobject dataBuffer, jint dataBufferPos, jlong dataBufferSize,
  jlongArray denseShape, jlongArray valuesShape, jint onnxTypeJava, jint sparsityTypeJava) {
    (void) cls; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
    const OrtMemoryInfo* allocatorInfo;
    checkOrtStatus(jniEnv, api, api->AllocatorGetInfo(allocator,&allocatorInfo));

    // Convert types to ONNX C enums
    ONNXTensorElementDataType onnxType = convertToONNXDataFormat(onnxTypeJava);
    OrtSparseFormat sparsityType = convertToOrtSparseFormat(sparsityTypeJava);

    // Extract the buffers
    char* indicesBufferArr = (char*)(*jniEnv)->GetDirectBufferAddress(jniEnv,indicesBuffer);
    char* dataBufferArr = (char*)(*jniEnv)->GetDirectBufferAddress(jniEnv,dataBuffer);
    // Increment by bufferPos bytes
    indicesBufferArr = indicesBufferArr + indicesBufferPos;
    dataBufferArr = dataBufferArr + dataBufferPos;

    // Extract the dense shape information
    jboolean mkCopy;
    jlong* shapeArr = (*jniEnv)->GetLongArrayElements(jniEnv,denseShape,&mkCopy);
    jsize shapeLen = (*jniEnv)->GetArrayLength(jniEnv,denseShape);

    // Extract the value shape information
    jlong* valuesShapeArr = (*jniEnv)->GetLongArrayElements(jniEnv,valuesShape,&mkCopy);
    jsize valuesShapeLen = (*jniEnv)->GetArrayLength(jniEnv,valuesShape);

    // Create the OrtValue
    OrtValue* ortValue;
    checkOrtStatus(jniEnv, api, api->CreateSparseTensorWithValuesAsOrtValue(allocatorInfo, dataBufferArr,
     (int64_t*) shapeArr, shapeLen, (int64_t*) valuesShapeArr, valuesShapeLen, onnxType, &ortValue));
    (*jniEnv)->ReleaseLongArrayElements(jniEnv,valuesShape,valuesShapeArr,JNI_ABORT);

    // Fill it with indices
    switch (sparsityType) {
        case ORT_SPARSE_COO: {
            // The cast is because we compute the offset in bytes in Java.
            checkOrtStatus(jniEnv,api,api->UseCooIndices(ortValue, (int64_t *) indicesBufferArr, indicesBufferSize));
            break;
        }
        case ORT_SPARSE_CSRC:
        case ORT_SPARSE_BLOCK_SPARSE:
        case ORT_SPARSE_UNDEFINED: {
            throwOrtException(jniEnv,convertErrorCode(ORT_NOT_IMPLEMENTED),"These types are unsupported - ORT_SPARSE_CSRC, ORT_SPARSE_BLOCK_SPARSE, ORT_SPARSE_UNDEFINED");
            break;
        }
    }

    // Return the pointer to the OrtValue
    return (jlong) ortValue;
}

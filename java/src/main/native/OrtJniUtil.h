/*
 * Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <stdlib.h>
#include "onnxruntime_config.h"
#include "onnxruntime/core/session/onnxruntime_c_api.h"

#ifndef __ONNXUtil_h
#define __ONNXUtil_h
#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
  /* The number of dimensions in the Tensor */
  size_t dimensions;
  /* The number of elements in the Tensor */
  size_t elementCount;
  /* The type of the Tensor */
  ONNXTensorElementDataType onnxTypeEnum;
} JavaTensorTypeShape;

jint JNI_OnLoad(JavaVM *vm, void *reserved);

OrtLoggingLevel convertLoggingLevel(jint level);

GraphOptimizationLevel convertOptimizationLevel(jint level);

ExecutionMode convertExecutionMode(jint mode);

OrtSparseFormat convertToOrtSparseFormat(jint format);

jint convertFromOrtSparseFormat(OrtSparseFormat format);

jint convertFromONNXDataFormat(ONNXTensorElementDataType type);

ONNXTensorElementDataType convertToONNXDataFormat(jint type);

size_t onnxTypeSize(ONNXTensorElementDataType type);

OrtErrorCode getTensorTypeShape(JNIEnv * jniEnv, JavaTensorTypeShape * output, const OrtApi * api, const OrtValue * value);

jfloat convertHalfToFloat(uint16_t half);

jfloat convertBF16ToFloat(uint16_t half);

jobject convertToValueInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtTypeInfo * info);

jobject convertToTensorInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtTensorTypeAndShapeInfo * info);

jobject convertToMapInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtMapTypeInfo * info);

jobject convertToSequenceInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtSequenceTypeInfo * info);

int64_t copyJavaToPrimitiveArray(JNIEnv* jniEnv, ONNXTensorElementDataType onnxType, jarray inputArray, uint8_t* outputTensor);

int64_t copyJavaToTensor(JNIEnv* jniEnv, ONNXTensorElementDataType onnxType, size_t tensorSize, size_t dimensionsRemaining, jarray inputArray, uint8_t* outputTensor);

int64_t copyPrimitiveArrayToJava(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, const uint8_t* inputTensor, jarray outputArray);

int64_t copyTensorToJava(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, const uint8_t* inputTensor, size_t tensorSize, size_t dimensionsRemaining, jarray outputArray);

jobject createStringFromStringTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor);

OrtErrorCode copyStringTensorToArray(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor, size_t length, jobjectArray outputArray);

jobjectArray createStringArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor);

jlongArray createLongArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor);

jfloatArray createFloatArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor);

jdoubleArray createDoubleArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor);

jobject createJavaTensorFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor);

jobject createJavaSparseTensorFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor);

jobject createJavaSequenceFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* sequence);

jobject createJavaMapFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* map);

jobject createMapInfoFromValue(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator * allocator, const OrtValue * map);

jobject convertOrtValueToONNXValue(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* onnxValue);

jint throwOrtException(JNIEnv *env, int messageId, const char *message);

jint convertErrorCode(OrtErrorCode code);

OrtErrorCode checkOrtStatus(JNIEnv * env, const OrtApi * api, OrtStatus * status);

jsize safecast_size_t_to_jsize(size_t v);

jsize safecast_int64_to_jsize(int64_t v);

#ifdef _WIN32
#include <Intsafe.h>
static inline void* allocarray(size_t nmemb, size_t size) {
  size_t out;
  HRESULT hr = SIZETMult(nmemb, size, &out);
  if (hr != S_OK) return NULL;
  return malloc(out);
}
#else
static inline void* allocarray(size_t nmemb, size_t size) {
#ifdef HAS_REALLOCARRAY
  return reallocarray(NULL, nmemb, size);
#else
  //TODO: find a safer way
  return malloc(nmemb * size);
#endif
}
#endif
#ifdef __cplusplus
}
#endif
#endif

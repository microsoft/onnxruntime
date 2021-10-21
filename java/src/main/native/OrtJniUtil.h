/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"

#ifndef __ONNXUtil_h
#define __ONNXUtil_h
#ifdef __cplusplus
extern "C" {
#endif

jsize safecast_size_t_to_jsize(size_t v);

jsize safecast_int64_to_jsize(int64_t v);

jint JNI_OnLoad(JavaVM *vm, void *reserved);

OrtLoggingLevel convertLoggingLevel(jint level);

GraphOptimizationLevel convertOptimizationLevel(jint level);

ExecutionMode convertExecutionMode(jint mode);

jint convertFromONNXDataFormat(ONNXTensorElementDataType type);

ONNXTensorElementDataType convertToONNXDataFormat(jint type);

size_t onnxTypeSize(ONNXTensorElementDataType type);

jfloat convertHalfToFloat(uint16_t half);

jobject convertToValueInfo(JNIEnv *jniEnv, const OrtApi * api, OrtTypeInfo * info);

jobject convertToTensorInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtTensorTypeAndShapeInfo * info);

jobject convertToMapInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtMapTypeInfo * info);
jobject convertToSequenceInfo(JNIEnv *jniEnv, const OrtApi * api, const OrtSequenceTypeInfo * info);

jobject createEmptyMapInfo(JNIEnv *jniEnv);
jobject createEmptySequenceInfo(JNIEnv *jniEnv);

size_t copyJavaToPrimitiveArray(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, uint8_t* tensor, jarray input);

size_t copyJavaToTensor(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, uint8_t* tensor, size_t tensorSize, size_t dimensionsRemaining, jarray input);

size_t copyPrimitiveArrayToJava(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, uint8_t* tensor, jarray output);

size_t copyTensorToJava(JNIEnv *jniEnv, ONNXTensorElementDataType onnxType, uint8_t* tensor, size_t tensorSize, size_t dimensionsRemaining, jarray output);

jobject createStringFromStringTensor(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor);

void copyStringTensorToArray(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor, size_t length, jobjectArray outputArray);

jobjectArray createStringArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor);

jlongArray createLongArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor);

jfloatArray createFloatArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor);

jdoubleArray createDoubleArrayFromTensor(JNIEnv *jniEnv, const OrtApi * api, OrtValue* tensor);

jobject createJavaTensorFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* tensor);

jobject createJavaSequenceFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* sequence);

jobject createJavaMapFromONNX(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* map);

jobject convertOrtValueToONNXValue(JNIEnv *jniEnv, const OrtApi * api, OrtAllocator* allocator, OrtValue* onnxValue);

jint throwOrtException(JNIEnv *env, int messageId, const char *message);

jint convertErrorCode(OrtErrorCode code);

void checkOrtStatus(JNIEnv * env, const OrtApi * api, OrtStatus * status);

#ifdef __cplusplus
}
#endif
#endif

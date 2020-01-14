/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtSession_SessionOptions.h"

// Providers
#include "onnxruntime/core/providers/cpu/cpu_provider_factory.h"
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#include "onnxruntime/core/providers/dnnl/dnnl_provider_factory.h"
#include "onnxruntime/core/providers/ngraph/ngraph_provider_factory.h"
#include "onnxruntime/core/providers/nnapi/nnapi_provider_factory.h"
#include "onnxruntime/core/providers/nuphar/nuphar_provider_factory.h"
#include "onnxruntime/core/providers/openvino/openvino_provider_factory.h"
#include "onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    setExecutionMode
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_setExecutionMode
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint mode) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    ExecutionMode exMode = convertExecutionMode(mode);
    checkOrtStatus(jniEnv,api,api->SetSessionExecutionMode((OrtSessionOptions*) handle,exMode));
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    setOptimizationLevel
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_setOptimizationLevel
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint optLevel) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkOrtStatus(jniEnv,api,api->SetSessionGraphOptimizationLevel((OrtSessionOptions*) handle, convertOptimizationLevel(optLevel)));
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    setIntraOpNumThreads
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_setIntraOpNumThreads
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint numThreads) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkOrtStatus(jniEnv,api,api->SetIntraOpNumThreads((OrtSessionOptions*) handle, numThreads));
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    setInterOpNumThreads
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_setInterOpNumThreads
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint numThreads) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkOrtStatus(jniEnv,api,api->SetInterOpNumThreads((OrtSessionOptions*) handle, numThreads));
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    setOptimizationModelFilePath
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_setOptimizationModelFilePath
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jstring pathString) {
    (void) jobj; // Required JNI parameter not needed by function which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
#ifdef _WIN32
    const jchar* path = (*jniEnv)->GetStringChars(jniEnv, pathString, NULL);
    size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, pathString);
    wchar_t* newString = (wchar_t*)calloc(stringLength+1,sizeof(jchar));
    wcsncpy_s(newString, stringLength+1, (const wchar_t*) path, stringLength);
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,api->SetOptimizedModelFilePath((OrtSessionOptions*) handle, (const wchar_t*) newString));
    free(newString);
    (*jniEnv)->ReleaseStringChars(jniEnv,pathString,path);
#else
    const char* path = (*jniEnv)->GetStringUTFChars(jniEnv, pathString, NULL);
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,api->SetOptimizedModelFilePath((OrtSessionOptions*) handle, path));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,pathString,path);
#endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    createOptions
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_createOptions
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtSessionOptions* opts;
    checkOrtStatus(jniEnv,api,api->CreateSessionOptions(&opts));
    checkOrtStatus(jniEnv,api,api->SetInterOpNumThreads(opts, 1));
    checkOrtStatus(jniEnv,api,api->SetIntraOpNumThreads(opts, 1));
    return (jlong) opts;
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    closeOptions
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_closeOptions
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseSessionOptions((OrtSessionOptions*) handle);
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addCPU
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addCPU
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint useArena) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_CPU((OrtSessionOptions*)handle,useArena));
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addCUDA
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addCUDA
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint deviceID) {
    (void)jobj;
  #ifdef USE_CUDA
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_CUDA((OrtSessionOptions*) handle, deviceID));
  #else
    (void)apiHandle;(void)handle;(void)deviceID; // Parameters used when CUDA is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with CUDA support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addDnnl
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addDnnl
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint useArena) {
    (void)jobj;
  #ifdef USE_DNNL
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Dnnl((OrtSessionOptions*) handle,useArena));
  #else
    (void)apiHandle;(void)handle;(void)useArena; // Parameters used when DNNL is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with DNNL support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addNGraph
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addNGraph
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jstring backendString) {
    (void)jobj;
  #ifdef USE_NGRAPH
    const char* backendType = (*jniEnv)->GetStringUTFChars(jniEnv, backendString, NULL);
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_NGraph((OrtSessionOptions*) handle, backendType));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,backendString,backendType);
  #else
    (void)apiHandle;(void)handle;(void)backendString; // Parameters used when NGraph is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with NGraph support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addOpenVINO
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addOpenVINO
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jstring deviceIDString) {
    (void)jobj;
  #ifdef USE_OPENVINO
    const char* deviceID = (*jniEnv)->GetStringUTFChars(jniEnv, deviceIDString, NULL);
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_OpenVINO((OrtSessionOptions*) handle, deviceID));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,deviceIDString,deviceID);
  #else
    (void)apiHandle;(void)handle;(void)deviceIDString; // Parameters used when OpenVINO is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with OpenVINO support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addTensorrt
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addTensorrt
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint deviceNum) {
    (void)jobj;
  #ifdef USE_TENSORRT
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Tensorrt((OrtSessionOptions*) handle, deviceNum));
  #else
    (void)apiHandle;(void)handle;(void)deviceNum; // Parameters used when TensorRT is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with TensorRT support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addNnapi
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addNnapi
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void)jobj;
  #ifdef USE_NNAPI
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Nnapi((OrtSessionOptions*) handle));
  #else
    (void)apiHandle;(void)handle; // Parameters used when NNAPI is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with NNAPI support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addNuphar
 * Signature: (JILjava/lang/String {
	})V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addNuphar
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint allowUnalignedBuffers, jstring settingsString) {
    (void)jobj;
  #ifdef USE_NUPHAR
    const char* settings = (*jniEnv)->GetStringUTFChars(jniEnv, settingsString, NULL);
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Nuphar((OrtSessionOptions*) handle, allowUnalignedBuffers, settings));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,settingsString,settings);
  #else
    (void)apiHandle;(void)handle;(void)allowUnalignedBuffers;(void)settingsString; // Parameters used when Nuphar is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with Nuphar support.");
  #endif
}

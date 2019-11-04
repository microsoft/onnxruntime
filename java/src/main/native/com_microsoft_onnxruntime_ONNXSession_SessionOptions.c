/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "ONNXUtil.h"
#include "com_microsoft_onnxruntime_ONNXSession_SessionOptions.h"

// Providers
#include "onnxruntime/core/providers/cpu/cpu_provider_factory.h"
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#include "onnxruntime/core/providers/mkldnn/mkldnn_provider_factory.h"
#include "onnxruntime/core/providers/ngraph/ngraph_provider_factory.h"
#include "onnxruntime/core/providers/nnapi/nnapi_provider_factory.h"
#include "onnxruntime/core/providers/nuphar/nuphar_provider_factory.h"
#include "onnxruntime/core/providers/openvino/openvino_provider_factory.h"
#include "onnxruntime/core/providers/tensorrt/tensorrt_provider_factory.h"

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    setSequentialExecution
 * Signature: (JJZ)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_setSequentialExecution
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jboolean setSequential) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    if (setSequential) {
        checkONNXStatus(jniEnv,api,api->SetSessionExecutionMode((OrtSessionOptions*) handle,ORT_SEQUENTIAL));
    } else {
        checkONNXStatus(jniEnv,api,api->SetSessionExecutionMode((OrtSessionOptions*) handle,ORT_PARALLEL));
    }
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    setOptimisationLevel
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_setOptimisationLevel
  (JNIEnv * jniEnv, jobject obj, jlong apiHandle, jlong handle, jint optLevel) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkONNXStatus(jniEnv,api,api->SetSessionGraphOptimizationLevel((OrtSessionOptions*) handle, convertOptimizationLevel(optLevel)));
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    setIntraThreadPoolSize
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_setIntraThreadPoolSize
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint numThreads) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkONNXStatus(jniEnv,api,api->SetIntraOpNumThreads((OrtSessionOptions*) handle, numThreads));
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    setIntraThreadPoolSize
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_setInterThreadPoolSize
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint numThreads) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkONNXStatus(jniEnv,api,api->SetInterOpNumThreads((OrtSessionOptions*) handle, numThreads));
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    createOptions
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_createOptions
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtSessionOptions* opts;
    checkONNXStatus(jniEnv,api,api->CreateSessionOptions(&opts));
    checkONNXStatus(jniEnv,api,api->SetInterOpNumThreads(opts, 1));
    checkONNXStatus(jniEnv,api,api->SetIntraOpNumThreads(opts, 1));
    return (jlong) opts;
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    closeOptions
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_closeOptions
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseSessionOptions((OrtSessionOptions*) handle);
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    addCPU
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_addCPU
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint useArena) {
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_CPU((OrtSessionOptions*)handle,useArena));
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    addCUDA
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_addCUDA
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint deviceID) {
  #ifdef BUILD_CUDA
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_CUDA((OrtSessionOptions*) handle, deviceID));
  #else
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with CUDA support.");
  #endif
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    addMkldnn
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_addMkldnn
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint useArena) {
  #ifdef BUILD_MKLDNN
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Mkldnn((OrtSessionOptions*) handle,useArena));
  #else
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with MKL-DNN support.");
  #endif
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    addNGraph
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_addNGraph
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jstring backendString) {
  #ifdef BUILD_NGRAPH
    const char* backendType = (*jniEnv)->GetStringUTFChars(jniEnv, backendString, NULL);
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_NGraph((OrtSessionOptions*) handle, backendType));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,backendString,backendType);
  #else
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with NGraph support.");
  #endif
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    addOpenVINO
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_addOpenVINO
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jstring deviceIDString) {
  #ifdef BUILD_OPEN_VINO
    const char* deviceID = (*jniEnv)->GetStringUTFChars(jniEnv, deviceIDString, NULL);
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_OpenVINO((OrtSessionOptions*) handle, deviceID));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,deviceIDString,deviceID);
  #else
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with OpenVINO support.");
  #endif
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    addTensorrt
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_addTensorrt
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint deviceNum) {
  #ifdef BUILD_TENSOR_RT
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Tensorrt((OrtSessionOptions*) handle, deviceNum));
  #else
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with TensorRT support.");
  #endif
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    addNnapi
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_addNnapi
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
  #ifdef BUILD_NNAPI
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Nnapi((OrtSessionOptions*) handle));
  #else
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with NNAPI support.");
  #endif
}

/*
 * Class:     com_microsoft_onnxruntime_ONNXSession_SessionOptions
 * Method:    addNuphar
 * Signature: (JILjava/lang/String {
	})V
 */
JNIEXPORT void JNICALL Java_com_microsoft_onnxruntime_ONNXSession_00024SessionOptions_addNuphar
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint allowUnalignedBuffers, jstring settingsString) {
  #ifdef BUILD_NUPHAR
    const char* settings = (*jniEnv)->GetStringUTFChars(jniEnv, settingsString, NULL);
    checkONNXStatus(jniEnv,OrtSessionOptionsAppendExecutionProvider_Nuphar((OrtSessionOptions*) handle, allowUnalignedBuffers, settings));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,settingsString,settings);
  #else
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with Nuphar support.");
  #endif
}

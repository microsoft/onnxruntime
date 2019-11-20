/*
 * Copyright © 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "ONNXUtil.h"
#include "ai_onnxruntime_ONNXSession_SessionOptions.h"

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
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    setSequentialExecution
 * Signature: (JJZ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_setSequentialExecution
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jboolean setSequential) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    if (setSequential) {
        checkONNXStatus(jniEnv,api,api->SetSessionExecutionMode((OrtSessionOptions*) handle,ORT_SEQUENTIAL));
    } else {
        checkONNXStatus(jniEnv,api,api->SetSessionExecutionMode((OrtSessionOptions*) handle,ORT_PARALLEL));
    }
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    setOptimizationLevel
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_setOptimizationLevel
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint optLevel) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkONNXStatus(jniEnv,api,api->SetSessionGraphOptimizationLevel((OrtSessionOptions*) handle, convertOptimizationLevel(optLevel)));
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    setIntraOpNumThreads
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_setIntraOpNumThreads
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint numThreads) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkONNXStatus(jniEnv,api,api->SetIntraOpNumThreads((OrtSessionOptions*) handle, numThreads));
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    setInterOpNumThreads
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_setInterOpNumThreads
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint numThreads) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    checkONNXStatus(jniEnv,api,api->SetInterOpNumThreads((OrtSessionOptions*) handle, numThreads));
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    createOptions
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_createOptions
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtSessionOptions* opts;
    checkONNXStatus(jniEnv,api,api->CreateSessionOptions(&opts));
    checkONNXStatus(jniEnv,api,api->SetInterOpNumThreads(opts, 1));
    checkONNXStatus(jniEnv,api,api->SetIntraOpNumThreads(opts, 1));
    return (jlong) opts;
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    closeOptions
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_closeOptions
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
    (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    api->ReleaseSessionOptions((OrtSessionOptions*) handle);
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    addCPU
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_addCPU
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint useArena) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_CPU((OrtSessionOptions*)handle,useArena));
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    addCUDA
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_addCUDA
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint deviceID) {
  #ifdef USE_CUDA
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_CUDA((OrtSessionOptions*) handle, deviceID));
  #else
    (void)jobj;(void)apiHandle;(void)handle;(void)deviceID; // Parameters used when CUDA is defined.
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with CUDA support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    addMkldnn
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_addMkldnn
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint useArena) {
  #ifdef USE_MKLDNN
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Mkldnn((OrtSessionOptions*) handle,useArena));
  #else
    (void)jobj;(void)apiHandle;(void)handle;(void)useArena; // Parameters used when MKL-DNN is defined.
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with MKL-DNN support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    addNGraph
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_addNGraph
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jstring backendString) {
  #ifdef USE_NGRAPH
    const char* backendType = (*jniEnv)->GetStringUTFChars(jniEnv, backendString, NULL);
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_NGraph((OrtSessionOptions*) handle, backendType));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,backendString,backendType);
  #else
    (void)jobj;(void)apiHandle;(void)handle;(void)backendString; // Parameters used when NGraph is defined.
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with NGraph support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    addOpenVINO
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_addOpenVINO
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jstring deviceIDString) {
  #ifdef USE_OPENVINO
    const char* deviceID = (*jniEnv)->GetStringUTFChars(jniEnv, deviceIDString, NULL);
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_OpenVINO((OrtSessionOptions*) handle, deviceID));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,deviceIDString,deviceID);
  #else
    (void)jobj;(void)apiHandle;(void)handle;(void)deviceIDString; // Parameters used when OpenVINO is defined.
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with OpenVINO support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    addTensorrt
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_addTensorrt
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint deviceNum) {
  #ifdef USE_TENSORRT
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Tensorrt((OrtSessionOptions*) handle, deviceNum));
  #else
    (void)jobj;(void)apiHandle;(void)handle;(void)deviceNum; // Parameters used when TensorRT is defined.
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with TensorRT support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    addNnapi
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_addNnapi
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
  #ifdef USE_NNAPI
    checkONNXStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Nnapi((OrtSessionOptions*) handle));
  #else
    (void)jobj;(void)apiHandle;(void)handle; // Parameters used when NNAPI is defined.
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with NNAPI support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_ONNXSession_SessionOptions
 * Method:    addNuphar
 * Signature: (JILjava/lang/String {
	})V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_ONNXSession_00024SessionOptions_addNuphar
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint allowUnalignedBuffers, jstring settingsString) {
  #ifdef USE_NUPHAR
    const char* settings = (*jniEnv)->GetStringUTFChars(jniEnv, settingsString, NULL);
    checkONNXStatus(jniEnv,OrtSessionOptionsAppendExecutionProvider_Nuphar((OrtSessionOptions*) handle, allowUnalignedBuffers, settings));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,settingsString,settings);
  #else
    (void)jobj;(void)apiHandle;(void)handle;(void)allowUnalignedBuffers;(void)settingsString; // Parameters used when Nuphar is defined.
    throwONNXException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with Nuphar support.");
  #endif
}

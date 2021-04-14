/*
 * Copyright (c) 2019, 2020 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtSession_SessionOptions.h"
#ifdef WIN32
#include <Windows.h>
#else
#include <dlfcn.h>
#endif

// Providers
#include "onnxruntime/core/providers/cpu/cpu_provider_factory.h"
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#include "onnxruntime/core/providers/dnnl/dnnl_provider_factory.h"
#include "onnxruntime/core/providers/nnapi/nnapi_provider_factory.h"
#include "onnxruntime/core/providers/nuphar/nuphar_provider_factory.h"
#include "onnxruntime/core/providers/migraphx/migraphx_provider_factory.h"
#include "onnxruntime/core/providers/acl/acl_provider_factory.h"
#include "onnxruntime/core/providers/armnn/armnn_provider_factory.h"
#include "onnxruntime/core/providers/coreml/coreml_provider_factory.h"
#include "onnxruntime/core/providers/rocm/rocm_provider_factory.h"
#ifdef USE_DML
#include "onnxruntime/core/providers/dml/dml_provider_factory.h"
#endif

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
    if(newString == NULL) throwOrtException(jniEnv, 1, "Not enough memory");
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
    // Commented out due to constant OpenMP warning as this API is invalid when running with OpenMP.
    // Not sure how to detect that from within the C API though.
    //checkOrtStatus(jniEnv,api,api->SetIntraOpNumThreads(opts, 1));
    return (jlong) opts;
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    closeOptions
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_closeOptions
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
  (void)jniEnv; (void)jobj;  // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  api->ReleaseSessionOptions((OrtSessionOptions*)handle);
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    setLoggerId
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_setLoggerId
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jstring loggerId) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;
  const char* loggerIdStr = (*jniEnv)->GetStringUTFChars(jniEnv, loggerId, NULL);
  checkOrtStatus(jniEnv,api,api->SetSessionLogId(options, loggerIdStr));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv,loggerId,loggerIdStr);
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    enableProfiling
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_enableProfiling
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jstring pathString) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;
#ifdef _WIN32
  const jchar* path = (*jniEnv)->GetStringChars(jniEnv, pathString, NULL);
  size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, pathString);
  wchar_t* newString = (wchar_t*)calloc(stringLength+1,sizeof(jchar));
  if(newString == NULL) throwOrtException(jniEnv, 1, "Not enough memory");
  wcsncpy_s(newString, stringLength+1, (const wchar_t*) path, stringLength);
  checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,api->EnableProfiling(options, (const wchar_t*) newString));
  free(newString);
  (*jniEnv)->ReleaseStringChars(jniEnv,pathString,path);
#else
  const char* path = (*jniEnv)->GetStringUTFChars(jniEnv, pathString, NULL);
  checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,api->EnableProfiling(options, path));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv,pathString,path);
#endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    disableProfiling
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_disableProfiling
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;
  checkOrtStatus(jniEnv,api,api->DisableProfiling(options));
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    setMemoryPatternOptimization
 * Signature: (JJZ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_setMemoryPatternOptimization
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jboolean memPattern) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;
  if (memPattern) {
    checkOrtStatus(jniEnv,api,api->EnableMemPattern(options));
  } else {
    checkOrtStatus(jniEnv,api,api->DisableMemPattern(options));
  }
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    setCPUArenaAllocator
 * Signature: (JJZ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_setCPUArenaAllocator
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jboolean useArena) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;
  if (useArena) {
    checkOrtStatus(jniEnv,api,api->EnableCpuMemArena(options));
  } else {
    checkOrtStatus(jniEnv,api,api->DisableCpuMemArena(options));
  }
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    setSessionLogLevel
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_setSessionLogLevel
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jint logLevel) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;
  checkOrtStatus(jniEnv,api,api->SetSessionLogSeverityLevel(options,logLevel));
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    setSessionLogVerbosityLevel
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_setSessionLogVerbosityLevel
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jint logLevel) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;
  checkOrtStatus(jniEnv,api,api->SetSessionLogVerbosityLevel(options,logLevel));
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    registerCustomOpLibrary
 * Signature: (JJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_registerCustomOpLibrary
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jstring libraryPath) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;

  // Extract the string chars
  const char* cPath = (*jniEnv)->GetStringUTFChars(jniEnv, libraryPath, NULL);

  // Load the library
  void* libraryHandle;
  checkOrtStatus(jniEnv,api,api->RegisterCustomOpsLibrary((OrtSessionOptions*)optionsHandle,cPath,&libraryHandle));

  // Release the string chars
  (*jniEnv)->ReleaseStringUTFChars(jniEnv,libraryPath,cPath);

  return (jlong) libraryHandle;
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    closeCustomLibraries
 * Signature: ([J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_closeCustomLibraries
    (JNIEnv * jniEnv, jobject jobj, jlongArray libraryHandles) {
  (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.

  // Get the number of elements in the array
  jsize numHandles = (*jniEnv)->GetArrayLength(jniEnv, libraryHandles);

  // Get the elements of the libraryHandles array
  jlong* handles = (*jniEnv)->GetLongArrayElements(jniEnv,libraryHandles,NULL);

  // Iterate the handles, calling the appropriate close function
  for (jint i = 0; i < numHandles; i++) {
#ifdef WIN32
    FreeLibrary((void*)handles[i]);
#else
    dlclose((void*)handles[i]);
#endif
  }

  // Release the long array
  (*jniEnv)->ReleaseLongArrayElements(jniEnv,libraryHandles,handles,JNI_ABORT);
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addFreeDimensionOverrideByName
 * Signature: (JJLjava/lang/String;J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addFreeDimensionOverrideByName
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jstring dimensionName, jlong dimensionValue) {
  (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;

  // Extract the string chars
  const char* cName = (*jniEnv)->GetStringUTFChars(jniEnv, dimensionName, NULL);

  checkOrtStatus(jniEnv,api,api->AddFreeDimensionOverrideByName(options,cName,dimensionValue));

  // Release the string chars
  (*jniEnv)->ReleaseStringUTFChars(jniEnv,dimensionName,cName);
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    disablePerSessionThreads
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_disablePerSessionThreads
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle) {
  (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;
  checkOrtStatus(jniEnv,api,api->DisablePerSessionThreads(options));
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addConfigEntry
 * Signature: (JJLjava/lang/String;Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addConfigEntry
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jstring configKey, jstring configValue) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;
  const char* configKeyStr = (*jniEnv)->GetStringUTFChars(jniEnv, configKey, NULL);
  const char* configValueStr = (*jniEnv)->GetStringUTFChars(jniEnv, configValue, NULL);
  checkOrtStatus(jniEnv,api,api->AddSessionConfigEntry(options, configKeyStr, configValueStr));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, configKey, configKeyStr);
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, configValue, configValueStr);
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
 * Method:    addNnapi
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addNnapi
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint nnapiFlags) {
    (void)jobj;
  #ifdef USE_NNAPI
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Nnapi((OrtSessionOptions*) handle, (uint32_t) nnapiFlags));
  #else
    (void)apiHandle;(void)handle;(void)nnapiFlags; // Parameters used when NNAPI is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with NNAPI support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addNuphar
 * Signature: (JILjava/lang/String)V
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

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addMIGraphX
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addMIGraphX
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint deviceNum) {
    (void)jobj;
  #ifdef USE_MIGRAPHX
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_MIGraphX((OrtSessionOptions*) handle, deviceNum));
  #else
    (void)apiHandle;(void)handle;(void)deviceNum; // Parameters used when MIGraphX is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with MIGraphX support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addDirectML
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addDirectML
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint deviceID) {
  (void)jobj;
  #ifdef USE_DIRECTML
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_DML((OrtSessionOptions*) handle, deviceID));
  #else
    (void)apiHandle;(void)handle;(void)deviceID; // Parameters used when DirectML is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with DirectML support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addACL
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addACL
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint useArena) {
  (void)jobj;
  #ifdef USE_ACL
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_ACL((OrtSessionOptions*) handle,useArena));
  #else
    (void)apiHandle;(void)handle;(void)useArena; // Parameters used when ACL is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with ACL support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addArmNN
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addArmNN
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint useArena) {
  (void)jobj;
  #ifdef USE_ARMNN
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_ArmNN((OrtSessionOptions*) handle,useArena));
  #else
    (void)apiHandle;(void)handle;(void)useArena; // Parameters used when ARMNN is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with ArmNN support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addCoreML
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addCoreML
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint coreMLFlags) {
    (void)jobj;
  #ifdef USE_CORE_ML
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_CoreML((OrtSessionOptions*) handle, (uint32_t) coreMLFlags));
  #else
    (void)apiHandle;(void)handle;(void)coreMLFlags; // Parameters used when CoreML is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with CoreML support.");
  #endif
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addROCM
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addROCM
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint deviceID, jlong memLimit) {
    (void)jobj;
  #ifdef USE_ROCM
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_ROCM((OrtSessionOptions*) handle, deviceID, (size_t) memLimit));
  #else
    (void)apiHandle;(void)handle;(void)deviceID;(void)memLimit; // Parameters used when ROCM is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with ROCM support.");
  #endif
}


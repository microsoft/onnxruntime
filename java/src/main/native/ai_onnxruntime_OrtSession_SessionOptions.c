/*
 * Copyright (c) 2019, 2023 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include <stdlib.h>
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
#include "onnxruntime/core/providers/nnapi/nnapi_provider_factory.h"
#include "onnxruntime/core/providers/tvm/tvm_provider_factory.h"
#include "onnxruntime/core/providers/openvino/openvino_provider_factory.h"
#include "onnxruntime/core/providers/acl/acl_provider_factory.h"
#include "onnxruntime/core/providers/armnn/armnn_provider_factory.h"
#include "onnxruntime/core/providers/coreml/coreml_provider_factory.h"
#ifdef USE_DML
#include "onnxruntime/core/providers/dml/dml_provider_factory.h"
#endif

#ifdef USE_DNNL
#include "core/providers/dnnl/dnnl_provider_options.h"
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
    if (newString == NULL) {
        throwOrtException(jniEnv, 1, "Not enough memory");
        return;
    }
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
#ifdef ENABLE_EXTENSION_CUSTOM_OPS
    // including all custom ops from onnxruntime-extensions
    checkOrtStatus(jniEnv,api,api->EnableOrtCustomOps(opts));
#endif
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
  if (newString == NULL) {
      throwOrtException(jniEnv, 1, "Not enough memory");
      return;
  }
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
 * Method:    registerCustomOpsUsingFunction
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_registerCustomOpsUsingFunction
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jstring functionName) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;

  // Extract the string chars
  const char* cFuncName = (*jniEnv)->GetStringUTFChars(jniEnv, functionName, NULL);

  // Register the custom ops by calling the function
  checkOrtStatus(jniEnv,api,api->RegisterCustomOpsUsingFunction((OrtSessionOptions*)optionsHandle,cFuncName));

  // Release the string chars
  (*jniEnv)->ReleaseStringUTFChars(jniEnv,functionName,cFuncName);
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
 * Method:    addExternalInitializers
 * Signature: (JJ[Ljava/lang/String;[J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addExternalInitializers
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jobjectArray namesArray, jlongArray handlesArray) {
  (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;

  size_t arrLength = (*jniEnv)->GetArrayLength(jniEnv, handlesArray);

  const char** names = allocarray(arrLength, sizeof(char*));
  if (names == NULL) {
    // Nothing to cleanup, return and throw exception
    throwOrtException(jniEnv, 1, "Not enough memory");
    return;
  }
  jobject* javaNameStrings = allocarray(arrLength, sizeof(jobject));
  if (javaNameStrings == NULL) {
    goto cleanup_names;
  }
  const OrtValue** initializers = allocarray(arrLength, sizeof(OrtValue*));
  if (initializers == NULL) {
    goto cleanup_java_input_strings;
  }

  // Extract a C array of longs which are pointers to the input tensors.
  // The Java-side objects store native pointers as 64-bit longs, and on 32-bit systems
  // we cannot cast the long array to a pointer array as they are different sizes,
  // so we copy the longs applying the appropriate cast.
  jlong* initializersArr = (*jniEnv)->GetLongArrayElements(jniEnv, handlesArray, NULL);

  for (size_t i = 0; i < arrLength; i++) {
    // Extract the string chars and cast the tensor
    javaNameStrings[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, namesArray, (jint) i);
    names[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaNameStrings[i], NULL);
    initializers[i] = (const OrtValue*) initializersArr[i];
  }

  checkOrtStatus(jniEnv,api,api->AddExternalInitializers(options,names,initializers,arrLength));

  // Release the java array copy of pointers to the tensors.
  (*jniEnv)->ReleaseLongArrayElements(jniEnv, handlesArray, initializersArr, JNI_ABORT);
  free(initializers);
cleanup_java_input_strings:
  // Release the Java strings
  for (size_t i = 0; i < arrLength; i++) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaNameStrings[i], names[i]);
  }
  free(javaNameStrings);
cleanup_names:
  free(names);
}

/*
 * Class:     ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addInitializer
 * Signature: (JJLjava/lang/String;J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addInitializer
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jstring name, jlong tensorHandle) {
  (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*) optionsHandle;

  // Extract the string chars
  const char* cName = (*jniEnv)->GetStringUTFChars(jniEnv, name, NULL);

  // Cast the onnx value
  const OrtValue* tensor = (const OrtValue*) tensorHandle;

  checkOrtStatus(jniEnv,api,api->AddInitializer(options,cName,tensor));

  // Release the string chars
  (*jniEnv)->ReleaseStringUTFChars(jniEnv,name,cName);
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
 * Method:    addCUDAV2
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addCUDAV2
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong optsHandle) {
    (void)jobj;
  #ifdef USE_CUDA
    const OrtApi* api = (OrtApi*) apiHandle;
    checkOrtStatus(jniEnv,api,api->SessionOptionsAppendExecutionProvider_CUDA_V2((OrtSessionOptions*) handle, (const OrtCUDAProviderOptionsV2*) optsHandle));
  #else
    (void)apiHandle;(void)handle;(void)optsHandle; // Parameters used when CUDA is defined.
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
    OrtDnnlProviderOptions dnnl_options;
    dnnl_options.use_arena = useArena;  // Follow the user command
    const OrtApi* api = (OrtApi*)apiHandle;
    checkOrtStatus(jniEnv, api, api->SessionOptionsAppendExecutionProvider_Dnnl((OrtSessionOptions*)handle, &dnnl_options));
#else
    (void)apiHandle; (void)handle; (void)useArena; // Parameters used when DNNL is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with DNNL support.");
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
 * Method:    addTensorrtV2
 * Signature: (JJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addTensorrtV2
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jlong optsHandle) {
    (void)jobj;
  #ifdef USE_TENSORRT
    const OrtApi* api = (OrtApi*) apiHandle;
    checkOrtStatus(jniEnv,api,api->SessionOptionsAppendExecutionProvider_TensorRT_V2((OrtSessionOptions*) handle, (const OrtTensorRTProviderOptionsV2*) optsHandle));
  #else
    (void)apiHandle;(void)handle;(void)optsHandle; // Parameters used when TensorRT is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with TensorRT support.");
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
 * Class::    ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addTvm
 * Signature: (JILjava/lang/String)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addTvm
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jstring settingsString) {
    (void)jobj;
  #ifdef USE_TVM
    const char* settings = (*jniEnv)->GetStringUTFChars(jniEnv, settingsString, NULL);
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_Tvm((OrtSessionOptions*) handle, settings));
    (*jniEnv)->ReleaseStringUTFChars(jniEnv,settingsString,settings);
  #else
    (void)apiHandle;(void)handle;(void)settingsString; // Parameters used when TVM is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with TVM support.");
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
  #ifdef USE_COREML
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
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle, jint deviceID) {
    (void)jobj;
  #ifdef USE_ROCM
    checkOrtStatus(jniEnv,(const OrtApi*)apiHandle,OrtSessionOptionsAppendExecutionProvider_ROCM((OrtSessionOptions*) handle, deviceID));
  #else
    (void)apiHandle;(void)handle;(void)deviceID; // Parameters used when ROCM is defined.
    throwOrtException(jniEnv,convertErrorCode(ORT_INVALID_ARGUMENT),"This binary was not compiled with ROCM support.");
  #endif
}

/*
 * Class::    ai_onnxruntime_OrtSession_SessionOptions
 * Method:    addExecutionProvider
 * Signature: (JILjava/lang/String)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024SessionOptions_addExecutionProvider(
    JNIEnv* jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle,
    jstring jepName, jobjectArray configKeyArr, jobjectArray configValueArr) {
  (void)jobj;

  const char* epName = (*jniEnv)->GetStringUTFChars(jniEnv, jepName, NULL);
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtSessionOptions* options = (OrtSessionOptions*)optionsHandle;
  int keyCount = (*jniEnv)->GetArrayLength(jniEnv, configKeyArr);

  const char** keyArray = (const char**)allocarray(keyCount, sizeof(const char*));
  const char** valueArray = (const char**)allocarray(keyCount, sizeof(const char*));
  jstring* jkeyArray = (jstring*)allocarray(keyCount, sizeof(jstring));
  jstring* jvalueArray = (jstring*)allocarray(keyCount, sizeof(jstring));

  for (int i = 0; i < keyCount; i++) {
    jkeyArray[i] = (jstring)((*jniEnv)->GetObjectArrayElement(jniEnv, configKeyArr, i));
    jvalueArray[i] = (jstring)((*jniEnv)->GetObjectArrayElement(jniEnv, configValueArr, i));
    keyArray[i] = (*jniEnv)->GetStringUTFChars(jniEnv, jkeyArray[i], NULL);
    valueArray[i] = (*jniEnv)->GetStringUTFChars(jniEnv, jvalueArray[i], NULL);
  }

  checkOrtStatus(jniEnv, api, api->SessionOptionsAppendExecutionProvider(options, epName, keyArray, valueArray, keyCount));

  for (int i = 0; i < keyCount; i++) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, jkeyArray[i], keyArray[i]);
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, jvalueArray[i], valueArray[i]);
  }
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, jepName, epName);
  free((void*)keyArray);
  free((void*)valueArray);
  free((void*)jkeyArray);
  free((void*)jvalueArray);
}

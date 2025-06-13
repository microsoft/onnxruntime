/*
 * Copyright (c) 2025 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtModelCompilationOptions.h"

/*
 * Class:     ai_onnxruntime_OrtModelCompilationOptions
 * Method:    createFromSessionOptions
 * Signature: (JJJJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtModelCompilationOptions_createFromSessionOptions
    (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong compileApiHandle, jlong envHandle, jlong sessionOptionsHandle) {
  (void)jclazz;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtCompileApi* compileApi = (const OrtCompileApi*)compileApiHandle;
  const OrtEnv* env = (const OrtEnv*)envHandle;
  const OrtSessionOptions* sessionOptions = (const OrtSessionOptions*) sessionOptionsHandle;
  OrtModelCompilationOptions* output = NULL;
  checkOrtStatus(jniEnv, api, compileApi->CreateModelCompilationOptionsFromSessionOptions(env, sessionOptions, &output));
  return (jlong) output;
}

/*
 * Class:     ai_onnxruntime_OrtModelCompilationOptions
 * Method:    close
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtModelCompilationOptions_close
    (JNIEnv * jniEnv, jclass jclazz, jlong compileApiHandle, jlong nativeHandle) {
  (void)jniEnv; (void)jclazz;  // Required JNI parameters not needed by functions which don't need to access their host object or the JVM.
  const OrtCompileApi* compileApi = (const OrtCompileApi*)compileApiHandle;
  compileApi->ReleaseModelCompilationOptions((OrtModelCompilationOptions *)nativeHandle);
}

/*
 * Class:     ai_onnxruntime_OrtModelCompilationOptions
 * Method:    setInputModelPath
 * Signature: (JJJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtModelCompilationOptions_setInputModelPath
    (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong compileApiHandle, jlong nativeHandle, jstring modelPath) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  const OrtCompileApi* compileApi = (const OrtCompileApi*) compileApiHandle;
  OrtModelCompilationOptions* compOpts = (OrtModelCompilationOptions *) nativeHandle;
#ifdef _WIN32
  const jchar* cPath = (*jniEnv)->GetStringChars(jniEnv, modelPath, NULL);
  size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, modelPath);
  wchar_t* newString = (wchar_t*)calloc(stringLength + 1, sizeof(wchar_t));
  if (newString == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, modelPath, cPath);
    throwOrtException(jniEnv, 1, "Not enough memory");
    return 0;
  }
  wcsncpy_s(newString, stringLength + 1, (const wchar_t*)cPath, stringLength);
  checkOrtStatus(jniEnv, api, compileApi->ModelCompilationOptions_SetInputModelPath(compOpts, newString));
  free(newString);
  (*jniEnv)->ReleaseStringChars(jniEnv, modelPath, cPath);
#else
  const char* cPath = (*jniEnv)->GetStringUTFChars(jniEnv, modelPath, NULL);
  checkOrtStatus(jniEnv, api, compileApi->ModelCompilationOptions_SetInputModelPath(compOpts, cPath));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, modelPath, cPath);
#endif
}

/*
 * Class:     ai_onnxruntime_OrtModelCompilationOptions
 * Method:    setInputModelFromBuffer
 * Signature: (JJJLjava/nio/ByteBuffer;JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtModelCompilationOptions_setInputModelFromBuffer
    (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong compileApiHandle, jlong nativeHandle, jobject buffer, jlong bufferPos, jlong bufferRemaining) {
  (void)jclazz;  // Required JNI parameter not needed by functions which don't need to access their host object.
  // Cast to pointers
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtCompileApi* compileApi = (const OrtCompileApi*)compileApiHandle;
  OrtModelCompilationOptions* compOpts = (OrtModelCompilationOptions *) nativeHandle;

  // Extract the buffer
  char* bufferArr = (char*)(*jniEnv)->GetDirectBufferAddress(jniEnv, buffer);
  // Increment by bufferPos bytes
  bufferArr = bufferArr + bufferPos;
  checkOrtStatus(jniEnv, api, compileApi->ModelCompilationOptions_SetInputModelFromBuffer(compOpts, bufferArr, bufferRemaining));
}

/*
 * Class:     ai_onnxruntime_OrtModelCompilationOptions
 * Method:    setOutputModelPath
 * Signature: (JJJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtModelCompilationOptions_setOutputModelPath
    (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong compileApiHandle, jlong nativeHandle, jstring outputPath) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  const OrtCompileApi* compileApi = (const OrtCompileApi*) compileApiHandle;
  OrtModelCompilationOptions* compOpts = (OrtModelCompilationOptions *) nativeHandle;
#ifdef _WIN32
  const jchar* cPath = (*jniEnv)->GetStringChars(jniEnv, outputPath, NULL);
  size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, outputPath);
  wchar_t* newString = (wchar_t*)calloc(stringLength + 1, sizeof(wchar_t));
  if (newString == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, outputPath, cPath);
    throwOrtException(jniEnv, 1, "Not enough memory");
    return 0;
  }
  wcsncpy_s(newString, stringLength + 1, (const wchar_t*)cPath, stringLength);
  checkOrtStatus(jniEnv, api, compileApi->ModelCompilationOptions_SetOutputModelPath(compOpts, newString));
  free(newString);
  (*jniEnv)->ReleaseStringChars(jniEnv, outputPath, cPath);
#else
  const char* cPath = (*jniEnv)->GetStringUTFChars(jniEnv, outputPath, NULL);
  checkOrtStatus(jniEnv, api, compileApi->ModelCompilationOptions_SetOutputModelPath(compOpts, cPath));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, outputPath, cPath);
#endif
}

/*
 * Class:     ai_onnxruntime_OrtModelCompilationOptions
 * Method:    setOutputExternalInitializersPath
 * Signature: (JJJLjava/lang/String;J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtModelCompilationOptions_setOutputExternalInitializersPath
    (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong compileApiHandle, jlong nativeHandle, jstring initializersPath, jlong threshold) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  const OrtCompileApi* compileApi = (const OrtCompileApi*) compileApiHandle;
  OrtModelCompilationOptions* compOpts = (OrtModelCompilationOptions *) nativeHandle;
#ifdef _WIN32
  const jchar* cPath = (*jniEnv)->GetStringChars(jniEnv, initializersPath, NULL);
  size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, initializersPath);
  wchar_t* newString = (wchar_t*)calloc(stringLength + 1, sizeof(wchar_t));
  if (newString == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, initializersPath, cPath);
    throwOrtException(jniEnv, 1, "Not enough memory");
    return 0;
  }
  wcsncpy_s(newString, stringLength + 1, (const wchar_t*)cPath, stringLength);
  checkOrtStatus(jniEnv, api, compileApi->ModelCompilationOptions_SetOutputModelExternalInitializersFile(compOpts, newString, threshold));
  free(newString);
  (*jniEnv)->ReleaseStringChars(jniEnv, initializersPath, cPath);
#else
  const char* cPath = (*jniEnv)->GetStringUTFChars(jniEnv, initializersPath, NULL);
  checkOrtStatus(jniEnv, api, compileApi->ModelCompilationOptions_SetOutputModelExternalInitializersFile(compOpts, cPath, threshold));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, initializersPath, cPath);
#endif
}

/*
 * Class:     ai_onnxruntime_OrtModelCompilationOptions
 * Method:    setEpContextEmbedMode
 * Signature: (JJJZ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtModelCompilationOptions_setEpContextEmbedMode
    (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong compileApiHandle, jlong nativeHandle, jboolean embedMode) {
  (void)jclazz;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtCompileApi* compileApi = (const OrtCompileApi*)compileApiHandle;
  OrtModelCompilationOptions* compOpts = (OrtModelCompilationOptions *) nativeHandle;
  checkOrtStatus(jniEnv, api, compileApi->ModelCompilationOptions_SetEpContextEmbedMode(compOpts, (bool) embedMode));
}

/*
 * Class:     ai_onnxruntime_OrtModelCompilationOptions
 * Method:    setCompilationFlags
 * Signature: (JJJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtModelCompilationOptions_setCompilationFlags
    (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong compileApiHandle, jlong nativeHandle, jint flags) {
  (void)jclazz;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtCompileApi* compileApi = (const OrtCompileApi*)compileApiHandle;
  OrtModelCompilationOptions* compOpts = (OrtModelCompilationOptions *) nativeHandle;
  checkOrtStatus(jniEnv, api, compileApi->ModelCompilationOptions_SetFlags(compOpts, flags));
}

/*
 * Class:     ai_onnxruntime_OrtModelCompilationOptions
 * Method:    compileModel
 * Signature: (JJJJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtModelCompilationOptions_compileModel
    (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong compileApiHandle, jlong envHandle, jlong nativeHandle) {
  (void)jclazz;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtCompileApi* compileApi = (const OrtCompileApi*)compileApiHandle;
  const OrtEnv* env = (const OrtEnv*)envHandle;
  OrtModelCompilationOptions* compOpts = (OrtModelCompilationOptions *) nativeHandle;
  checkOrtStatus(jniEnv, api, compileApi->CompileModel(env, compOpts));
}

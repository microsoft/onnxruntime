/*
 * Copyright (c) 2020 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtSession_RunOptions.h"

/*
 * Class:     ai_onnxruntime_OrtSession_RunOptions
 * Method:    createRunOptions
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtSession_00024RunOptions_createRunOptions
  (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle) {
    (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtRunOptions* opts;
    checkOrtStatus(jniEnv,api,api->CreateRunOptions(&opts));
    return (jlong) opts;
}

/*
 * Class:     ai_onnxruntime_OrtSession_RunOptions
 * Method:    setLogLevel
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024RunOptions_setLogLevel
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong nativeHandle, jint logLevel) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  checkOrtStatus(jniEnv,api,api->RunOptionsSetRunLogSeverityLevel((OrtRunOptions*) nativeHandle,logLevel));
}

/*
 * Class:     ai_onnxruntime_OrtSession_RunOptions
 * Method:    getLogLevel
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_ai_onnxruntime_OrtSession_00024RunOptions_getLogLevel
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong nativeHandle) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  int logLevel;
  checkOrtStatus(jniEnv,api,api->RunOptionsGetRunLogSeverityLevel((OrtRunOptions*) nativeHandle,&logLevel));
  return (jint)logLevel;
}

/*
 * Class:     ai_onnxruntime_OrtSession_RunOptions
 * Method:    setLogVerbosityLevel
 * Signature: (JJI)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024RunOptions_setLogVerbosityLevel
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong nativeHandle, jint logLevel) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  checkOrtStatus(jniEnv,api,api->RunOptionsSetRunLogVerbosityLevel((OrtRunOptions*) nativeHandle,logLevel));
}

/*
 * Class:     ai_onnxruntime_OrtSession_RunOptions
 * Method:    getLogVerbosityLevel
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_ai_onnxruntime_OrtSession_00024RunOptions_getLogVerbosityLevel
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong nativeHandle) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  int logLevel;
  checkOrtStatus(jniEnv,api,api->RunOptionsGetRunLogVerbosityLevel((OrtRunOptions*) nativeHandle,&logLevel));
  return (jint)logLevel;
}

/*
 * Class:     ai_onnxruntime_OrtSession_RunOptions
 * Method:    setRunTag
 * Signature: (JJLjava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024RunOptions_setRunTag
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong nativeHandle, jstring runTag) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  const char* runTagStr = (*jniEnv)->GetStringUTFChars(jniEnv, runTag, NULL);
  checkOrtStatus(jniEnv,api,api->RunOptionsSetRunTag((OrtRunOptions*) nativeHandle, runTagStr));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv,runTag,runTagStr);
}

/*
 * Class:     ai_onnxruntime_OrtSession_RunOptions
 * Method:    getRunTag
 * Signature: (JJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_OrtSession_00024RunOptions_getRunTag
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong nativeHandle) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  const char* runTagStr;
  // This is a reference to the C str, and should not be freed.
  checkOrtStatus(jniEnv,api,api->RunOptionsGetRunTag((OrtRunOptions*)nativeHandle,&runTagStr));
  jstring runTag = (*jniEnv)->NewStringUTF(jniEnv,runTagStr);
  return runTag;
}

/*
 * Class:     ai_onnxruntime_OrtSession_RunOptions
 * Method:    setTerminate
 * Signature: (JJZ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024RunOptions_setTerminate
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong nativeHandle, jboolean terminate) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtRunOptions* runOptions = (OrtRunOptions*) nativeHandle;
  if (terminate) {
    checkOrtStatus(jniEnv,api,api->RunOptionsSetTerminate(runOptions));
  } else {
    checkOrtStatus(jniEnv,api,api->RunOptionsUnsetTerminate(runOptions));
  }
}

/*
 * Class:     ai_onnxruntime_OrtSession_RunOptions
 * Method:    close
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtSession_00024RunOptions_close
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
  (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  api->ReleaseRunOptions((OrtRunOptions*) handle);
}

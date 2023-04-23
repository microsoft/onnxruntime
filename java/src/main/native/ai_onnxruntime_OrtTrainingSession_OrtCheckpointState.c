/*
 * Copyright (c) 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "onnxruntime_training_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtTrainingSession_OrtCheckpointState.h"

/*
 * Class:     ai_onnxruntime_OrtTrainingSession_OrtCheckpointState
 * Method:    loadCheckpoint
 * Signature: (JJLjava/lang/String;)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtTrainingSession_00024OrtCheckpointState_loadCheckpoint
  (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong trainingApiHandle, jstring directory) {
  (void) jclazz; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*) trainingApiHandle;

  OrtCheckpointState* checkpoint = NULL;

#ifdef _WIN32
  const jchar* cPath = (*jniEnv)->GetStringChars(jniEnv, directory, NULL);
  size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, directory);
  wchar_t* newString = (wchar_t*)calloc(stringLength + 1, sizeof(wchar_t));
  if (newString == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, directory, cPath);
    throwOrtException(jniEnv, 1, "Not enough memory");
    return 0;
  }
  wcsncpy_s(newString, stringLength + 1, (const wchar_t*)cPath, stringLength);
  checkOrtStatus(jniEnv, api,
                 trainApi->LoadCheckpoint(newString, &checkpoint));
  free(newString);
  (*jniEnv)->ReleaseStringChars(jniEnv, directory, cPath);
#else
  const char* cPath = (*jniEnv)->GetStringUTFChars(jniEnv, directory, NULL);
  checkOrtStatus(jniEnv, api, trainApi->LoadCheckpoint(cPath, &checkpoint));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, directory, cPath);
#endif

  return (jlong) checkpoint;
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession
 * Method:    saveCheckpoint
 * Signature: (JJJLjava/lang/String;Z)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_00024OrtCheckpointState_saveCheckpoint
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong trainingApiHandle, jlong nativeHandle, jstring outputPath, jboolean saveOptimizer) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  const OrtTrainingApi* trainApi = (const OrtTrainingApi*) trainingApiHandle;

  OrtCheckpointState* checkpointState = (OrtCheckpointState*) nativeHandle;

#ifdef _WIN32
  // The output of GetStringChars is not null-terminated, so we copy it and add a terminator
  const jchar* cPath = (*jniEnv)->GetStringChars(jniEnv, outputPath, NULL);
  size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, outputPath);
  wchar_t* newString = (wchar_t*)calloc(stringLength + 1, sizeof(wchar_t));
  if (newString == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, outputPath, cPath);
    throwOrtException(jniEnv, 1, "Not enough memory");
  } else {
    wcsncpy_s(newString, stringLength + 1, (const wchar_t*)cPath, stringLength);
    checkOrtStatus(jniEnv, api,
                   trainApi->SaveCheckpoint(checkpointState, newString, saveOptimizer));
    free(newString);
    (*jniEnv)->ReleaseStringChars(jniEnv, outputPath, cPath);
  }
#else
  // GetStringUTFChars is null terminated, so can be used directly
  const char* cPath = (*jniEnv)->GetStringUTFChars(jniEnv, outputPath, NULL);
  checkOrtStatus(jniEnv, api, trainApi->SaveCheckpoint(checkpointState, cPath, saveOptimizer));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, outputPath, cPath);
#endif
}

/*
 * Class:     ai_onnxruntime_OrtTrainingSession_OrtCheckpointState
 * Method:    close
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtTrainingSession_00024OrtCheckpointState_close
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
  (void) jniEnv; (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtTrainingApi* api = (const OrtTrainingApi*) apiHandle;
  api->ReleaseCheckpointState((OrtCheckpointState*) handle);
}

/*
 * Copyright (c) 2022, 2024 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_providers_OrtCUDAProviderOptions.h"

/*
 * Class:     ai_onnxruntime_providers_OrtCUDAProviderOptions
 * Method:    create
 * Signature: (J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_providers_OrtCUDAProviderOptions_create
  (JNIEnv * jniEnv, jobject jobj, jlong apiHandle) {
    (void) jobj; // Required JNI parameter not needed by functions which don't need to access their host object.
    const OrtApi* api = (const OrtApi*) apiHandle;
    OrtCUDAProviderOptionsV2* opts;
    checkOrtStatus(jniEnv,api,api->CreateCUDAProviderOptions(&opts));
    return (jlong) opts;
}

/*
 * Class:     ai_onnxruntime_providers_OrtCUDAProviderOptions
 * Method:    applyToNative
 * Signature: (JJ[Ljava/lang/String;[Ljava/lang/String;)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_providers_OrtCUDAProviderOptions_applyToNative
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong optionsHandle, jobjectArray jKeyArr, jobjectArray jValueArr) {
  (void) jobj; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtCUDAProviderOptionsV2* opts = (OrtCUDAProviderOptionsV2*) optionsHandle;

  size_t keyLength = (*jniEnv)->GetArrayLength(jniEnv, jKeyArr);
  const char** keys = allocarray(keyLength, sizeof(char*));
  const char** values = allocarray(keyLength, sizeof(char*));
  if ((keys == NULL) || (values == NULL)) {
    if (keys != NULL) {
      free(keys);
    }
    if (values != NULL) {
      free(values);
    }
    throwOrtException(jniEnv, 1, "Not enough memory");
  } else {
    // Copy out strings into UTF-8.
    for (size_t i = 0; i < keyLength; i++) {
      jobject key = (*jniEnv)->GetObjectArrayElement(jniEnv, jKeyArr, i);
      keys[i] = (*jniEnv)->GetStringUTFChars(jniEnv, key, NULL);
      jobject value = (*jniEnv)->GetObjectArrayElement(jniEnv, jValueArr, i);
      values[i] = (*jniEnv)->GetStringUTFChars(jniEnv, value, NULL);
    }
    // Write to the provider options.
    checkOrtStatus(jniEnv,api,api->UpdateCUDAProviderOptions(opts, keys, values, keyLength));
    // Release allocated strings.
    for (size_t i = 0; i < keyLength; i++) {
      jobject key = (*jniEnv)->GetObjectArrayElement(jniEnv, jKeyArr, i);
      (*jniEnv)->ReleaseStringUTFChars(jniEnv,key,keys[i]);
      jobject value = (*jniEnv)->GetObjectArrayElement(jniEnv, jKeyArr, i);
      (*jniEnv)->ReleaseStringUTFChars(jniEnv,value,values[i]);
    }
  }
}

/*
 * Class:     ai_onnxruntime_providers_OrtCUDAProviderOptions
 * Method:    close
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_providers_OrtCUDAProviderOptions_close
    (JNIEnv * jniEnv, jobject jobj, jlong apiHandle, jlong handle) {
  (void)jniEnv; (void)jobj;  // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  api->ReleaseCUDAProviderOptions((OrtCUDAProviderOptionsV2*)handle);
}

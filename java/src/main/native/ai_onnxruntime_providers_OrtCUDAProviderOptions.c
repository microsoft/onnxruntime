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

  jsize keyLength = (*jniEnv)->GetArrayLength(jniEnv, jKeyArr);
  const char** keys = (const char**) allocarray(keyLength, sizeof(const char*));
  const char** values = (const char**) allocarray(keyLength, sizeof(const char*));
  jobject* javaKeys = (jobject*) allocarray(keyLength, sizeof(jobject));
  jobject* javaValues = (jobject*) allocarray(keyLength, sizeof(jobject));
  if ((keys == NULL) || (values == NULL) || (javaKeys == NULL) || (javaValues == NULL)) {
    if (keys != NULL) {
    free((void*)keys);
}
if (values != NULL) {
    free((void*)values);
}
if (javaKeys != NULL) {
    free((void*)javaKeys);
}
if (javaValues != NULL) {
    free((void*)javaValues);
}
throwOrtException(jniEnv, 1, "Not enough memory");
  } else {
    // Copy out strings into UTF-8.
for (jsize i = 0; i < keyLength; i++) {
    javaKeys[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, jKeyArr, i);
    keys[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaKeys[i], NULL);

    javaValues[i] = (*jniEnv)->GetObjectArrayElement(jniEnv, jValueArr, i);
    values[i] = (*jniEnv)->GetStringUTFChars(jniEnv, javaValues[i], NULL);
}
    // Write to the provider options.
    checkOrtStatus(jniEnv,api,api->UpdateCUDAProviderOptions(opts, keys, values, keyLength));
    // Release allocated strings.
for (jsize i = 0; i < keyLength; i++) {
    (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaKeys[i], keys[i]);
    (*jniEnv)->DeleteLocalRef(jniEnv, javaKeys[i]);

    (*jniEnv)->ReleaseStringUTFChars(jniEnv, javaValues[i], values[i]);
    (*jniEnv)->DeleteLocalRef(jniEnv, javaValues[i]);
}
free((void*)keys);
free((void*)values);
free((void*)javaKeys);
free((void*)javaValues);
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

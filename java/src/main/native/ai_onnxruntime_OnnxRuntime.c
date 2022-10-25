/*
 * Copyright (c) 2019, 2022 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "ai_onnxruntime_OnnxRuntime.h"
#include "OrtJniUtil.h"

/*
 * Class:     ai_onnxruntime_OnnxRuntime
 * Method:    initialiseAPIBase
 * Signature: (I)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxRuntime_initialiseAPIBase(JNIEnv* jniEnv, jclass clazz,
                                                                          jint apiVersion) {
  (void)jniEnv; (void)clazz;  // required JNI parameters not needed by functions which don't call back into Java.
  const OrtApi* ortPtr = OrtGetApiBase()->GetApi((uint32_t)apiVersion);
  return (jlong)ortPtr;
}

/*
 * Class:     ai_onnxruntime_OnnxRuntime
 * Method:    getAvailableProviders
 * Signature: (J)[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OnnxRuntime_getAvailableProviders(JNIEnv* jniEnv, jclass clazz,
                                                                                     jlong apiHandle) {
  (void)jniEnv; (void)clazz;  // required JNI parameters not needed by functions which don't call back into Java.
  const OrtApi* api = (const OrtApi*)apiHandle;

  char** providers = NULL;
  int numProviders = 0;

  // Extract the provider array
  jobjectArray providerArray = NULL;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetAvailableProviders(&providers, &numProviders));
  if (code == ORT_OK) {
    // Convert to Java String Array
    char* stringClassName = "java/lang/String";
    jclass stringClazz = (*jniEnv)->FindClass(jniEnv, stringClassName);
    providerArray = (*jniEnv)->NewObjectArray(jniEnv, numProviders, stringClazz, NULL);

    for (int i = 0; i < numProviders; i++) {
      // Read out the provider name and convert it to a java.lang.String
      jstring provider = (*jniEnv)->NewStringUTF(jniEnv, providers[i]);
      (*jniEnv)->SetObjectArrayElement(jniEnv, providerArray, i, provider);
    }

    // Release providers
    // if this fails we return immediately anyway
    checkOrtStatus(jniEnv, api, api->ReleaseAvailableProviders(providers, numProviders));
    providers = NULL;
  }
  return providerArray;
}

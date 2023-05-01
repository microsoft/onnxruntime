/*
 * Copyright (c) 2019, 2023 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <assert.h>
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
  return (jlong) ortPtr;
}
/*
 * Class:     ai_onnxruntime_OnnxRuntime
 * Method:    initialiseTrainingAPIBase
 * Signature: (JI)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OnnxRuntime_initialiseTrainingAPIBase
  (JNIEnv * jniEnv, jclass clazz, jlong apiHandle, jint trainingApiVersion) {
  (void)jniEnv; (void)clazz;  // required JNI parameters not needed by functions which don't call back into Java.
  const OrtApi* api = (const OrtApi*)apiHandle;
  const OrtTrainingApi* trainingApi = api->GetTrainingApi((uint32_t)trainingApiVersion);
  return (jlong) trainingApi;
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

/*
 * Class:     ai_onnxruntime_OnnxRuntime
 * Method:    initialiseVersion
 * Signature: ()Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_OnnxRuntime_initialiseVersion
  (JNIEnv * jniEnv, jclass clazz) {
  (void)clazz;  // required JNI parameter not needed by functions which don't access their host class.
  const ORTCHAR_T* version = OrtGetApiBase()->GetVersionString();
  assert(version != NULL);
#ifdef _WIN32
  jsize len = (jsize)(wcslen(version));
  jstring versionStr = (*jniEnv)->NewString(jniEnv, (const jchar*)version, len);
#else
  jstring versionStr = (*jniEnv)->NewStringUTF(jniEnv, version);
#endif
  return versionStr;
}

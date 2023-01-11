/*
 * Copyright (c) 2019, 2022, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OnnxSequence.h"

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getMaps
 * Signature: (JJJ)[Lai/onnxruntime/OnnxMap;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OnnxSequence_getMaps(JNIEnv* jniEnv, jobject jobj,
                                                                        jlong apiHandle, jlong handle,
                                                                        jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtValue* sequence = (OrtValue*)handle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  jobjectArray outputArray = NULL;

  // Get the element count of this sequence
  size_t count;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetValueCount(sequence, &count));
  if (code == ORT_OK) {
    jclass tensorClazz = (*jniEnv)->FindClass(jniEnv, "ai/onnxruntime/OnnxMap");
    outputArray = (*jniEnv)->NewObjectArray(jniEnv, safecast_size_t_to_jsize(count), tensorClazz, NULL);
    for (size_t i = 0; i < count; i++) {
      // Extract element
      OrtValue* element;
      code = checkOrtStatus(jniEnv, api, api->GetValue(sequence, (int)i, allocator, &element));
      if (code == ORT_OK) {
        jobject str = createJavaMapFromONNX(jniEnv, api, allocator, element);
        if (str == NULL) {
          api->ReleaseValue(element);
          // bail out as exception has been thrown
          return NULL;
        }
        (*jniEnv)->SetObjectArrayElement(jniEnv, outputArray, (jsize)i, str);
      } else {
        // bail out as exception has been thrown
        return NULL;
      }
    }
  }
  return outputArray;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    getTensors
 * Signature: (JJJ)[Lai/onnxruntime/OnnxTensor;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OnnxSequence_getTensors(JNIEnv* jniEnv, jobject jobj,
                                                                           jlong apiHandle, jlong handle,
                                                                           jlong allocatorHandle) {
  (void)jobj;  // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  OrtValue* sequence = (OrtValue*)handle;
  OrtAllocator* allocator = (OrtAllocator*)allocatorHandle;

  jobjectArray outputArray = NULL;

  // Get the element count of this sequence
  size_t count;
  OrtErrorCode code = checkOrtStatus(jniEnv, api, api->GetValueCount(sequence, &count));
  if (code == ORT_OK) {
    jclass tensorClazz = (*jniEnv)->FindClass(jniEnv, "ai/onnxruntime/OnnxTensor");
    outputArray = (*jniEnv)->NewObjectArray(jniEnv, safecast_size_t_to_jsize(count), tensorClazz, NULL);
    for (size_t i = 0; i < count; i++) {
      // Extract element
      OrtValue* element;
      code = checkOrtStatus(jniEnv, api, api->GetValue(sequence, (int)i, allocator, &element));
      if (code == ORT_OK) {
        jobject str = createJavaTensorFromONNX(jniEnv, api, allocator, element);
        if (str == NULL) {
          api->ReleaseValue(element);
          // bail out as exception has been thrown
          return NULL;
        }
        (*jniEnv)->SetObjectArrayElement(jniEnv, outputArray, (jsize)i, str);
      } else {
        // bail out as exception has been thrown
        return NULL;
      }
    }
  }
  return outputArray;
}

/*
 * Class:     ai_onnxruntime_OnnxSequence
 * Method:    close
 * Signature: (J)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OnnxSequence_close(JNIEnv* jniEnv, jobject jobj, jlong apiHandle,
                                                              jlong handle) {
  (void)jniEnv;
  (void)jobj;  // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*)apiHandle;
  api->ReleaseValue((OrtValue*)handle);
}

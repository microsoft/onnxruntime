/*
 * Copyright (c) 2024, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include <string.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtLoraAdapter.h"

/*
 * Class:     ai_onnxruntime_OrtLoraAdapter
 * Method:    createLoraAdapter
 * Signature: (JLjava/lang/String;J)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtLoraAdapter_createLoraAdapter
    (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jstring loraPath, jlong allocatorHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
  OrtLoraAdapter* lora;

#ifdef _WIN32
  const jchar* cPath = (*jniEnv)->GetStringChars(jniEnv, loraPath, NULL);
  size_t stringLength = (*jniEnv)->GetStringLength(jniEnv, loraPath);
  wchar_t* newString = (wchar_t*)calloc(stringLength + 1, sizeof(wchar_t));
  if (newString == NULL) {
    (*jniEnv)->ReleaseStringChars(jniEnv, loraPath, cPath);
    throwOrtException(jniEnv, 1, "Not enough memory");
    return 0;
  }
  wcsncpy_s(newString, stringLength + 1, (const wchar_t*)cPath, stringLength);
  checkOrtStatus(jniEnv, api, api->CreateLoraAdapter(newString, allocator, &lora));
  free(newString);
  (*jniEnv)->ReleaseStringChars(jniEnv, loraPath, cPath);
#else
  const char* cPath = (*jniEnv)->GetStringUTFChars(jniEnv, loraPath, NULL);
  checkOrtStatus(jniEnv, api, api->CreateLoraAdapter(cPath, allocator, &lora));
  (*jniEnv)->ReleaseStringUTFChars(jniEnv, loraPath, cPath);
#endif

  return (jlong) lora;
}

/*
 * Class:     ai_onnxruntime_OrtLoraAdapter
 * Method:    createLoraAdapterFromBuffer
 * Signature: (JLjava/nio/ByteBuffer;IIJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtLoraAdapter_createLoraAdapterFromBuffer
    (JNIEnv* jniEnv, jclass jclazz, jlong apiHandle, jobject buffer, jint bufferPos, jint bufferSize, jlong allocatorHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
  OrtLoraAdapter* lora;

  // Extract the buffer
  char* bufferArr = (char*)(*jniEnv)->GetDirectBufferAddress(jniEnv, buffer);
  // Increment by bufferPos bytes
  bufferArr = bufferArr + bufferPos;

  // Create the adapter
  checkOrtStatus(jniEnv, api, api->CreateLoraAdapterFromArray((const uint8_t*) bufferArr, bufferSize, allocator, &lora));

  return (jlong) lora;
}

/*
 * Class:     ai_onnxruntime_OrtLoraAdapter
 * Method:    createLoraAdapterFromArray
 * Signature: (J[BJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtLoraAdapter_createLoraAdapterFromArray
    (JNIEnv* jniEnv, jclass jclazz, jlong apiHandle, jbyteArray jLoraArray, jlong allocatorHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtAllocator* allocator = (OrtAllocator*) allocatorHandle;
  OrtLoraAdapter* lora;

  size_t loraLength = (*jniEnv)->GetArrayLength(jniEnv, jLoraArray);
  if (loraLength == 0) {
    throwOrtException(jniEnv, 2, "Invalid LoRA, the byte array is zero length.");
    return 0;
  }

  // Get a reference to the byte array elements
  jbyte* loraArr = (*jniEnv)->GetByteArrayElements(jniEnv, jLoraArray, NULL);
  checkOrtStatus(jniEnv, api, api->CreateLoraAdapterFromArray((const uint8_t*) loraArr, loraLength, allocator, &lora));
  // Release the C array.
  (*jniEnv)->ReleaseByteArrayElements(jniEnv, jLoraArray, loraArr, JNI_ABORT);

  return (jlong) lora;
}

/*
 * Class:     ai_onnxruntime_OrtLoraAdapter
 * Method:    close
 * Signature: (JJ)V
 */
JNIEXPORT void JNICALL Java_ai_onnxruntime_OrtLoraAdapter_close
    (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong loraHandle) {
  (void) jniEnv; (void) jclazz; // Required JNI parameters not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  api->ReleaseLoraAdapter((OrtLoraAdapter*) loraHandle);
}


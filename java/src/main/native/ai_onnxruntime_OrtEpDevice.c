/*
 * Copyright (c) 2025 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtEpDevice.h"

/*
 * Class:     ai_onnxruntime_OrtEpDevice
 * Method:    getName
 * Signature: (JJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_OrtEpDevice_getName
  (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong nativeHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtEpDevice* epDevice = (OrtEpDevice*) nativeHandle;
  const char* name = api->EpDevice_EpName(epDevice);
  jstring nameStr = (*jniEnv)->NewStringUTF(jniEnv, name);
  return nameStr;
}

/*
 * Class:     ai_onnxruntime_OrtEpDevice
 * Method:    getVendor
 * Signature: (JJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_OrtEpDevice_getVendor
  (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong nativeHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtEpDevice* epDevice = (OrtEpDevice*) nativeHandle;
  const char* vendor = api->EpDevice_EpVendor(epDevice);
  jstring vendorStr = (*jniEnv)->NewStringUTF(jniEnv, vendor);
  return vendorStr;
}

/*
 * Class:     ai_onnxruntime_OrtEpDevice
 * Method:    getMetadata
 * Signature: (JJ)[[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtEpDevice_getMetadata
  (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong nativeHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtEpDevice* epDevice = (OrtEpDevice*) nativeHandle;
  const OrtKeyValuePairs* kvp = api->EpDevice_EpMetadata(epDevice);
  jobjectArray pair = convertOrtKeyValuePairsToArrays(jniEnv, api, kvp);
  return pair;
}

/*
 * Class:     ai_onnxruntime_OrtEpDevice
 * Method:    getOptions
 * Signature: (JJ)[[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtEpDevice_getOptions
  (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong nativeHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtEpDevice* epDevice = (OrtEpDevice*) nativeHandle;
  const OrtKeyValuePairs* kvp = api->EpDevice_EpOptions(epDevice);
  jobjectArray pair = convertOrtKeyValuePairsToArrays(jniEnv, api, kvp);
  return pair;
}

/*
 * Class:     ai_onnxruntime_OrtEpDevice
 * Method:    getDeviceHandle
 * Signature: (JJ)J
 */
JNIEXPORT jlong JNICALL Java_ai_onnxruntime_OrtEpDevice_getDeviceHandle
  (JNIEnv * jniEnv, jclass jclazz, jlong apiHandle, jlong nativeHandle) {
  (void) jniEnv; (void) jclazz; // Required JNI parameters not needed by functions which don't need to access their host object or the JVM.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtEpDevice* epDevice = (OrtEpDevice*) nativeHandle;
  const OrtHardwareDevice* device = api->EpDevice_Device(epDevice);
  return (jlong) device;
}

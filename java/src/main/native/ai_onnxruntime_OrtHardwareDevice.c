/*
 * Copyright (c) 2025 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
#include <jni.h>
#include "onnxruntime/core/session/onnxruntime_c_api.h"
#include "OrtJniUtil.h"
#include "ai_onnxruntime_OrtHardwareDevice.h"

/*
 * Class:     ai_onnxruntime_OrtHardwareDevice
 * Method:    getVendor
 * Signature: (JJ)Ljava/lang/String;
 */
JNIEXPORT jstring JNICALL Java_ai_onnxruntime_OrtHardwareDevice_getVendor
  (JNIEnv * jniEnv, jclass clazz, jlong apiHandle, jlong nativeHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtHardwareDevice* device = (OrtHardwareDevice*) nativeHandle;
  const char* vendor = api->HardwareDevice_Vendor(device);
  jstring vendorStr = (*jniEnv)->NewStringUTF(jniEnv, vendor);
  return vendorStr;
}

/*
 * Class:     ai_onnxruntime_OrtHardwareDevice
 * Method:    getMetadata
 * Signature: (JJ)[[Ljava/lang/String;
 */
JNIEXPORT jobjectArray JNICALL Java_ai_onnxruntime_OrtHardwareDevice_getMetadata
  (JNIEnv * jniEnv, jclass clazz, jlong apiHandle, jlong nativeHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtHardwareDevice* device = (OrtHardwareDevice*) nativeHandle;
  OrtKeyValuePairs* kvp = api->HardwareDevice_Metadata(device);
  jobjectarray pair = convertOrtKeyValuePairsToArrays(*jniEnv, api, kvp);
  return pair;
}

/*
 * Class:     ai_onnxruntime_OrtHardwareDevice
 * Method:    getDeviceType
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_ai_onnxruntime_OrtHardwareDevice_getDeviceType
  (JNIEnv * jniEnv, jclass clazz, jlong apiHandle, jlong nativeHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtHardwareDevice* device = (OrtHardwareDevice*) nativeHandle;
  OrtHardwareDeviceType type = api->HardwareDevice_DeviceId(device);
  jint output = 0;
  // Must be kept aligned with the Java OrtHardwareDeviceType enum.
  switch (type) {
    case OrtHardwareDeviceType_CPU:
      output = 0;
      break;
    case OrtHardwareDeviceType_GPU:
      output = 1;
      break;
    case OrtHardwareDeviceType_NPU:
      output = 2;
      break;
    default:
      throwOrtException(jniEnv, convertErrorCode(ORT_NOT_IMPLEMENTED), "Unexpected device type found. Only CPU, GPU and NPU are supported.");
      break;
  }
  return output;
}

/*
 * Class:     ai_onnxruntime_OrtHardwareDevice
 * Method:    getDeviceId
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_ai_onnxruntime_OrtHardwareDevice_getDeviceId
  (JNIEnv * jniEnv, jclass clazz, jlong apiHandle, jlong nativeHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtHardwareDevice* device = (OrtHardwareDevice*) nativeHandle;
  uint32_t id = api->HardwareDevice_DeviceId(device);
  return (jint) id;
}

/*
 * Class:     ai_onnxruntime_OrtHardwareDevice
 * Method:    getVendorId
 * Signature: (JJ)I
 */
JNIEXPORT jint JNICALL Java_ai_onnxruntime_OrtHardwareDevice_getVendorId
  (JNIEnv * jniEnv, jclass clazz, jlong apiHandle, jlong nativeHandle) {
  (void) jclazz; // Required JNI parameter not needed by functions which don't need to access their host object.
  const OrtApi* api = (const OrtApi*) apiHandle;
  OrtHardwareDevice* device = (OrtHardwareDevice*) nativeHandle;
  uint32_t id = api->HardwareDevice_VendorId(device);
  return (jint) id;
}

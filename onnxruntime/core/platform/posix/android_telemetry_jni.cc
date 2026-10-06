// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <jni.h>

extern "C" {

void Java_com_microsoft_applications_events_HttpClient_createClientInstance(JNIEnv*, jobject);
void Java_com_microsoft_applications_events_HttpClient_deleteClientInstance(JNIEnv*);
void Java_com_microsoft_applications_events_HttpClient_dispatchCallback(
    JNIEnv*, jobject, jstring, jint, jobjectArray, jbyteArray);
void Java_com_microsoft_applications_events_HttpClient_onCostChange(JNIEnv*, jobject, jboolean);
void Java_com_microsoft_applications_events_HttpClient_onPowerChange(JNIEnv*, jobject, jboolean, jboolean);
void Java_com_microsoft_applications_events_HttpClient_setCacheFilePath(JNIEnv*, jobject, jstring);
void Java_com_microsoft_applications_events_HttpClient_setDeviceInfo(
    JNIEnv*, jobject, jstring, jstring, jstring);
void Java_com_microsoft_applications_events_HttpClient_setSystemInfo(
    JNIEnv*, jobject, jstring, jstring, jstring, jstring, jstring, jstring, jstring);

JNIEXPORT void JNICALL Java_ai_onnxruntime_telemetry_HttpClient_createClientInstance(
    JNIEnv* env, jobject client) {
  Java_com_microsoft_applications_events_HttpClient_createClientInstance(env, client);
}

JNIEXPORT void JNICALL Java_ai_onnxruntime_telemetry_HttpClient_deleteClientInstance(
    JNIEnv* env, jobject) {
  Java_com_microsoft_applications_events_HttpClient_deleteClientInstance(env);
}

JNIEXPORT void JNICALL Java_ai_onnxruntime_telemetry_HttpClient_dispatchCallback(
    JNIEnv* env, jobject client, jstring id, jint status_code, jobjectArray headers, jbyteArray body) {
  Java_com_microsoft_applications_events_HttpClient_dispatchCallback(
      env, client, id, status_code, headers, body);
}

JNIEXPORT void JNICALL Java_ai_onnxruntime_telemetry_HttpClient_onCostChange(
    JNIEnv* env, jobject client, jboolean is_metered) {
  Java_com_microsoft_applications_events_HttpClient_onCostChange(env, client, is_metered);
}

JNIEXPORT void JNICALL Java_ai_onnxruntime_telemetry_HttpClient_onPowerChange(
    JNIEnv* env, jobject client, jboolean is_charging, jboolean is_low) {
  Java_com_microsoft_applications_events_HttpClient_onPowerChange(
      env, client, is_charging, is_low);
}

JNIEXPORT void JNICALL Java_ai_onnxruntime_telemetry_HttpClient_setCacheFilePath(
    JNIEnv* env, jobject client, jstring path) {
  Java_com_microsoft_applications_events_HttpClient_setCacheFilePath(env, client, path);
}

JNIEXPORT void JNICALL Java_ai_onnxruntime_telemetry_HttpClient_setDeviceInfo(
    JNIEnv* env, jobject client, jstring id, jstring manufacturer, jstring model) {
  Java_com_microsoft_applications_events_HttpClient_setDeviceInfo(
      env, client, id, manufacturer, model);
}

JNIEXPORT void JNICALL Java_ai_onnxruntime_telemetry_HttpClient_setSystemInfo(
    JNIEnv* env, jobject client, jstring app_id, jstring app_version, jstring app_language,
    jstring os_major_version, jstring os_full_version, jstring time_zone, jstring device_class) {
  Java_com_microsoft_applications_events_HttpClient_setSystemInfo(
      env, client, app_id, app_version, app_language, os_major_version, os_full_version, time_zone,
      device_class);
}

}  // extern "C"

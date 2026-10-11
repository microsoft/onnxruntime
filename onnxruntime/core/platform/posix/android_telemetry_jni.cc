// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <jni.h>
#include "core/platform/telemetry_strings.h"

namespace {
jstring BoundedTelemetryJniString(JNIEnv* env, jstring value,
                                  size_t max_bytes = onnxruntime::kMaxTelemetryStringLength, bool truncate = true) {
  if (value == nullptr) return nullptr;
  const jsize length = env->GetStringLength(value);
  if (env->ExceptionCheck()) return nullptr;
  std::array<jchar, onnxruntime::telemetry_detail::kMaxTelemetryPathBytes + 1> buffer{};
  const jsize count = static_cast<jsize>(std::min(static_cast<size_t>(length), max_bytes + 1));
  env->GetStringRegion(value, 0, count, buffer.data());
  if (env->ExceptionCheck()) return nullptr;
  const size_t prefix_length = onnxruntime::telemetry_detail::TelemetryUnicodePrefixLength(
      buffer.data(), static_cast<size_t>(count), max_bytes, true);
  if (prefix_length == static_cast<size_t>(length)) return value;
  if (!truncate) {
    const jclass exception = env->FindClass("java/lang/IllegalArgumentException");
    if (exception != nullptr) env->ThrowNew(exception, "Telemetry cache path exceeds the byte limit");
    return nullptr;
  }
  return env->NewString(buffer.data(), static_cast<jsize>(prefix_length));
}
}  // namespace

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
  path = BoundedTelemetryJniString(env, path, onnxruntime::telemetry_detail::kMaxTelemetryPathBytes, false);
  if (env->ExceptionCheck()) return;
  Java_com_microsoft_applications_events_HttpClient_setCacheFilePath(env, client, path);
}

JNIEXPORT void JNICALL Java_ai_onnxruntime_telemetry_HttpClient_setDeviceInfo(
    JNIEnv* env, jobject client, jstring id, jstring manufacturer, jstring model) {
  std::array<jstring, 3> values{id, manufacturer, model};
  for (auto& value : values) {
    value = BoundedTelemetryJniString(env, value);
    if (env->ExceptionCheck()) return;
  }
  Java_com_microsoft_applications_events_HttpClient_setDeviceInfo(
      env, client, values[0], values[1], values[2]);
}

JNIEXPORT void JNICALL Java_ai_onnxruntime_telemetry_HttpClient_setSystemInfo(
    JNIEnv* env, jobject client, jstring app_id, jstring app_version, jstring app_language,
    jstring os_major_version, jstring os_full_version, jstring time_zone, jstring device_class) {
  std::array<jstring, 7> values{
      app_id, app_version, app_language, os_major_version, os_full_version, time_zone, device_class};
  for (auto& value : values) {
    value = BoundedTelemetryJniString(env, value);
    if (env->ExceptionCheck()) return;
  }
  Java_com_microsoft_applications_events_HttpClient_setSystemInfo(
      env, client, values[0], values[1], values[2], values[3], values[4], values[5], values[6]);
}

}  // extern "C"

#pragma once

#ifdef __ANDROID__
#include <android/log.h>

#define LOG_TAG "onnxruntimejsi"

#define LOGI(fmt, ...) \
  __android_log_print(ANDROID_LOG_INFO, LOG_TAG, fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) \
  __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, fmt, ##__VA_ARGS__)
#define LOGD(fmt, ...) \
  __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, fmt, ##__VA_ARGS__)

#elif defined(__APPLE__)

#import <Foundation/Foundation.h>

#define LOGI(fmt, ...) NSLog(@"[INFO] " fmt, ##__VA_ARGS__)
#define LOGE(fmt, ...) NSLog(@"[ERROR] " fmt, ##__VA_ARGS__)
#define LOGD(fmt, ...) NSLog(@"[DEBUG] " fmt, ##__VA_ARGS__)

#else

#define LOGI(fmt, ...) printf("[INFO] " fmt "\n", ##__VA_ARGS__)
#define LOGE(fmt, ...) printf("[ERROR] " fmt "\n", ##__VA_ARGS__)
#define LOGD(fmt, ...) printf("[DEBUG] " fmt "\n", ##__VA_ARGS__)

#endif

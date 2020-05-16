//
// Created by daquexian on 5/21/18.
//

#ifndef PROJECT_ANDROID_LOG_HELPER_H
#define PROJECT_ANDROID_LOG_HELPER_H

#include <android/log.h>

#include <common/log_helper.h>

#define LOG_TAG "DNN Library"

#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)
#define LOGW(...) __android_log_print(ANDROID_LOG_WARN, LOG_TAG, __VA_ARGS__)
#define LOGD(...) __android_log_print(ANDROID_LOG_DEBUG, LOG_TAG, __VA_ARGS__)
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)

#endif  // PROJECT_ANDROID_LOG_HELPER_H

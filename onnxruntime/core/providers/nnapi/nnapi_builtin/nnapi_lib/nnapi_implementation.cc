/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include "nnapi_implementation.h"

#include <dlfcn.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdlib>

#include "NeuralNetworksTypes.h"

#ifdef __ANDROID__
#include <sys/system_properties.h>
#endif  // __ANDROID__

#define NNAPI_LOG(format, ...) fprintf(stderr, format "\n", __VA_ARGS__);

namespace {

#ifdef __ANDROID__
int32_t GetAndroidSdkVersion() {
  const char* sdkProp = "ro.build.version.sdk";
  char sdkVersion[PROP_VALUE_MAX];
  int length = __system_property_get(sdkProp, sdkVersion);
  if (length != 0) {
    int32_t result = 0;
    for (int i = 0; i < length; ++i) {
      int digit = sdkVersion[i] - '0';
      if (digit < 0 || digit > 9) {
        // Non-numeric SDK version, assume it's higher than expected;
        return 0xffff;
      }
      result = result * 10 + digit;
    }
    return result;
  }
  return 0;
}
#endif  // __ANDROID__

void* LoadFunction(void* handle, const char* name, bool optional) {
  if (handle == nullptr) {
    return nullptr;
  }
  void* fn = dlsym(handle, name);
  if (fn == nullptr && !optional) {
    NNAPI_LOG("nnapi error: unable to open function %s", name);
  }
  return fn;
}

#ifndef __ANDROID__
// Add /dev/shm implementation of shared memory for non-Android platforms
int ASharedMemory_create(const char* name, size_t size) {
  // Each call to ASharedMemory_create produces a unique memory space, hence
  // name should be unique, otherwise two calls to create memory regions using
  // the same 'name', will collide.
  // Caller is responsible to provide a unique name.

  // Make sure new shared memory region is created: shm_open return an error if
  // shm object with given name already exists (O_CREAT | O_EXCL)
  int fd = shm_open(name, O_RDWR | O_CREAT | O_EXCL, 0644);
  if (fd < 0) {
    return fd;
  }
  int result = ftruncate(fd, size);
  if (result < 0) {
    close(fd);
    return -1;
  }
  return fd;
}

// Determine the NnApi version from loaded entry points
uint32_t CalculateAndroidSdkVersion(NnApi const& nnapi) {
  // Test for specific NNAPI 1.0, 1.1, 1.2 and 1.3 functions
  bool has_10 = nnapi.ANeuralNetworksMemory_createFromFd != nullptr;
  bool has_11 =
      nnapi.ANeuralNetworksModel_relaxComputationFloat32toFloat16 != nullptr;
  bool has_12 = nnapi.ANeuralNetworks_getDeviceCount != nullptr;
  bool has_13 = nnapi.ANeuralNetworksCompilation_setTimeout != nullptr;
  bool has_14 = nnapi.ANeuralNetworks_getRuntimeFeatureLevel != nullptr;

  uint32_t sdk_version = 0;
  if (has_10) {
    sdk_version = 27;
  }
  if (sdk_version == 27 && has_11) {
    sdk_version = 28;
  }
  if (sdk_version == 28 && has_12) {
    sdk_version = 29;
  }
  if (sdk_version == 29 && has_13) {
    sdk_version = 30;
  }
  if (sdk_version == 30 && has_14) {
    sdk_version = 31;
  }
  return sdk_version;
}
#else

ASharedMemory_create_fn getASharedMemory_create() {
  // ASharedMemory_create has different implementations in Android depending on
  // the partition. Generally it can be loaded from libandroid.so but in vendor
  // partition (e.g. if a HAL wants to use NNAPI) it is only accessible through
  // libcutils.
  void* libandroid = nullptr;
  libandroid = dlopen("libandroid.so", RTLD_LAZY | RTLD_LOCAL);
  if (libandroid != nullptr) {
    return reinterpret_cast<ASharedMemory_create_fn>(
        LoadFunction(libandroid, "ASharedMemory_create", false));
  }

  std::string libandroid_error = dlerror();
  void* cutils_handle = dlopen("libcutils.so", RTLD_LAZY | RTLD_LOCAL);
  if (cutils_handle != nullptr) {
    return reinterpret_cast<ASharedMemory_create_fn>(
        LoadFunction(cutils_handle, "ashmem_create_region", false));
  }

  NNAPI_LOG(
      "nnapi error: unable to open both library %s (%s) and library %s "
      "(%s)",
      "libandroid.so", libandroid_error.c_str(), "libcutils.so", dlerror());
  return nullptr;
}

#endif  // __ANDROID__

#define LOAD_FUNCTION(handle, name)         \
  nnapi.name = reinterpret_cast<name##_fn>( \
      LoadFunction(handle, #name, /*optional*/ false));

#define LOAD_FUNCTION_OPTIONAL(handle, name) \
  nnapi.name = reinterpret_cast<name##_fn>(  \
      LoadFunction(handle, #name, /*optional*/ true));

#define LOAD_FUNCTION_RENAME(handle, name, symbol) \
  nnapi.name = reinterpret_cast<name##_fn>(        \
      LoadFunction(handle, symbol, /*optional*/ false));

const NnApi LoadNnApi() {
  NnApi nnapi = {};
  nnapi.android_sdk_version = 0;

#ifdef __ANDROID__
  nnapi.android_sdk_version = GetAndroidSdkVersion();
  if (nnapi.android_sdk_version < 27) {
    NNAPI_LOG("nnapi error: requires android sdk version to be at least %d",
              27);
    nnapi.nnapi_exists = false;
    return nnapi;
  }
#endif  // __ANDROID__

  void* libneuralnetworks = nullptr;
  // TODO(b/123243014): change RTLD_LOCAL? Assumes there can be multiple
  // instances of nn api RT
  static const char nnapi_library_name[] = "libneuralnetworks.so";
  libneuralnetworks = dlopen(nnapi_library_name, RTLD_LAZY | RTLD_LOCAL);
#ifdef __ANDROID__
  // Note: If there is an problem trying to open the NNAPI library on a
  // non-Android system, the error message is suppressed. This is to avoid
  // showing confusing errors when running in environments that do not support
  // NNAPI. As more platforms support NNAPI, the #ifdef logic above can be
  // expanded.
  if (libneuralnetworks == nullptr) {
    const char* error = dlerror();
    if (error) {
      NNAPI_LOG("%s\n", error);
    }
    NNAPI_LOG("nnapi error: unable to open library %s", nnapi_library_name);
  }
#endif  // __ANDROID__

  nnapi.nnapi_exists = libneuralnetworks != nullptr;

  // API 27 (NN 1.0) methods.
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksMemory_createFromFd);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksMemory_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_create);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_finish);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_addOperand);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_setOperandValue);
  LOAD_FUNCTION_OPTIONAL(
      libneuralnetworks,
      ANeuralNetworksModel_setOperandSymmPerChannelQuantParams);
  LOAD_FUNCTION(libneuralnetworks,
                ANeuralNetworksModel_setOperandValueFromMemory);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksModel_addOperation);
  LOAD_FUNCTION(libneuralnetworks,
                ANeuralNetworksModel_identifyInputsAndOutputs);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_create);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_setPreference);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksCompilation_finish);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_create);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_free);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setInput);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setInputFromMemory);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_setOutput);
  LOAD_FUNCTION(libneuralnetworks,
                ANeuralNetworksExecution_setOutputFromMemory);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksExecution_startCompute);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksEvent_wait);
  LOAD_FUNCTION(libneuralnetworks, ANeuralNetworksEvent_free);

#ifdef __ANDROID__
  nnapi.ASharedMemory_create = getASharedMemory_create();
#else
  // Mock ASharedMemory_create only if libneuralnetworks.so was successfully
  // loaded. This ensures identical behaviour on platforms which use this
  // implementation, but don't have libneuralnetworks.so library, and
  // platforms which use nnapi_implementation_disabled.cc stub.
  if (libneuralnetworks != nullptr) {
    nnapi.ASharedMemory_create = ASharedMemory_create;
  }
#endif  // __ANDROID__

  // API 28 (NN 1.1) methods.
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksModel_relaxComputationFloat32toFloat16);

  // API 29 (NN 1.2) methods.
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworks_getDeviceCount);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworks_getDevice);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksDevice_getName);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksDevice_getVersion);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksDevice_getFeatureLevel);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksDevice_getType);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksModel_getSupportedOperationsForDevices);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksCompilation_createForDevices);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksCompilation_setCaching);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksExecution_compute);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_getOutputOperandRank);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_getOutputOperandDimensions);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksBurst_create);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksBurst_free);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_burstCompute);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksMemory_createFromAHardwareBuffer);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_setMeasureTiming);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_getDuration);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksDevice_getExtensionSupport);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksModel_getExtensionOperandType);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksModel_getExtensionOperationType);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksModel_setOperandExtensionData);

  // API 30 (NNAPI 1.3) methods.
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksCompilation_setTimeout);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksCompilation_setPriority);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_setTimeout);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_setLoopTimeout);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_create);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_free);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksMemoryDesc_addInputRole);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksMemoryDesc_addOutputRole);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksMemoryDesc_setDimensions);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemoryDesc_finish);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksMemory_createFromDesc);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks, ANeuralNetworksMemory_copy);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksEvent_createFromSyncFenceFd);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksEvent_getSyncFenceFd);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_startComputeWithDependencies);

  // API 31 methods
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworks_getRuntimeFeatureLevel);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_enableInputAndOutputPadding);
  LOAD_FUNCTION_OPTIONAL(libneuralnetworks,
                         ANeuralNetworksExecution_setReusable);
#ifndef __ANDROID__
  // If libneuralnetworks.so is loaded, but android_sdk_version is not set,
  // then determine android_sdk_version by testing which functions are
  // available.
  if (nnapi.nnapi_exists && nnapi.android_sdk_version == 0) {
    nnapi.android_sdk_version = CalculateAndroidSdkVersion(nnapi);
  }
#endif  // __ANDROID__
  // Determin NNAPI Runtime feature level.
  if (nnapi.ANeuralNetworks_getRuntimeFeatureLevel) {
    nnapi.nnapi_runtime_feature_level =
        nnapi.ANeuralNetworks_getRuntimeFeatureLevel();
  } else {
    nnapi.nnapi_runtime_feature_level = nnapi.android_sdk_version;
  }

  return nnapi;
}

}  // namespace

const NnApi* NnApiImplementation() {
  static const NnApi nnapi = LoadNnApi();
  return &nnapi;
}

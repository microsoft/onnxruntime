// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <cstdlib>
#include <cstring>
#include <exception>
#include <new>
#include <thread>

#include "OrtJniUtil.h"

namespace {

struct ReadCallbackTestAllocator {
  OrtAllocator allocator{};
  const OrtMemoryInfo* memory_info = nullptr;
  jint mode = 0;
  size_t allocation_count = 0;
};

void* ORT_API_CALL Allocate(OrtAllocator* allocator, size_t size) {
  auto& test_allocator = *reinterpret_cast<ReadCallbackTestAllocator*>(allocator);
  ++test_allocator.allocation_count;
  if (test_allocator.mode == 1) {
    return nullptr;
  }
#ifndef ORT_NO_EXCEPTIONS
  if (test_allocator.mode == 2) {
    throw std::bad_alloc();
  }
#endif
  return std::malloc(size);
}

void ORT_API_CALL Free(OrtAllocator*, void* buffer) {
  std::free(buffer);
}

const OrtMemoryInfo* ORT_API_CALL GetInfo(const OrtAllocator* allocator) {
  return reinterpret_cast<const ReadCallbackTestAllocator*>(allocator)->memory_info;
}

struct HolderAllocationFailure {
  const JNINativeInterface_* original_functions;
  jclass out_of_memory_error;
  const OrtApi* api;
  jint counts[3]{};
};

thread_local HolderAllocationFailure* holder_failure = nullptr;

jobject JNICALL CountNewGlobalRef(JNIEnv* env, jobject object) {
  jobject result = holder_failure->original_functions->NewGlobalRef(env, object);
  if (result != nullptr) {
    ++holder_failure->counts[0];
  }
  return result;
}

void JNICALL CountDeleteGlobalRef(JNIEnv* env, jobject object) {
  ++holder_failure->counts[1];
  holder_failure->original_functions->DeleteGlobalRef(env, object);
}

jobject JNICALL FailNewObject(JNIEnv* env, jclass, jmethodID, ...) {
  holder_failure->original_functions->ThrowNew(
      env, holder_failure->out_of_memory_error, "Injected EPContext ownership holder allocation failure");
  return nullptr;
}

OrtStatus* ORT_API_CALL RejectRegistration(OrtSessionOptions*, OrtReadNamedBufferFunc, void*) noexcept {
  ++holder_failure->counts[2];
  return holder_failure->api->CreateStatus(
      ORT_FAIL, "Native registration attempted before ownership holder construction");
}

}  // namespace

extern "C" JNIEXPORT jboolean JNICALL
Java_ai_onnxruntime_EpContextDataCallbackTest_supportsAllocatorExceptions(JNIEnv*, jclass) {
#ifdef ORT_NO_EXCEPTIONS
  return JNI_FALSE;
#else
  return JNI_TRUE;
#endif
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_EpContextDataCallbackTest_exerciseReadCallbackAllocator(
    JNIEnv* env, jclass, jlong api_handle, jobject callback, jint allocation_mode) {
  const auto* api = reinterpret_cast<const OrtApi*>(api_handle);
  OrtAllocator* default_allocator = nullptr;
  if (checkOrtStatus(env, api, api->GetAllocatorWithDefaultOptions(&default_allocator)) != ORT_OK) {
    return;
  }

  ReadCallbackTestAllocator allocator;
  allocator.allocator.version = ORT_API_VERSION;
  allocator.allocator.Alloc = Allocate;
  allocator.allocator.Free = Free;
  allocator.allocator.Info = GetInfo;
  allocator.mode = allocation_mode;
  if (checkOrtStatus(env, api, api->AllocatorGetInfo(default_allocator, &allocator.memory_info)) != ORT_OK) {
    return;
  }

  EpContextDataCallbackState* state = createEpContextDataCallbackState(
      env, api, callback, "read", "(Ljava/lang/String;)[B", 1024);
  if (state == nullptr) {
    return;
  }

  JavaVM* jvm = nullptr;
  if (env->GetJavaVM(&jvm) != JNI_OK) {
    releaseEpContextDataCallbackState(env, state);
    throwOrtException(env, ORT_FAIL, "Failed to obtain the JVM for the native worker test");
    return;
  }

  const char* failure = nullptr;
#ifndef ORT_NO_EXCEPTIONS
  try {
#endif
    std::thread worker([&]() {
      void* worker_env = nullptr;
      if (jvm->GetEnv(&worker_env, JNI_VERSION_1_6) != JNI_EDETACHED) {
        failure = "The native worker was already attached before the callback";
      }

      void* output = nullptr;
      size_t output_size = 0;
      OrtStatus* status = nullptr;
#ifndef ORT_NO_EXCEPTIONS
      try {
#endif
        status = javaEpContextDataReadCallback(state, "allocation-test", &allocator.allocator, &output, &output_size);
#ifndef ORT_NO_EXCEPTIONS
      } catch (const std::bad_alloc&) {
        failure = "An allocator exception escaped the EPContext read callback";
      }
#endif

      const jint attachment_status = jvm->GetEnv(&worker_env, JNI_VERSION_1_6);
      if (attachment_status != JNI_EDETACHED) {
        failure = "The EPContext read callback did not detach its native worker";
        if (attachment_status == JNI_OK) {
          jvm->DetachCurrentThread();
        }
      }

      const unsigned char expected[] = {1, 2, 3};
      if (allocation_mode == 0) {
        if (status != nullptr || output == nullptr || output_size != sizeof(expected) ||
            std::memcmp(output, expected, sizeof(expected)) != 0) {
          failure = "The successful EPContext read callback did not return the expected payload";
        }
      } else if (status == nullptr || output != nullptr || output_size != 0) {
        failure = "The failed EPContext read allocation did not return a status with empty outputs";
      }
      if (allocator.allocation_count != 1) {
        failure = "The EPContext read callback did not exercise the injected allocator";
      }
      if (output != nullptr) {
        allocator.allocator.Free(&allocator.allocator, output);
      }
      if (status != nullptr) {
        api->ReleaseStatus(status);
      }
    });
    worker.join();
#ifndef ORT_NO_EXCEPTIONS
  } catch (const std::exception& exception) {
    releaseEpContextDataCallbackState(env, state);
    throwOrtException(env, ORT_FAIL, exception.what());
    return;
  }
#endif
  releaseEpContextDataCallbackState(env, state);
  if (failure != nullptr) {
    throwOrtException(env, ORT_FAIL, failure);
  }
}

extern "C" JNIEXPORT void JNICALL
Java_ai_onnxruntime_EpContextDataCallbackTest_failReadCallbackHolderAllocation(
    JNIEnv* env, jclass, jlong api_handle, jlong options_handle, jobject callback, jintArray counts) {
  const auto* api = reinterpret_cast<const OrtApi*>(api_handle);
  jclass out_of_memory_error = env->FindClass("java/lang/OutOfMemoryError");
  if (out_of_memory_error == nullptr) {
    return;
  }
  jclass options_class = env->FindClass("ai/onnxruntime/OrtSession$SessionOptions");
  if (options_class == nullptr) {
    env->DeleteLocalRef(out_of_memory_error);
    return;
  }
  jmethodID set_callback = env->GetStaticMethodID(
      options_class, "setEpContextDataReadCallback",
      "(JJLai/onnxruntime/OrtSession$SessionOptions$EpContextDataReadCallback;J)"
      "Lai/onnxruntime/OrtSession$SessionOptions$EpContextDataReadCallbackRegistration;");
  if (set_callback == nullptr) {
    env->DeleteLocalRef(options_class);
    env->DeleteLocalRef(out_of_memory_error);
    return;
  }

  HolderAllocationFailure failure{env->functions, out_of_memory_error, api, {}};
  JNINativeInterface_ injected_functions = *env->functions;
  injected_functions.NewGlobalRef = CountNewGlobalRef;
  injected_functions.DeleteGlobalRef = CountDeleteGlobalRef;
  injected_functions.NewObject = FailNewObject;
  OrtApi injected_api = *api;
  injected_api.SessionOptionsSetEpContextDataReadFunc = RejectRegistration;

  // Replace only this calling thread's JNI table; other JVM threads are unaffected.
  holder_failure = &failure;
  env->functions = &injected_functions;
  jobject registration = env->CallStaticObjectMethod(
      options_class, set_callback, reinterpret_cast<jlong>(&injected_api), options_handle, callback, jlong{1024});
  env->functions = failure.original_functions;
  holder_failure = nullptr;

  jthrowable exception = env->ExceptionOccurred();
  env->ExceptionClear();
  env->SetIntArrayRegion(counts, 0, 3, failure.counts);
  env->DeleteLocalRef(options_class);
  env->DeleteLocalRef(out_of_memory_error);
  if (registration != nullptr) {
    env->DeleteLocalRef(registration);
  }
  if (exception != nullptr) {
    env->Throw(exception);
    env->DeleteLocalRef(exception);
  } else {
    throwOrtException(env, ORT_FAIL, "The injected ownership holder allocation did not fail");
  }
}

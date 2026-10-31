#include "JsiMain.h"
#include <ReactCommon/CallInvoker.h>
#include <ReactCommon/CallInvokerHolder.h>
#include <fbjni/detail/Registration.h>
#include <fbjni/fbjni.h>
#include <jni.h>
#include <jsi/jsi.h>
#include <memory>

using namespace facebook;

class OnnxruntimeModule
    : public jni::JavaClass<OnnxruntimeModule> {
 public:
  static constexpr auto kJavaDescriptor =
      "Lai/onnxruntime/reactnative/OnnxruntimeModule;";

  static void registerNatives() {
    javaClassStatic()->registerNatives(
        {makeNativeMethod("nativeInstall",
                          OnnxruntimeModule::nativeInstall),
         makeNativeMethod("nativeCleanup",
                          OnnxruntimeModule::nativeCleanup)});
  }

 private:
  static jlong nativeInstall(
      jni::alias_ref<jni::JObject> thiz, jlong jsContextNativePointer,
      jni::alias_ref<react::CallInvokerHolder::javaobject>
          jsCallInvokerHolder) {
    auto runtime = reinterpret_cast<jsi::Runtime*>(jsContextNativePointer);
    auto jsCallInvoker = jsCallInvokerHolder->cthis()->getCallInvoker();
    auto env = onnxruntimejsi::install(*runtime, jsCallInvoker);
    return reinterpret_cast<jlong>(
        new std::shared_ptr<onnxruntimejsi::Env>(std::move(env)));
  }

  static void nativeCleanup(jni::alias_ref<jni::JObject> thiz,
                            jlong envHandle) {
    auto env = std::unique_ptr<std::shared_ptr<onnxruntimejsi::Env>>(
        reinterpret_cast<std::shared_ptr<onnxruntimejsi::Env>*>(envHandle));
    // invalidate() runs first because this hook is not guaranteed to run on the JS thread: it
    // stops dispatching to a runtime that is going away and releases any thread blocked on it.
    if (env && *env) {
      (*env)->invalidate();
    }
  }
};

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  return jni::initialize(
      vm, [] { OnnxruntimeModule::registerNatives(); });
}

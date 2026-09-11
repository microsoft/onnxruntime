#include "JsiMain.h"
#include <ReactCommon/CallInvoker.h>
#include <ReactCommon/CallInvokerHolder.h>
#include <fbjni/detail/Registration.h>
#include <fbjni/fbjni.h>
#include <jni.h>
#include <jsi/jsi.h>

using namespace facebook;

static std::shared_ptr<onnxruntimejsi::Env> env;

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
  static void nativeInstall(jni::alias_ref<jni::JObject> thiz,
                            jlong jsContextNativePointer,
                            jni::alias_ref<react::CallInvokerHolder::javaobject>
                                jsCallInvokerHolder) {
    auto runtime = reinterpret_cast<jsi::Runtime*>(jsContextNativePointer);
    auto jsCallInvoker = jsCallInvokerHolder->cthis()->getCallInvoker();
    env = onnxruntimejsi::install(*runtime, jsCallInvoker);
  }

  static void nativeCleanup(jni::alias_ref<jni::JObject> thiz) { env.reset(); }
};

JNIEXPORT jint JNI_OnLoad(JavaVM* vm, void*) {
  return jni::initialize(
      vm, [] { OnnxruntimeModule::registerNatives(); });
}

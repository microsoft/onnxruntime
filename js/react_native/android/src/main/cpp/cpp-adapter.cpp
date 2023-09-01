#include <jni.h>
#include <jsi/jsi.h>
#include <string>

using namespace facebook;

typedef u_int8_t byte;

std::string jstring2string(JNIEnv *env, jstring jStr) {
  if (!jStr) return "";

  jclass stringClass = env->GetObjectClass(jStr);
  jmethodID getBytes = env->GetMethodID(stringClass, "getBytes", "(Ljava/lang/String;)[B");
  const auto stringJbytes = (jbyteArray) env->CallObjectMethod(jStr, getBytes, env->NewStringUTF("UTF-8"));

  auto length = (size_t) env->GetArrayLength(stringJbytes);
  jbyte* pBytes = env->GetByteArrayElements(stringJbytes, nullptr);

  std::string ret = std::string((char *)pBytes, length);
  env->ReleaseByteArrayElements(stringJbytes, pBytes, JNI_ABORT);

  env->DeleteLocalRef(stringJbytes);
  env->DeleteLocalRef(stringClass);
  return ret;
}

byte* getBytesFromBlob(JNIEnv *env, jobject instanceGlobal, const std::string& blobId, int offset, int size) {
  if (!env) throw std::runtime_error("JNI Environment is gone!");

  // get java class
  jclass clazz = env->GetObjectClass(instanceGlobal);
  // get method in java class
  jmethodID getBufferJava = env->GetMethodID(clazz, "getBlobBuffer", "(Ljava/lang/String;II)[B");
  // call method
  auto jstring = env->NewStringUTF(blobId.c_str());
  auto boxedBytes = (jbyteArray) env->CallObjectMethod(instanceGlobal,
                                                        getBufferJava,
                                                        // arguments
                                                        jstring,
                                                        offset,
                                                        size);
  env->DeleteLocalRef(jstring);

  jboolean isCopy = true;
  jbyte* bytes = env->GetByteArrayElements(boxedBytes, &isCopy);
  env->DeleteLocalRef(boxedBytes);
  return reinterpret_cast<byte*>(bytes);
};

std::string createBlob(JNIEnv *env, jobject instanceGlobal, byte* bytes, size_t size) {
  if (!env) throw std::runtime_error("JNI Environment is gone!");

  // get java class
  jclass clazz = env->GetObjectClass(instanceGlobal);
  // get method in java class
  jmethodID getBufferJava = env->GetMethodID(clazz, "createBlob", "([B)Ljava/lang/String;");
  // call method
  auto byteArray = env->NewByteArray(size);
  env->SetByteArrayRegion(byteArray, 0, size, reinterpret_cast<jbyte*>(bytes));
  auto blobId = (jstring) env->CallObjectMethod(instanceGlobal, getBufferJava, byteArray);
  env->DeleteLocalRef(byteArray);

  return jstring2string(env, blobId);
};

extern "C"
JNIEXPORT void JNICALL
Java_ai_onnxruntime_reactnative_OnnxruntimeJSIHelper_nativeInstall(JNIEnv *env, jclass _, jlong jsiPtr, jobject instance) {
  auto jsiRuntime = reinterpret_cast<jsi::Runtime*>(jsiPtr);

  auto& runtime = *jsiRuntime;

  auto instanceGlobal = env->NewGlobalRef(instance);

  auto resolveArrayBuffer = jsi::Function::createFromHostFunction(runtime,
                                                                  jsi::PropNameID::forAscii(runtime, "jsiOnnxruntimeResolveArrayBuffer"),
                                                                  1,
                                                                  [=](jsi::Runtime& runtime,
                                                                    const jsi::Value& thisValue,
                                                                    const jsi::Value* arguments,
                                                                    size_t count) -> jsi::Value {
    if (count != 1) {
      throw jsi::JSError(runtime, "jsiOnnxruntimeResolveArrayBuffer(..) expects one argument (object)!");
    }

    jsi::Object data = arguments[0].asObject(runtime);
    auto blobId = data.getProperty(runtime, "blobId").asString(runtime);
    auto offset = data.getProperty(runtime, "offset").asNumber();
    auto size = data.getProperty(runtime, "size").asNumber();

    auto bytes = getBytesFromBlob(env, instanceGlobal, blobId.utf8(runtime), offset, size);

    size_t totalSize = size - offset;
    jsi::Function arrayBufferCtor = runtime.global().getPropertyAsFunction(runtime, "ArrayBuffer");
    jsi::Object o = arrayBufferCtor.callAsConstructor(runtime, (int) totalSize).getObject(runtime);
    jsi::ArrayBuffer buf = o.getArrayBuffer(runtime);
    memcpy(buf.data(runtime), reinterpret_cast<byte*>(bytes), totalSize);

    return buf;
  });
  runtime.global().setProperty(runtime, "jsiOnnxruntimeResolveArrayBuffer", std::move(resolveArrayBuffer));

  auto storeArrayBuffer = jsi::Function::createFromHostFunction(runtime,
                                                                jsi::PropNameID::forAscii(runtime, "jsiOnnxruntimeStoreArrayBuffer"),
                                                                1,
                                                                [=](jsi::Runtime& runtime,
                                                                    const jsi::Value& thisValue,
                                                                    const jsi::Value* arguments,
                                                                    size_t count) -> jsi::Value {
    if (count != 1) {
      throw jsi::JSError(runtime, "jsiOnnxruntimeStoreArrayBuffer(..) expects one argument (object)!");
    }

    auto arrayBuffer = arguments[0].asObject(runtime).getArrayBuffer(runtime);
    auto size = arrayBuffer.size(runtime);

    std::string blobId = createBlob(env, instanceGlobal, arrayBuffer.data(runtime), size);

    jsi::Object result(runtime);
    auto blobIdString = jsi::String::createFromUtf8(runtime, blobId);
    result.setProperty(runtime, "blobId", blobIdString);
    result.setProperty(runtime, "offset", jsi::Value(0));
    result.setProperty(runtime, "size", jsi::Value(static_cast<double>(size)));
    return result;
  });
  runtime.global().setProperty(runtime, "jsiOnnxruntimeStoreArrayBuffer", std::move(storeArrayBuffer));
}

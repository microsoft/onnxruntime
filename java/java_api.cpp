#include "jni_helper.h"

#include <vector>

#include <core/providers/nnapi/nnapi_provider_factory.h>
#include <onnxruntime_cxx_api.h>

#define ORT_JAVA_API_IMPL_BEGIN \
  try {
#define ORT_JAVA_API_IMPL_END                                                                              \
  }                                                                                                        \
  catch (const Ort::Exception& e) {                                                                        \
    auto ort_error_message = e.what();                                                                     \
    auto ort_error_code = e.GetOrtErrorCode();                                                             \
    auto j_ort_exception_cls = env->FindClass("ml/microsoft/onnxruntime/OrtException");                    \
    auto j_ort_error_message = env->NewStringUTF(ort_error_message);                                       \
    auto ctor = env->GetMethodID(j_ort_exception_cls, "<init>", "(Ljava/lang/String;I)V");                 \
    auto j_ort_exception = reinterpret_cast<jthrowable>(                                                   \
        env->NewObject(j_ort_exception_cls, ctor, j_ort_error_message, static_cast<int>(ort_error_code))); \
    env->Throw(j_ort_exception);                                                                           \
  }

#define ORT_JAVA_API_IMPL_END_WITH_RETURN \
  ORT_JAVA_API_IMPL_END                   \
  return 0;

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_Env_initHandle(JNIEnv* env, jobject obj /* this */, jobject j_logging_level, jstring logid) {
  ORT_JAVA_API_IMPL_BEGIN
  OrtEnv* ort_env;
  const auto logging_level_value = javaEnumToCEnum<OrtLoggingLevel>(env, j_logging_level, "ml/microsoft/onnxruntime/LoggingLevel");
  ORT_THROW_ON_ERROR(OrtCreateEnv(logging_level_value,
                                  javaStringtoStdString(env, logid).c_str(), &ort_env));
  setHandle(env, obj, ort_env);
  ORT_JAVA_API_IMPL_END
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_RunOptions_initHandle(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  OrtRunOptions* run_options;
  ORT_THROW_ON_ERROR(OrtCreateRunOptions(&run_options));

  setHandle(env, obj, run_options);
  ORT_JAVA_API_IMPL_END
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_Session_initHandle(JNIEnv* env, jobject obj /* this */,
                                                 jobject j_ort_env, jstring j_model_path,
                                                 const jobject j_options) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* ort_env = getHandle<OrtEnv>(env, j_ort_env);
  const auto* options = getHandle<OrtSessionOptions>(env, j_options);
  OrtSession* session;
  ORT_THROW_ON_ERROR(OrtCreateSession(ort_env,
                                      javaStringtoStdString(env, j_model_path).c_str(), options, &session));
  setHandle(env, obj, session);
  ORT_JAVA_API_IMPL_END
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_initHandle(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  OrtSessionOptions* session_options;
  ORT_THROW_ON_ERROR(OrtCreateSessionOptions(&session_options));

  setHandle(env, obj, session_options);
  ORT_JAVA_API_IMPL_END
}

#define DEFINE_DISPOSE(class_name)                                                                \
  extern "C" JNIEXPORT void JNICALL                                                               \
      Java_ml_microsoft_onnxruntime_##class_name##_dispose(JNIEnv* env, jobject obj /* this */) { \
    auto handle = getHandle<Ort##class_name>(env, obj);                                           \
    OrtRelease##class_name(handle);                                                               \
    handle = nullptr;                                                                             \
  }

DEFINE_DISPOSE(SessionOptions);
DEFINE_DISPOSE(Session);
DEFINE_DISPOSE(RunOptions);
DEFINE_DISPOSE(Env);
DEFINE_DISPOSE(Value);

#undef DEFINE_DISPOSE

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_setThreadPoolSize(JNIEnv* env, jobject obj /* this */,
                                                               jint session_thread_pool_size) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session_options = getHandle<OrtSessionOptions>(env, obj);
  ORT_THROW_ON_ERROR(OrtSetSessionThreadPoolSize(session_options, session_thread_pool_size));
  ORT_JAVA_API_IMPL_END
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_setGraphOptimizationLevel(JNIEnv* env, jobject obj /* this */,
                                                                       jint graph_optimization_level) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session_options = getHandle<OrtSessionOptions>(env, obj);
  ORT_THROW_ON_ERROR(OrtSetSessionGraphOptimizationLevel(session_options, graph_optimization_level));
  ORT_JAVA_API_IMPL_END
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_enableProfiling(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session_options = getHandle<OrtSessionOptions>(env, obj);
  ORT_THROW_ON_ERROR(OrtEnableProfiling(session_options, ""));
  ORT_JAVA_API_IMPL_END
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_disableProfiling(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session_options = getHandle<OrtSessionOptions>(env, obj);
  ORT_THROW_ON_ERROR(OrtDisableProfiling(session_options));
  ORT_JAVA_API_IMPL_END
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_enableCpuMemArena(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session_options = getHandle<OrtSessionOptions>(env, obj);
  ORT_THROW_ON_ERROR(OrtEnableCpuMemArena(session_options));
  ORT_JAVA_API_IMPL_END
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_disableCpuMemArena(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session_options = getHandle<OrtSessionOptions>(env, obj);
  ORT_THROW_ON_ERROR(OrtDisableCpuMemArena(session_options));
  ORT_JAVA_API_IMPL_END
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_SessionOptions_appendNnapiExecutionProvider(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session_options = getHandle<OrtSessionOptions>(env, obj);
  ORT_THROW_ON_ERROR(OrtSessionOptionsAppendExecutionProvider_Nnapi(session_options));
  ORT_JAVA_API_IMPL_END
}

extern "C" JNIEXPORT jobjectArray JNICALL
Java_ml_microsoft_onnxruntime_Session_run(JNIEnv* env, jobject obj /* this */,
                                          jobject j_run_options, jobjectArray j_input_names,
                                          jobjectArray j_input_values, jobjectArray j_output_names) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session = getHandle<OrtSession>(env, obj);
  auto* run_options = getHandle<OrtRunOptions>(env, j_run_options);
  auto input_count = env->GetArrayLength(j_input_values);
  auto output_count = env->GetArrayLength(j_output_names);
  std::vector<const char*> input_names;
  std::vector<OrtValue*> input_values;
  for (int i = 0; i < input_count; i++) {
    auto jstr = env->GetObjectArrayElement(j_input_names, i);
    const char* char_ptr = env->GetStringUTFChars(static_cast<jstring>(jstr), nullptr);
    input_names.push_back(char_ptr);

    auto j_value = env->GetObjectArrayElement(j_input_values, i);
    auto value_ptr = getHandle<OrtValue>(env, j_value);
    input_values.push_back(value_ptr);
  }
  std::vector<const char*> output_names;
  for (int i = 0; i < output_count; i++) {
    auto jstr = env->GetObjectArrayElement(j_output_names, i);
    const char* char_ptr = env->GetStringUTFChars(static_cast<jstring>(jstr), nullptr);
    output_names.push_back(char_ptr);
  }

  std::vector<OrtValue*> output_values(output_count, nullptr);
  ORT_THROW_ON_ERROR(OrtRun(session, run_options, input_names.data(), input_values.data(),
                            input_count, output_names.data(), output_count, output_values.data()));

  const char* class_name = "ml/microsoft/onnxruntime/Value";
  jclass cls = env->FindClass(class_name);
  auto j_value_arr = env->NewObjectArray(output_count, cls, nullptr);
  for (int i = 0; i < output_count; i++) {
    env->SetObjectArrayElement(j_value_arr, i, newObject(env, class_name, output_values[i]));
  }
  return j_value_arr;
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jlong JNICALL
Java_ml_microsoft_onnxruntime_Session_getInputCount(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session = getHandle<OrtSession>(env, obj);
  size_t count;
  ORT_THROW_ON_ERROR(OrtSessionGetInputCount(session, &count));
  return count;
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jlong JNICALL
Java_ml_microsoft_onnxruntime_Session_getOutputCount(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session = getHandle<OrtSession>(env, obj);
  size_t count;
  ORT_THROW_ON_ERROR(OrtSessionGetOutputCount(session, &count));
  return count;
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jstring JNICALL
Java_ml_microsoft_onnxruntime_Session_getInputName(JNIEnv* env, jobject obj /* this */,
                                                   jlong index, jobject j_allocator) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session = getHandle<OrtSession>(env, obj);
  auto* allocator = getHandle<OrtAllocator>(env, j_allocator);
  char* name;
  ORT_THROW_ON_ERROR(OrtSessionGetInputName(session, index, allocator, &name));
  return env->NewStringUTF(name);
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jstring JNICALL
Java_ml_microsoft_onnxruntime_Session_getOutputName(JNIEnv* env, jobject obj /* this */,
                                                    jlong index, jobject j_allocator) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session = getHandle<OrtSession>(env, obj);
  auto* allocator = getHandle<OrtAllocator>(env, j_allocator);
  char* name;
  ORT_THROW_ON_ERROR(OrtSessionGetOutputName(session, index, allocator, &name));
  return env->NewStringUTF(name);
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Session_getInputTypeInfo(JNIEnv* env, jobject obj /* this */,
                                                       jlong index) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session = getHandle<OrtSession>(env, obj);
  OrtTypeInfo* info;
  ORT_THROW_ON_ERROR(OrtSessionGetInputTypeInfo(session, index, &info));
  return newObject(env, "ml/microsoft/onnxruntime/TypeInfo", info);
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Session_getOutputTypeInfo(JNIEnv* env, jobject obj /* this */,
                                                        jlong index) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session = getHandle<OrtSession>(env, obj);
  OrtTypeInfo* info;
  ORT_THROW_ON_ERROR(OrtSessionGetOutputTypeInfo(session, index, &info));
  return newObject(env, "ml/microsoft/onnxruntime/TypeInfo", info);
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_TypeInfo_getTensorTypeAndShapeInfo(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* session = getHandle<OrtTypeInfo>(env, obj);
  const OrtTensorTypeAndShapeInfo* out;
  ORT_THROW_ON_ERROR(OrtCastTypeInfoToTensorInfo(session, &out));
  return newObject(env, "ml/microsoft/onnxruntime/TensorTypeAndShapeInfo",
                   const_cast<OrtTensorTypeAndShapeInfo*>(out));
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Value_createTensor(JNIEnv* env, jobject /* this */,
                                                 jobject j_allocator_info,
                                                 jobject j_data, jlongArray j_shape,
                                                 jobject j_type) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* allocator_info = getHandle<OrtAllocatorInfo>(env, j_allocator_info);
  auto* data_ptr = env->GetDirectBufferAddress(j_data);
  auto data_len = env->GetDirectBufferCapacity(j_data);
  auto* shape_ptr = env->GetLongArrayElements(j_shape, nullptr);
  auto shape_len = env->GetArrayLength(j_shape);
  auto type = javaEnumToCEnum<ONNXTensorElementDataType>(env, j_type, "ml/microsoft/onnxruntime/TensorElementDataType");
  OrtValue* out;
  ORT_THROW_ON_ERROR(OrtCreateTensorWithDataAsOrtValue(allocator_info, data_ptr, data_len,
                                                       shape_ptr, shape_len, type, &out));
  return newObject(env, "ml/microsoft/onnxruntime/Value", out);
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Value_getTensorMutableData(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* value = getHandle<OrtValue>(env, obj);
  uint8_t* out;
  ORT_THROW_ON_ERROR(OrtGetTensorMutableData(value, (void**)&out));
  size_t count;
  OrtTensorTypeAndShapeInfo* info;
  ORT_THROW_ON_ERROR(OrtGetTensorTypeAndShape(value, &info));
  ORT_THROW_ON_ERROR(OrtGetTensorShapeElementCount(info, &count));
  auto byte_buf = env->NewDirectByteBuffer(out, count * 4);
  return byte_buf;
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Value_getTensorTypeAndShapeInfo(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* value = getHandle<OrtValue>(env, obj);
  OrtTensorTypeAndShapeInfo* info;
  ORT_THROW_ON_ERROR(OrtGetTensorTypeAndShape(value, &info));

  return newObject(env, "ml/microsoft/onnxruntime/TensorTypeAndShapeInfo", info);
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jlong JNICALL
Java_ml_microsoft_onnxruntime_TensorTypeAndShapeInfo_getElementCount(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* info = getHandle<OrtTensorTypeAndShapeInfo>(env, obj);

  size_t count;
  ORT_THROW_ON_ERROR(OrtGetTensorShapeElementCount(info, &count));

  return static_cast<jlong>(count);
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_ml_microsoft_onnxruntime_TensorTypeAndShapeInfo_getShape(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* info = getHandle<OrtTensorTypeAndShapeInfo>(env, obj);

  size_t count;
  ORT_THROW_ON_ERROR(OrtGetDimensionsCount(info, &count));
  jlong shape[count];
  auto j_shape = env->NewLongArray(count);
  ORT_THROW_ON_ERROR(OrtGetDimensions(info, shape, count));
  env->SetLongArrayRegion(j_shape, 0, count, shape);

  return j_shape;
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jlong JNICALL
Java_ml_microsoft_onnxruntime_TensorTypeAndShapeInfo_getDimensionsCount(JNIEnv* env, jobject obj /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* info = getHandle<OrtTensorTypeAndShapeInfo>(env, obj);

  size_t count;
  ORT_THROW_ON_ERROR(OrtGetDimensionsCount(info, &count));

  return static_cast<jlong>(count);
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_AllocatorInfo_createCpu(JNIEnv* env, jobject /* this */,
                                                      jobject j_allocator_type, jobject j_mem_type) {
  ORT_JAVA_API_IMPL_BEGIN
  const auto allocator_type_value = javaEnumToCEnum<OrtAllocatorType>(env, j_allocator_type, "ml/microsoft/onnxruntime/AllocatorType");
  const auto mem_type_value = javaEnumToCEnum<OrtMemType>(env, j_mem_type, "ml/microsoft/onnxruntime/MemType");
  OrtAllocatorInfo* allocator_info;

  ORT_THROW_ON_ERROR(OrtCreateCpuAllocatorInfo(allocator_type_value,
                                               mem_type_value, &allocator_info));
  return newObject(env, "ml/microsoft/onnxruntime/AllocatorInfo", allocator_info);
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Allocator_createDefault(JNIEnv* env, jobject /* this */) {
  ORT_JAVA_API_IMPL_BEGIN
  OrtAllocator* allocator;
  ORT_THROW_ON_ERROR(OrtCreateDefaultAllocator(&allocator));
  return newObject(env, "ml/microsoft/onnxruntime/Allocator", allocator);
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Allocator_alloc(JNIEnv* env, jobject obj, jlong size) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* allocator = getHandle<OrtAllocator>(env, obj);
  void* buf;
  ORT_THROW_ON_ERROR(OrtAllocatorAlloc(allocator, size, &buf));
  return env->NewDirectByteBuffer(buf, size);
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

extern "C" JNIEXPORT void JNICALL
Java_ml_microsoft_onnxruntime_Allocator_free(JNIEnv* env, jobject obj, jobject buffer) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* allocator = getHandle<OrtAllocator>(env, obj);
  auto* ptr = env->GetDirectBufferAddress(buffer);
  ORT_THROW_ON_ERROR(OrtAllocatorFree(allocator, ptr));
  ORT_JAVA_API_IMPL_END
}

extern "C" JNIEXPORT jobject JNICALL
Java_ml_microsoft_onnxruntime_Allocator_getInfo(JNIEnv* env, jobject obj) {
  ORT_JAVA_API_IMPL_BEGIN
  auto* allocator = getHandle<OrtAllocator>(env, obj);
  const OrtAllocatorInfo* info;
  ORT_THROW_ON_ERROR(OrtAllocatorGetInfo(allocator, &info));
  return newObject(env, "ml/microsoft/onnxruntime/AllocatorInfo", const_cast<OrtAllocatorInfo*>(info));
  ORT_JAVA_API_IMPL_END_WITH_RETURN
}

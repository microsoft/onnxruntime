// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>

#include <gtest/gtest.h>

#include "OrtJniUtil.h"

namespace {

JNIEnv* env = nullptr;

// Only tensor metadata is synthesized; all Java allocations and exceptions use the real JVM.
struct TestTensor {
  ONNXTensorElementDataType type;
  size_t length;
  void* data = nullptr;
  int releases = 0;
  int data_reads = 0;
};

OrtStatus* ORT_API_CALL GetTensorTypeAndShape(const OrtValue* value, OrtTensorTypeAndShapeInfo** info) noexcept {
  *info = reinterpret_cast<OrtTensorTypeAndShapeInfo*>(const_cast<OrtValue*>(value));
  return nullptr;
}

OrtStatus* ORT_API_CALL GetTensorElementType(const OrtTensorTypeAndShapeInfo* info,
                                             ONNXTensorElementDataType* type) noexcept {
  *type = reinterpret_cast<const TestTensor*>(info)->type;
  return nullptr;
}

OrtStatus* ORT_API_CALL GetTensorShapeElementCount(const OrtTensorTypeAndShapeInfo* info, size_t* length) noexcept {
  *length = reinterpret_cast<const TestTensor*>(info)->length;
  return nullptr;
}

OrtStatus* ORT_API_CALL GetTensorMutableData(OrtValue* value, void** data) noexcept {
  auto* tensor = reinterpret_cast<TestTensor*>(value);
  ++tensor->data_reads;
  *data = tensor->data;
  return nullptr;
}

void ORT_API_CALL ReleaseTensorTypeAndShapeInfo(OrtTensorTypeAndShapeInfo* info) {
  ++reinterpret_cast<TestTensor*>(info)->releases;
}

OrtStatus* ORT_API_CALL GetStringTensorDataLength(const OrtValue*, size_t* length) noexcept {
  *length = 6;
  return nullptr;
}

OrtStatus* ORT_API_CALL GetStringTensorContent(const OrtValue*, void* buffer, size_t,
                                               size_t* offsets, size_t) noexcept {
  std::memcpy(buffer, "onetwo", 6);
  offsets[0] = 0;
  offsets[1] = 3;
  return nullptr;
}

const OrtApi api = [] {
  OrtApi result{};
  result.GetTensorTypeAndShape = GetTensorTypeAndShape;
  result.GetTensorElementType = GetTensorElementType;
  result.GetTensorShapeElementCount = GetTensorShapeElementCount;
  result.GetTensorMutableData = GetTensorMutableData;
  result.ReleaseTensorTypeAndShapeInfo = ReleaseTensorTypeAndShapeInfo;
  result.GetStringTensorDataLength = GetStringTensorDataLength;
  result.GetStringTensorContent = GetStringTensorContent;
  return result;
}();

class JniArrayTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ASSERT_EQ(env->PushLocalFrame(32), JNI_OK);
  }

  void TearDown() override {
    if (env->ExceptionCheck()) {
      env->ExceptionDescribe();
      env->ExceptionClear();
      ADD_FAILURE() << "Unexpected pending Java exception";
    }
    env->PopLocalFrame(nullptr);
  }

  void ExpectException(const char* class_name) {
    jthrowable exception = env->ExceptionOccurred();
    env->ExceptionClear();
    ASSERT_NE(exception, nullptr);
    jclass expected = env->FindClass(class_name);
    ASSERT_NE(expected, nullptr);
    EXPECT_TRUE(env->IsInstanceOf(exception, expected));
    env->DeleteLocalRef(expected);
    env->DeleteLocalRef(exception);
  }
};

TEST_F(JniArrayTest, UnsignedSizeBounds) {
  for (size_t value : {size_t{0}, size_t{1}, size_t{INT32_MAX}}) {
    jsize result = -1;
    EXPECT_TRUE(safecast_size_t_to_jsize(env, value, &result));
    EXPECT_EQ(result, static_cast<jsize>(value));
    EXPECT_FALSE(env->ExceptionCheck());
  }
  for (size_t value : {size_t{INT32_MAX} + 1, size_t{3000000000ULL}, std::numeric_limits<size_t>::max()}) {
    jsize result = 123;
    EXPECT_FALSE(safecast_size_t_to_jsize(env, value, &result));
    EXPECT_EQ(result, 123);
    ExpectException("ai/onnxruntime/OrtException");
  }
  if (sizeof(size_t) > sizeof(jsize)) {
    jsize result = 123;
    EXPECT_FALSE(safecast_size_t_to_jsize(env, static_cast<size_t>(uint64_t{1} << 32), &result));
    EXPECT_EQ(result, 123);
    ExpectException("ai/onnxruntime/OrtException");
  }
}

TEST_F(JniArrayTest, SignedSizeBounds) {
  for (int64_t value : {int64_t{0}, int64_t{1}, int64_t{INT32_MAX}}) {
    jsize result = -1;
    EXPECT_TRUE(safecast_int64_to_jsize(env, value, &result));
    EXPECT_EQ(result, static_cast<jsize>(value));
    EXPECT_FALSE(env->ExceptionCheck());
  }
  for (int64_t value : {int64_t{-1}, int64_t{INT32_MAX} + 1, int64_t{1} << 32,
                        std::numeric_limits<int64_t>::max()}) {
    jsize result = 123;
    EXPECT_FALSE(safecast_int64_to_jsize(env, value, &result));
    EXPECT_EQ(result, 123);
    ExpectException("ai/onnxruntime/OrtException");
  }
}

class TensorArrayTest : public JniArrayTest,
                        public ::testing::WithParamInterface<ONNXTensorElementDataType> {
 protected:
  jarray CreateArray(TestTensor& tensor) {
    auto* value = reinterpret_cast<OrtValue*>(&tensor);
    switch (tensor.type) {
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
        return createLongArrayFromTensor(env, &api, value);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
        return createFloatArrayFromTensor(env, &api, value);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
        return createDoubleArrayFromTensor(env, &api, value);
      case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING:
        return createStringArrayFromTensor(env, &api, value);
      default:
        ADD_FAILURE() << "Unexpected test tensor type";
        return nullptr;
    }
  }
};

TEST_P(TensorArrayTest, OversizedTensorRejectedBeforeReadingData) {
  for (size_t length : {size_t{INT32_MAX} + 1, size_t{3000000000ULL},
                        std::numeric_limits<size_t>::max()}) {
    TestTensor tensor{GetParam(), length};
    EXPECT_EQ(CreateArray(tensor), nullptr);
    EXPECT_EQ(tensor.releases, 1);
    EXPECT_EQ(tensor.data_reads, 0);
    ExpectException("ai/onnxruntime/OrtException");
  }
}

TEST_P(TensorArrayTest, AllocationFailurePreservesOutOfMemoryError) {
  // HotSpot rejects this length before allocating: "Requested array size exceeds VM limit".
  // The small backing buffer must never be copied after New*Array fails.
  std::array<double, 2> data{1, 2};
  TestTensor tensor{GetParam(), size_t{INT32_MAX}, data.data()};
  EXPECT_EQ(CreateArray(tensor), nullptr);
  EXPECT_EQ(tensor.releases, 1);
  ExpectException("java/lang/OutOfMemoryError");
}

TEST_P(TensorArrayTest, SmallTensorConversion) {
  std::array<jlong, 2> longs{1, 2};
  std::array<jfloat, 2> floats{1, 2};
  std::array<jdouble, 2> doubles{1, 2};
  void* data = GetParam() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64   ? static_cast<void*>(longs.data())
               : GetParam() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? static_cast<void*>(floats.data())
                                                                   : static_cast<void*>(doubles.data());
  TestTensor tensor{GetParam(), 2, data};
  jarray result = CreateArray(tensor);
  ASSERT_FALSE(env->ExceptionCheck());
  ASSERT_NE(result, nullptr);
  EXPECT_EQ(env->GetArrayLength(result), 2);
  EXPECT_EQ(tensor.releases, 1);
  switch (GetParam()) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64: {
      std::array<jlong, 2> output{};
      env->GetLongArrayRegion(static_cast<jlongArray>(result), 0, 2, output.data());
      EXPECT_EQ(output, longs);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT: {
      std::array<jfloat, 2> output{};
      env->GetFloatArrayRegion(static_cast<jfloatArray>(result), 0, 2, output.data());
      EXPECT_EQ(output, floats);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE: {
      std::array<jdouble, 2> output{};
      env->GetDoubleArrayRegion(static_cast<jdoubleArray>(result), 0, 2, output.data());
      EXPECT_EQ(output, doubles);
      break;
    }
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING: {
      for (jsize i = 0; i < 2; ++i) {
        auto string = static_cast<jstring>(env->GetObjectArrayElement(static_cast<jobjectArray>(result), i));
        ASSERT_NE(string, nullptr);
        const char* chars = env->GetStringUTFChars(string, nullptr);
        ASSERT_NE(chars, nullptr);
        EXPECT_STREQ(chars, i == 0 ? "one" : "two");
        env->ReleaseStringUTFChars(string, chars);
      }
      break;
    }
    default:
      FAIL() << "Unexpected test tensor type";
  }
}

INSTANTIATE_TEST_SUITE_P(Arrays, TensorArrayTest,
                         ::testing::Values(ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
                                           ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING));

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    std::cerr << "Expected path to the compiled Java classes\n";
    return 1;
  }
  std::string class_path = std::string("-Djava.class.path=") + argv[1];
  char check_jni[] = "-Xcheck:jni";
  char heap_limit[] = "-Xmx64m";
  JavaVMOption options[] = {{class_path.data(), nullptr}, {check_jni, nullptr}, {heap_limit, nullptr}};
  JavaVMInitArgs args{};
  args.version = JNI_VERSION_1_6;
  args.nOptions = 3;
  args.options = options;
  JavaVM* vm = nullptr;
  if (JNI_CreateJavaVM(&vm, reinterpret_cast<void**>(&env), &args) != JNI_OK) {
    std::cerr << "Failed to create the test JVM\n";
    return 1;
  }
  --argc;
  ++argv;
  ::testing::InitGoogleTest(&argc, argv);
  int result = RUN_ALL_TESTS();
  if (vm->DestroyJavaVM() != JNI_OK) {
    return 1;
  }
  return result;
}

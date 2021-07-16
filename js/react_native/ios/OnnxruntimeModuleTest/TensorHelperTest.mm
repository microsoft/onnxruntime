// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#import "TensorHelper.h"

#import <XCTest/XCTest.h>
#import <onnxruntime/onnxruntime_cxx_api.h>

@interface TensorHelperTest : XCTestCase

@end

@implementation TensorHelperTest

template <typename T>
static void testCreateInputTensorT(const std::array<T, 3> &outValues, std::function<NSNumber *(T value)> &convert,
                                   ONNXTensorElementDataType onnxType, NSString *jsTensorType) {
  NSMutableDictionary *inputTensorMap = [NSMutableDictionary dictionary];

  // dims
  NSArray *dims = @[ [NSNumber numberWithLong:outValues.size()] ];
  inputTensorMap[@"dims"] = dims;

  // type
  inputTensorMap[@"type"] = jsTensorType;

  // encoded data
  size_t byteBufferSize = sizeof(T) * outValues.size();
  unsigned char *byteBuffer = static_cast<unsigned char *>(malloc(byteBufferSize));
  NSData *byteBufferRef = [NSData dataWithBytesNoCopy:byteBuffer length:byteBufferSize];
  T *typePtr = (T *)[byteBufferRef bytes];
  for (size_t i = 0; i < outValues.size(); ++i) {
    typePtr[i] = outValues[i];
  }

  NSString *dataEncoded = [byteBufferRef base64EncodedStringWithOptions:0];
  inputTensorMap[@"data"] = dataEncoded;

  Ort::AllocatorWithDefaultOptions ortAllocator;
  std::vector<Ort::MemoryAllocation> allocations;
  Ort::Value inputTensor = [TensorHelper createInputTensor:inputTensorMap
                                              ortAllocator:ortAllocator
                                               allocations:allocations];

  XCTAssertEqual(inputTensor.GetTensorTypeAndShapeInfo().GetElementType(), onnxType);
  XCTAssertTrue(inputTensor.IsTensor());
  XCTAssertEqual(inputTensor.GetTensorTypeAndShapeInfo().GetDimensionsCount(), 1);
  XCTAssertEqual(inputTensor.GetTensorTypeAndShapeInfo().GetShape(),
                 std::vector<int64_t>{static_cast<int64_t>(outValues.size())});
  XCTAssertEqual(inputTensor.GetTensorTypeAndShapeInfo().GetElementCount(), outValues.size());
  const auto tensorData = inputTensor.GetTensorData<T>();
  for (size_t i = 0; i < outValues.size(); ++i) {
    XCTAssertEqual(tensorData[i], outValues[i]);
  }
}

- (void)testCreateInputTensorFloat {
  std::array<float_t, 3> outValues{std::numeric_limits<float_t>::min(), 2.0f, std::numeric_limits<float_t>::max()};
  std::function<NSNumber *(float_t value)> convert = [](float_t value) { return [NSNumber numberWithFloat:value]; };
  testCreateInputTensorT<float_t>(outValues, convert, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, JsTensorTypeFloat);
}

- (void)testCreateInputTensorDouble {
  std::array<double_t, 3> outValues{std::numeric_limits<double_t>::min(), 2.0f, std::numeric_limits<double_t>::max()};
  std::function<NSNumber *(double_t value)> convert = [](double_t value) { return [NSNumber numberWithDouble:value]; };
  testCreateInputTensorT<double_t>(outValues, convert, ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE, JsTensorTypeDouble);
}

- (void)testCreateInputTensorBool {
  std::array<bool, 3> outValues{false, true, true};
  std::function<NSNumber *(bool value)> convert = [](bool value) { return [NSNumber numberWithBool:value]; };
  testCreateInputTensorT<bool>(outValues, convert, ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL, JsTensorTypeBool);
}

- (void)testCreateInputTensorInt8 {
  std::array<int8_t, 3> outValues{std::numeric_limits<int8_t>::min(), 2, std::numeric_limits<int8_t>::max()};
  std::function<NSNumber *(int8_t value)> convert = [](int8_t value) { return [NSNumber numberWithChar:value]; };
  testCreateInputTensorT<int8_t>(outValues, convert, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8, JsTensorTypeByte);
}

- (void)testCreateInputTensorInt16 {
  std::array<int16_t, 3> outValues{std::numeric_limits<int16_t>::min(), 2, std::numeric_limits<int16_t>::max()};
  std::function<NSNumber *(int16_t value)> convert = [](int16_t value) { return [NSNumber numberWithShort:value]; };
  testCreateInputTensorT<int16_t>(outValues, convert, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16, JsTensorTypeShort);
}

- (void)testCreateInputTensorInt32 {
  std::array<int32_t, 3> outValues{std::numeric_limits<int32_t>::min(), 2, std::numeric_limits<int32_t>::max()};
  std::function<NSNumber *(int32_t value)> convert = [](int32_t value) { return [NSNumber numberWithInt:value]; };
  testCreateInputTensorT<int32_t>(outValues, convert, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32, JsTensorTypeInt);
}

- (void)testCreateInputTensorInt64 {
  std::array<int64_t, 3> outValues{std::numeric_limits<int64_t>::min(), 2, std::numeric_limits<int64_t>::max()};
  std::function<NSNumber *(int64_t value)> convert = [](int64_t value) { return [NSNumber numberWithLongLong:value]; };
  testCreateInputTensorT<int64_t>(outValues, convert, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, JsTensorTypeLong);
}

- (void)testCreateInputTensorString {
  std::array<std::string, 3> outValues{"a", "b", "c"};

  NSMutableDictionary *inputTensorMap = [NSMutableDictionary dictionary];

  // dims
  NSArray *dims = @[ [NSNumber numberWithLong:outValues.size()] ];
  inputTensorMap[@"dims"] = dims;

  // type
  inputTensorMap[@"type"] = JsTensorTypeString;

  // data
  NSMutableArray *data = [NSMutableArray array];
  for (auto value : outValues) {
    [data addObject:[NSString stringWithUTF8String:value.c_str()]];
  }
  inputTensorMap[@"data"] = data;

  Ort::AllocatorWithDefaultOptions ortAllocator;
  std::vector<Ort::MemoryAllocation> allocations;
  Ort::Value inputTensor = [TensorHelper createInputTensor:inputTensorMap
                                              ortAllocator:ortAllocator
                                               allocations:allocations];

  XCTAssertEqual(inputTensor.GetTensorTypeAndShapeInfo().GetElementType(), ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING);
  XCTAssertTrue(inputTensor.IsTensor());
  XCTAssertEqual(inputTensor.GetTensorTypeAndShapeInfo().GetDimensionsCount(), 1);
  XCTAssertEqual(inputTensor.GetTensorTypeAndShapeInfo().GetShape(),
                 std::vector<int64_t>{static_cast<int64_t>(outValues.size())});
  XCTAssertEqual(inputTensor.GetTensorTypeAndShapeInfo().GetElementCount(), outValues.size());
  for (int i = 0; i < inputTensor.GetTensorTypeAndShapeInfo().GetElementCount(); ++i) {
    size_t elementLength = inputTensor.GetStringTensorElementLength(i);
    std::string element(elementLength, '\0');
    inputTensor.GetStringTensorElement(elementLength, i, (void *)element.data());
    XCTAssertEqual(element, outValues[i]);
  }
}

template <typename T>
static void testCreateOutputTensorT(const std::array<T, 5> &outValues, std::function<NSNumber *(T value)> &convert,
                                    NSString *jsTensorType, NSString *testDataFileName,
                                    NSString *testDataFileExtension) {
  NSBundle *bundle = [NSBundle bundleForClass:[TensorHelperTest class]];
  NSString *dataPath = [bundle pathForResource:testDataFileName ofType:testDataFileExtension];

  std::unique_ptr<Ort::Env> ortEnv{new Ort::Env(ORT_LOGGING_LEVEL_INFO, "Default")};
  Ort::SessionOptions sessionOptions;
  std::unique_ptr<Ort::Session> session{new Ort::Session(*ortEnv, [dataPath UTF8String], sessionOptions)};

  Ort::AllocatorWithDefaultOptions ortAllocator;
  std::vector<Ort::MemoryAllocation> allocations;

  std::vector<const char *> inputNames;
  inputNames.reserve(session->GetInputCount());
  for (size_t i = 0; i < session->GetInputCount(); ++i) {
    auto inputName = session->GetInputName(i, ortAllocator);
    allocations.emplace_back(ortAllocator, inputName, strlen(inputName) + 1);
    inputNames.emplace_back(inputName);
  }

  std::vector<const char *> outputNames;
  outputNames.reserve(session->GetOutputCount());
  for (size_t i = 0; i < session->GetOutputCount(); ++i) {
    auto outputName = session->GetOutputName(i, ortAllocator);
    allocations.emplace_back(ortAllocator, outputName, strlen(outputName) + 1);
    outputNames.emplace_back(outputName);
  }

  NSMutableDictionary *inputTensorMap = [NSMutableDictionary dictionary];

  // dims
  NSArray *dims = @[ [NSNumber numberWithLong:1], [NSNumber numberWithLong:outValues.size()] ];
  inputTensorMap[@"dims"] = dims;

  // type
  inputTensorMap[@"type"] = jsTensorType;

  // encoded data
  size_t byteBufferSize = sizeof(T) * outValues.size();
  unsigned char *byteBuffer = static_cast<unsigned char *>(malloc(byteBufferSize));
  NSData *byteBufferRef = [NSData dataWithBytesNoCopy:byteBuffer length:byteBufferSize];
  T *typePtr = (T *)[byteBufferRef bytes];
  for (size_t i = 0; i < outValues.size(); ++i) {
    typePtr[i] = outValues[i];
  }

  NSString *dataEncoded = [byteBufferRef base64EncodedStringWithOptions:0];
  inputTensorMap[@"data"] = dataEncoded;

  Ort::Value inputTensor = [TensorHelper createInputTensor:inputTensorMap
                                              ortAllocator:ortAllocator
                                               allocations:allocations];

  std::vector<Ort::Value> feeds;
  feeds.emplace_back(std::move(inputTensor));

  Ort::RunOptions runOptions;
  auto output = session->Run(runOptions, inputNames.data(), feeds.data(), inputNames.size(), outputNames.data(),
                             outputNames.size());

  NSDictionary *resultMap = [TensorHelper createOutputTensor:outputNames values:output];

  XCTAssertTrue([[resultMap objectForKey:@"output"] isEqualToDictionary:inputTensorMap]);
}

- (void)testCreateOutputTensorFloat {
  std::array<float_t, 5> outValues{std::numeric_limits<float_t>::min(), 1.0f, 2.0f, 3.0f,
                                   std::numeric_limits<float_t>::max()};
  std::function<NSNumber *(float_t value)> convert = [](float_t value) { return [NSNumber numberWithFloat:value]; };
  testCreateOutputTensorT<float_t>(outValues, convert, JsTensorTypeFloat, @"test_types_float", @"ort");
}

- (void)testCreateOutputTensorDouble {
  std::array<double_t, 5> outValues{std::numeric_limits<double_t>::min(), 1.0f, 2.0f, 3.0f,
                                    std::numeric_limits<double_t>::max()};
  std::function<NSNumber *(double_t value)> convert = [](double_t value) { return [NSNumber numberWithDouble:value]; };
  testCreateOutputTensorT<double_t>(outValues, convert, JsTensorTypeDouble, @"test_types_double", @"ort");
}

- (void)testCreateOutputTensorBool {
  std::array<bool, 5> outValues{false, true, true, false, true};
  std::function<NSNumber *(bool value)> convert = [](bool value) { return [NSNumber numberWithBool:value]; };
  testCreateOutputTensorT<bool>(outValues, convert, JsTensorTypeBool, @"test_types_bool", @"ort");
}

- (void)testCreateOutputTensorInt8 {
  std::array<int8_t, 5> outValues{std::numeric_limits<int8_t>::min(), 1, -2, 3, std::numeric_limits<int8_t>::max()};
  std::function<NSNumber *(int8_t value)> convert = [](int8_t value) { return [NSNumber numberWithChar:value]; };
  testCreateOutputTensorT<int8_t>(outValues, convert, JsTensorTypeByte, @"test_types_int8", @"ort");
}

- (void)testCreateOutputTensorInt32 {
  std::array<int32_t, 5> outValues{std::numeric_limits<int32_t>::min(), 1, -2, 3, std::numeric_limits<int32_t>::max()};
  std::function<NSNumber *(int32_t value)> convert = [](int32_t value) { return [NSNumber numberWithInt:value]; };
  testCreateOutputTensorT<int32_t>(outValues, convert, JsTensorTypeInt, @"test_types_int32", @"ort");
}

- (void)testCreateOutputTensorInt64 {
  std::array<int64_t, 5> outValues{std::numeric_limits<int64_t>::min(), 1, -2, 3, std::numeric_limits<int64_t>::max()};
  std::function<NSNumber *(int64_t value)> convert = [](int64_t value) { return [NSNumber numberWithLongLong:value]; };
  testCreateOutputTensorT<int64_t>(outValues, convert, JsTensorTypeLong, @"test_types_int64", @"ort");
}

@end

#include "testPch.h"
#include "adapter_test.h"
#include "fileHelpers.h"
#include "winrt/Windows.Storage.h"
#include "winrt/Windows.Storage.Streams.h"

using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Storage::Streams;

static void AdapterTestSetup() {
  ortApi = OrtGetApiBase()->GetApi(2);
  winmlAdapter = OrtGetWinMLAdapter(ortApi);
  
  // for model tests
  std::wstring modulePath = FileHelpers::GetModulePath();
  std::string squeezenetPath = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(modulePath + L"squeezenet_modifiedforruntimestests.onnx");
  std::string metadataPath = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(modulePath + L"modelWith2MetaData.onnx");
  std::string float16Path = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(modulePath + L"starry-night-fp16.onnx");
  winmlAdapter->CreateModelFromPath(squeezenetPath.c_str(), squeezenetPath.size(), &squeezenetModel);
  winmlAdapter->CreateModelFromPath(metadataPath.c_str(), metadataPath.size(), &metadataModel);
  winmlAdapter->CreateModelFromPath(float16Path.c_str(), float16Path.size(), &float16Model);
}

static void AdapterTestTeardown() {
  winmlAdapter->ReleaseModel(squeezenetModel);
  winmlAdapter->ReleaseModel(metadataModel);
  winmlAdapter->ReleaseModel(float16Model);
}

static void CreateModelFromPath() {
  WINML_EXPECT_TRUE(squeezenetModel != nullptr);
  WINML_EXPECT_TRUE(metadataModel != nullptr);
  WINML_EXPECT_TRUE(float16Model != nullptr);
}

static void CreateModelFromData() {
  StorageFolder folder = StorageFolder::GetFolderFromPathAsync(FileHelpers::GetModulePath()).get();
  StorageFile file = folder.GetFileAsync(L"squeezenet_modifiedforruntimestests.onnx").get();
  IRandomAccessStream stream = file.OpenAsync(FileAccessMode::Read).get();
  DataReader dataReader(stream.GetInputStreamAt(0));
  dataReader.LoadAsync(static_cast<uint32_t>(stream.Size())).get();
  IBuffer dataBuffer = dataReader.DetachBuffer();
  OrtModel* squeezenetModelFromData = nullptr;
  winmlAdapter->CreateModelFromData(dataBuffer.data(), dataBuffer.Length(), &squeezenetModelFromData);
  WINML_EXPECT_TRUE(squeezenetModelFromData != nullptr);
  // Verify a function in the model for thoroughness
  const char* author;
  size_t len;
  winmlAdapter->ModelGetAuthor(squeezenetModelFromData, &author, &len);
  std::string authorStr(author);
  WINML_EXPECT_EQUAL(authorStr, "onnx-caffe2");
  winmlAdapter->ReleaseModel(squeezenetModelFromData);
}

static void CloneModel() {
  OrtModel* squeezenetClone = nullptr;
  winmlAdapter->CloneModel(squeezenetModel, &squeezenetClone);
  WINML_EXPECT_TRUE(squeezenetClone != nullptr);
  // Verify a function in clone
  const char* author;
  size_t len;
  winmlAdapter->ModelGetAuthor(squeezenetClone, &author, &len);
  std::string authorStr(author);
  WINML_EXPECT_EQUAL(authorStr, "onnx-caffe2");
}

static void ModelGetAuthor() {
  const char* author;
  size_t len;
  winmlAdapter->ModelGetAuthor(squeezenetModel, &author, &len);
  std::string authorStr(author);
  WINML_EXPECT_EQUAL(authorStr, "onnx-caffe2");
}

static void ModelGetName() {
  const char* name;
  size_t len;
  winmlAdapter->ModelGetName(squeezenetModel, &name, &len);
  std::string nameStr(name);
  WINML_EXPECT_EQUAL(nameStr, "squeezenet_old");
}

static void ModelGetDomain() {
  const char* domain;
  size_t len;
  winmlAdapter->ModelGetDomain(squeezenetModel, &domain, &len);
  std::string domainStr(domain);
  WINML_EXPECT_EQUAL(domainStr, "test-domain");
}

static void ModelGetDescription() {
  const char* description;
  size_t len;
  winmlAdapter->ModelGetDescription(squeezenetModel, &description, &len);
  std::string descriptionStr(description);
  WINML_EXPECT_EQUAL(descriptionStr, "test-doc_string");
}

static void ModelGetVersion() {
  int64_t version;
  winmlAdapter->ModelGetVersion(squeezenetModel, &version);
  WINML_EXPECT_EQUAL(version, 123456);
}

static void ModelGetInputCount() {
  size_t inputCount;
  winmlAdapter->ModelGetInputCount(squeezenetModel, &inputCount);
  WINML_EXPECT_EQUAL(inputCount, 1);
}

static void ModelGetOutputCount() {
  size_t outputCount;
  winmlAdapter->ModelGetOutputCount(squeezenetModel, &outputCount);
  WINML_EXPECT_EQUAL(outputCount, 1);
}

static void ModelGetInputName() {
  const char* inputName;
  size_t count;
  winmlAdapter->ModelGetInputName(squeezenetModel, 0, &inputName, &count);
  std::string inputNameStr(inputName);
  WINML_EXPECT_EQUAL(inputNameStr, "data_0");
}

static void ModelGetOutputName() {
  const char* outputName;
  size_t count;
  winmlAdapter->ModelGetOutputName(squeezenetModel, 0, &outputName, &count);
  std::string outputNameStr(outputName);
  WINML_EXPECT_EQUAL(outputNameStr, "softmaxout_1");
}

static void ModelGetInputDescription() {
  const char* inputDescription;
  size_t count;
  winmlAdapter->ModelGetInputDescription(metadataModel, 0, &inputDescription, &count);
  std::string inputDescriptionStr(inputDescription);
  WINML_EXPECT_EQUAL(inputDescriptionStr, "this is a long input description!");
}

static void ModelGetOutputDescription() {
  const char* outputDescription;
  size_t count;
  winmlAdapter->ModelGetOutputDescription(metadataModel, 0, &outputDescription, &count);
  std::string outputDescriptionStr(outputDescription);
  WINML_EXPECT_EQUAL(outputDescriptionStr, "this is a long output description!");
}

static void ModelGetInputTypeInfo() {
  OrtTypeInfo* inputTypeInfo;
  winmlAdapter->ModelGetInputTypeInfo(squeezenetModel, 0, &inputTypeInfo);

  ONNXType inputType;
  ortApi->GetOnnxTypeFromTypeInfo(inputTypeInfo, &inputType);
  WINML_EXPECT_EQUAL(inputType, ONNX_TYPE_TENSOR);

  const OrtTensorTypeAndShapeInfo* tensorInfo;
  ortApi->CastTypeInfoToTensorInfo(inputTypeInfo, &tensorInfo);

  ONNXTensorElementDataType tensorType;
  ortApi->GetTensorElementType(tensorInfo, &tensorType);
  WINML_EXPECT_EQUAL(tensorType, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  size_t dimCount;
  ortApi->GetDimensionsCount(tensorInfo, &dimCount);
  WINML_EXPECT_EQUAL(dimCount, 4);

  int64_t dimValues[4]; 
  ortApi->GetDimensions(tensorInfo, dimValues, 4);
  WINML_EXPECT_EQUAL(dimValues[0], 1);
  WINML_EXPECT_EQUAL(dimValues[1], 3);
  WINML_EXPECT_EQUAL(dimValues[2], 224);
  WINML_EXPECT_EQUAL(dimValues[3], 224);

  ortApi->ReleaseTypeInfo(inputTypeInfo);
}

static void ModelGetOutputTypeInfo() {
  OrtTypeInfo* outputTypeInfo;
  winmlAdapter->ModelGetOutputTypeInfo(squeezenetModel, 0, &outputTypeInfo);

  ONNXType outputType;
  ortApi->GetOnnxTypeFromTypeInfo(outputTypeInfo, &outputType);
  WINML_EXPECT_EQUAL(outputType, ONNX_TYPE_TENSOR);

  const OrtTensorTypeAndShapeInfo* tensorInfo;
  ortApi->CastTypeInfoToTensorInfo(outputTypeInfo, &tensorInfo);

  ONNXTensorElementDataType tensorType;
  ortApi->GetTensorElementType(tensorInfo, &tensorType);
  WINML_EXPECT_EQUAL(tensorType, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  size_t dimCount;
  ortApi->GetDimensionsCount(tensorInfo, &dimCount);
  WINML_EXPECT_EQUAL(dimCount, 4);

  int64_t dimValues[4];
  ortApi->GetDimensions(tensorInfo, dimValues, 4);
  WINML_EXPECT_EQUAL(dimValues[0], 1);
  WINML_EXPECT_EQUAL(dimValues[1], 1000);
  WINML_EXPECT_EQUAL(dimValues[2], 1);
  WINML_EXPECT_EQUAL(dimValues[3], 1);

  ortApi->ReleaseTypeInfo(outputTypeInfo);
}

static void ModelGetMetadataCount() {
  size_t metadataCount;
  winmlAdapter->ModelGetMetadataCount(metadataModel, &metadataCount);
  WINML_EXPECT_EQUAL(metadataCount, 2);
}

static void ModelGetMetadata() {
  const char* metadataKey;
  size_t metadataKeyLen;
  const char* metadataValue;
  size_t metadataValueLen;

  winmlAdapter->ModelGetMetadata(metadataModel, 0, &metadataKey, &metadataKeyLen, &metadataValue, &metadataValueLen);
  WINML_EXPECT_EQUAL(std::string(metadataKey), "thisisalongkey");
  WINML_EXPECT_EQUAL(metadataKeyLen, 14);
  WINML_EXPECT_EQUAL(std::string(metadataValue), "thisisalongvalue");
  WINML_EXPECT_EQUAL(metadataValueLen, 16);

  winmlAdapter->ModelGetMetadata(metadataModel, 1, &metadataKey, &metadataKeyLen, &metadataValue, &metadataValueLen);
  WINML_EXPECT_EQUAL(std::string(metadataKey), "key2");
  WINML_EXPECT_EQUAL(metadataKeyLen, 4);
  WINML_EXPECT_EQUAL(std::string(metadataValue), "val2");
  WINML_EXPECT_EQUAL(metadataValueLen, 4);
}

static void ModelEnsureNoFloat16() {
  OrtStatus* float16ErrorStatus;

  float16ErrorStatus = winmlAdapter->ModelEnsureNoFloat16(squeezenetModel);
  WINML_EXPECT_EQUAL(float16ErrorStatus, nullptr);

  float16ErrorStatus = winmlAdapter->ModelEnsureNoFloat16(float16Model);
  WINML_EXPECT_NOT_EQUAL(float16ErrorStatus, nullptr);
  WINML_EXPECT_EQUAL(ortApi->GetErrorCode(float16ErrorStatus), ORT_INVALID_GRAPH);
}

static void __stdcall TestLoggingCallback(void* param, OrtLoggingLevel severity, const char* category,
                                          const char* logger_id, const char* code_location, const char* message) noexcept {
  UNREFERENCED_PARAMETER(param);
  UNREFERENCED_PARAMETER(severity);
  UNREFERENCED_PARAMETER(category);
  UNREFERENCED_PARAMETER(logger_id);
  UNREFERENCED_PARAMETER(code_location);
  UNREFERENCED_PARAMETER(message);
  loggingFunctionCalled = true;
}

static void __stdcall TestProfileEventCallback(const OrtProfilerEventRecord* profiler_record) noexcept {
  UNREFERENCED_PARAMETER(profiler_record);
  profilingFunctionCalled = true;
}

static void EnvConfigureCustomLoggerAndProfiler() {
  OrtEnv* ortEnv = nullptr;
  ortApi->CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ortEnv);
  winmlAdapter->EnvConfigureCustomLoggerAndProfiler(ortEnv,
                                                    &TestLoggingCallback, &TestProfileEventCallback, nullptr,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ortEnv);
  loggingFunctionCalled = false;
  OrtSession* ortSession = nullptr;
  std::wstring squeezenetPath = FileHelpers::GetModulePath() + L"relu.onnx";
  ortApi->CreateSession(ortEnv, squeezenetPath.c_str(), nullptr, &ortSession);
  WINML_EXPECT_TRUE(loggingFunctionCalled);

  size_t inputTensorSize = 5;
  int64_t inputDimensions[] = {5};

  std::vector<float> inputTensorValues(inputTensorSize);
  std::vector<const char*> inputNodeNames = {"X"};
  std::vector<const char*> outputNodeNames = {"Y"};

  // initialize input data with values in [0.0, 1.0]
  for (size_t i = 0; i < inputTensorSize; i++)
    inputTensorValues[i] = (float)i / (inputTensorSize + 1);

  // create input tensor object from data values
  OrtMemoryInfo* memoryInfo;
  ortApi->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memoryInfo);
  OrtValue* inputTensor = nullptr;
  ortApi->CreateTensorWithDataAsOrtValue(memoryInfo, inputTensorValues.data(), inputTensorSize * sizeof(float), inputDimensions, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &inputTensor);
  int isTensor;
  ortApi->IsTensor(inputTensor, &isTensor);
  assert(isTensor);
  ortApi->ReleaseMemoryInfo(memoryInfo);
  OrtValue* outputTensor = nullptr;
  winmlAdapter->SessionStartProfiling(ortEnv, ortSession);
  profilingFunctionCalled = false;
  ortApi->Run(ortSession, nullptr, inputNodeNames.data(), (const OrtValue* const*)&inputTensor, 1, outputNodeNames.data(), 1, &outputTensor);
  WINML_EXPECT_TRUE(profilingFunctionCalled);
  winmlAdapter->SessionEndProfiling(ortSession);

  ortApi->ReleaseValue(outputTensor);
  ortApi->ReleaseValue(inputTensor);
  ortApi->ReleaseSession(ortSession);
  ortApi->ReleaseEnv(ortEnv);
}

const AdapterTestApi& getapi() {
  static constexpr AdapterTestApi api =
      {
          AdapterTestSetup,
          AdapterTestTeardown,
          CreateModelFromPath,
          CreateModelFromData,
          CloneModel,
          ModelGetAuthor,
          ModelGetName,
          ModelGetDomain,
          ModelGetDescription,
          ModelGetVersion,
          ModelGetInputCount,
          ModelGetOutputCount,
          ModelGetInputName,
          ModelGetOutputName,
          ModelGetInputDescription,
          ModelGetOutputDescription,
          ModelGetInputTypeInfo,
          ModelGetOutputTypeInfo,
          ModelGetMetadataCount,
          ModelGetMetadata,
          ModelEnsureNoFloat16,
          EnvConfigureCustomLoggerAndProfiler,
          };
  return api;
}
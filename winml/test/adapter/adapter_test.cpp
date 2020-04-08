#include "testPch.h"
#include "adapter_test.h"
#include "fileHelpers.h"
#include "../../winml/adapter/winml_adapter_model.h"
#include "core/providers/winml/winml_provider_factory.h"
#include "core\framework\onnxruntime_typeinfo.h"
#include "core\framework\tensor_shape.h"
#include "core\framework\tensor_type_and_shape.h"
#include "winrt/Windows.Storage.h"
#include "winrt/Windows.Storage.Streams.h"
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Storage::Streams;

static void AdapterModelTestSetup() {
  ortApi = OrtGetApiBase()->GetApi(2);
  winmlAdapter = OrtGetWinMLAdapter(ortApi);
  std::wstring modulePath = FileHelpers::GetModulePath();
  std::string squeezenetPath = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(modulePath + L"squeezenet_modifiedforruntimestests.onnx");
  std::string metadataPath = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(modulePath + L"modelWith2MetaData.onnx");
  std::string float16Path = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(modulePath + L"starry-night-fp16.onnx");

  winmlAdapter->CreateModelFromPath(squeezenetPath.c_str(), squeezenetPath.size(), &squeezenetModel);
  winmlAdapter->CreateModelFromPath(metadataPath.c_str(), metadataPath.size(), &metadataModel);
  winmlAdapter->CreateModelFromPath(float16Path.c_str(), float16Path.size(), &float16Model);
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
  // Verify a function in model for thoroughness
  const char* author;
  size_t len;
  winmlAdapter->ModelGetAuthor(squeezenetModelFromData, &author, &len);
  std::string authorStr(author);
  WINML_EXPECT_EQUAL(authorStr, "onnx-caffe2");
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
  WINML_EXPECT_EQUAL(inputTypeInfo->type, ONNX_TYPE_TENSOR);
  WINML_EXPECT_EQUAL(inputTypeInfo->map_type_info, nullptr);
  WINML_EXPECT_EQUAL(inputTypeInfo->sequence_type_info, nullptr);
  WINML_EXPECT_EQUAL(inputTypeInfo->data->type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  WINML_EXPECT_EQUAL(inputTypeInfo->data->shape[0], 1);
  WINML_EXPECT_EQUAL(inputTypeInfo->data->shape[1], 3);
  WINML_EXPECT_EQUAL(inputTypeInfo->data->shape[2], 224);
  WINML_EXPECT_EQUAL(inputTypeInfo->data->shape[3], 224);
  WINML_EXPECT_EQUAL(inputTypeInfo->data->dim_params[0], "");
  WINML_EXPECT_EQUAL(inputTypeInfo->data->dim_params[1], "");
  WINML_EXPECT_EQUAL(inputTypeInfo->data->dim_params[2], "");
  WINML_EXPECT_EQUAL(inputTypeInfo->data->dim_params[3], "");
}

static void ModelGetOutputTypeInfo() {
  OrtTypeInfo* outputTypeInfo;
  winmlAdapter->ModelGetOutputTypeInfo(squeezenetModel, 0, &outputTypeInfo);
  WINML_EXPECT_EQUAL(outputTypeInfo->type, ONNX_TYPE_TENSOR);
  WINML_EXPECT_EQUAL(outputTypeInfo->map_type_info, nullptr);
  WINML_EXPECT_EQUAL(outputTypeInfo->sequence_type_info, nullptr);
  WINML_EXPECT_EQUAL(outputTypeInfo->data->type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);
  WINML_EXPECT_EQUAL(outputTypeInfo->data->shape[0], 1);
  WINML_EXPECT_EQUAL(outputTypeInfo->data->shape[1], 1000);
  WINML_EXPECT_EQUAL(outputTypeInfo->data->shape[2], 1);
  WINML_EXPECT_EQUAL(outputTypeInfo->data->shape[3], 1);
  WINML_EXPECT_EQUAL(outputTypeInfo->data->dim_params[0], "");
  WINML_EXPECT_EQUAL(outputTypeInfo->data->dim_params[1], "");
  WINML_EXPECT_EQUAL(outputTypeInfo->data->dim_params[2], "");
  WINML_EXPECT_EQUAL(outputTypeInfo->data->dim_params[3], "");
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
  loggingFunctionCalled = true;
}

static void __stdcall WinmlOrtProfileEventCallback(const OrtProfilerEventRecord* profiler_record) noexcept {
  profilingFunctionCalled = true;
}

static void EnvConfigureCustomLoggerAndProfiler() {
  OrtEnv* ortEnv = nullptr;
  ortApi->CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ortEnv);
  winmlAdapter->EnvConfigureCustomLoggerAndProfiler(ortEnv,
                                                    &TestLoggingCallback, &WinmlOrtProfileEventCallback, nullptr,
                                                    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ortEnv);
  WINML_EXPECT_FALSE(loggingFunctionCalled);
  OrtSession* ortSession = nullptr;
  std::wstring squeezenetPath = FileHelpers::GetModulePath() + L"relu.onnx";
  ortApi->CreateSession(ortEnv, squeezenetPath.c_str(), nullptr, &ortSession);
  WINML_EXPECT_TRUE(loggingFunctionCalled);

  size_t input_tensor_size = 5;
  int64_t inputDimensions[] = {5};

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> input_node_names = {"X"};
  std::vector<const char*> output_node_names = {"Y"};

  // initialize input data with values in [0.0, 1.0]
  for (size_t i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  OrtMemoryInfo* memory_info;
  ortApi->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
  OrtValue* input_tensor = nullptr;
  ortApi->CreateTensorWithDataAsOrtValue(memory_info, input_tensor_values.data(), input_tensor_size * sizeof(float), inputDimensions, 1, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
  int is_tensor;
  ortApi->IsTensor(input_tensor, &is_tensor);
  assert(is_tensor);
  ortApi->ReleaseMemoryInfo(memory_info);
  OrtValue* output_tensor = nullptr;
  winmlAdapter->SessionStartProfiling(ortEnv, ortSession);
  WINML_EXPECT_FALSE(profilingFunctionCalled);
  ortApi->Run(ortSession, nullptr, input_node_names.data(), (const OrtValue* const*)&input_tensor, 1, output_node_names.data(), 1, &output_tensor);
  WINML_EXPECT_TRUE(profilingFunctionCalled);
  winmlAdapter->SessionEndProfiling(ortSession);
}

const AdapterTestApi& getapi() {
  static constexpr AdapterTestApi api =
      {
          AdapterModelTestSetup,
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
          EnvConfigureCustomLoggerAndProfiler};
  return api;
}
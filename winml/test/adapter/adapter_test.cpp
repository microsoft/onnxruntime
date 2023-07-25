#include "testPch.h"
#include "adapter_test.h"
#include "fileHelpers.h"
#include "winrt/Windows.Storage.h"
#include "winrt/Windows.Storage.Streams.h"

using namespace ws;
using namespace wss;

static void AdapterTestSetup() {
#ifdef BUILD_INBOX
  winrt_activation_handler = WINRT_RoGetActivationFactory;
#endif
  ort_api = OrtGetApiBase()->GetApi(ORT_API_VERSION);
  winml_adapter_api = OrtGetWinMLAdapter(ORT_API_VERSION);

    // for model tests
  std::wstring module_path = FileHelpers::GetModulePath();
  std::string squeezenet_path = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(
    module_path + L"squeezenet_modifiedforruntimestests.onnx"
  );
  std::string metadata_path =
    std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(module_path + L"modelWith2MetaData.onnx");
  std::string float16_path =
    std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(module_path + L"starry-night-fp16.onnx");
  winml_adapter_api->CreateModelFromPath(squeezenet_path.c_str(), squeezenet_path.size(), &squeezenet_model);
  winml_adapter_api->CreateModelFromPath(metadata_path.c_str(), metadata_path.size(), &metadata_model);
  winml_adapter_api->CreateModelFromPath(float16_path.c_str(), float16_path.size(), &float16_Model);
}

static void AdapterTestTeardown() {
  winml_adapter_api->ReleaseModel(squeezenet_model);
  winml_adapter_api->ReleaseModel(metadata_model);
  winml_adapter_api->ReleaseModel(float16_Model);
}

static void CreateModelFromPath() {
  WINML_EXPECT_TRUE(squeezenet_model != nullptr);
  WINML_EXPECT_TRUE(metadata_model != nullptr);
  WINML_EXPECT_TRUE(float16_Model != nullptr);
}

static void CreateModelFromData() {
  StorageFolder folder = StorageFolder::GetFolderFromPathAsync(FileHelpers::GetModulePath()).get();
  StorageFile file = folder.GetFileAsync(L"squeezenet_modifiedforruntimestests.onnx").get();
  IRandomAccessStream stream = file.OpenAsync(FileAccessMode::Read).get();
  DataReader data_reader(stream.GetInputStreamAt(0));
  data_reader.LoadAsync(static_cast<uint32_t>(stream.Size())).get();
  IBuffer data_buffer = data_reader.DetachBuffer();
  OrtModel* squeezenet_model_from_data = nullptr;
  winml_adapter_api->CreateModelFromData(data_buffer.data(), data_buffer.Length(), &squeezenet_model_from_data);
  WINML_EXPECT_TRUE(squeezenet_model_from_data != nullptr);
  // Verify a function in the model for thoroughness
  const char* author;
  size_t len;
  winml_adapter_api->ModelGetAuthor(squeezenet_model_from_data, &author, &len);
  std::string author_str(author);
  WINML_EXPECT_EQUAL(author_str, "onnx-caffe2");
  winml_adapter_api->ReleaseModel(squeezenet_model_from_data);
}

static void CloneModel() {
  OrtModel* squeezenet_clone = nullptr;
  winml_adapter_api->CloneModel(squeezenet_model, &squeezenet_clone);
  WINML_EXPECT_TRUE(squeezenet_clone != nullptr);
  // Verify a function in clone
  const char* author;
  size_t len;
  winml_adapter_api->ModelGetAuthor(squeezenet_clone, &author, &len);
  std::string author_str(author);
  WINML_EXPECT_EQUAL(author_str, "onnx-caffe2");
}

static void ModelGetAuthor() {
  const char* author;
  size_t len;
  winml_adapter_api->ModelGetAuthor(squeezenet_model, &author, &len);
  std::string author_str(author);
  WINML_EXPECT_EQUAL(author_str, "onnx-caffe2");
}

static void ModelGetName() {
  const char* name;
  size_t len;
  winml_adapter_api->ModelGetName(squeezenet_model, &name, &len);
  std::string name_str(name);
  WINML_EXPECT_EQUAL(name_str, "squeezenet_old");
}

static void ModelGetDomain() {
  const char* domain;
  size_t len;
  winml_adapter_api->ModelGetDomain(squeezenet_model, &domain, &len);
  std::string domain_str(domain);
  WINML_EXPECT_EQUAL(domain_str, "test-domain");
}

static void ModelGetDescription() {
  const char* description;
  size_t len;
  winml_adapter_api->ModelGetDescription(squeezenet_model, &description, &len);
  std::string description_str(description);
  WINML_EXPECT_EQUAL(description_str, "test-doc_string");
}

static void ModelGetVersion() {
  int64_t version;
  winml_adapter_api->ModelGetVersion(squeezenet_model, &version);
  WINML_EXPECT_EQUAL(version, 123456);
}

static void ModelGetInputCount() {
  size_t input_count;
  winml_adapter_api->ModelGetInputCount(squeezenet_model, &input_count);
  WINML_EXPECT_EQUAL(input_count, 1u);
}

static void ModelGetOutputCount() {
  size_t output_count;
  winml_adapter_api->ModelGetOutputCount(squeezenet_model, &output_count);
  WINML_EXPECT_EQUAL(output_count, 1u);
}

static void ModelGetInputName() {
  const char* input_name;
  size_t count;
  winml_adapter_api->ModelGetInputName(squeezenet_model, 0, &input_name, &count);
  std::string input_name_str(input_name);
  WINML_EXPECT_EQUAL(input_name_str, "data_0");
}

static void ModelGetOutputName() {
  const char* output_name;
  size_t count;
  winml_adapter_api->ModelGetOutputName(squeezenet_model, 0, &output_name, &count);
  std::string output_name_str(output_name);
  WINML_EXPECT_EQUAL(output_name_str, "softmaxout_1");
}

static void ModelGetInputDescription() {
  const char* input_description;
  size_t count;
  winml_adapter_api->ModelGetInputDescription(metadata_model, 0, &input_description, &count);
  std::string input_description_str(input_description);
  WINML_EXPECT_EQUAL(input_description_str, "this is a long input description!");
}

static void ModelGetOutputDescription() {
  const char* output_description;
  size_t count;
  winml_adapter_api->ModelGetOutputDescription(metadata_model, 0, &output_description, &count);
  std::string output_description_str(output_description);
  WINML_EXPECT_EQUAL(output_description_str, "this is a long output description!");
}

static void ModelGetInputTypeInfo() {
  OrtTypeInfo* input_type_info;
  winml_adapter_api->ModelGetInputTypeInfo(squeezenet_model, 0, &input_type_info);

  ONNXType input_type;
  ort_api->GetOnnxTypeFromTypeInfo(input_type_info, &input_type);
  WINML_EXPECT_EQUAL(input_type, ONNX_TYPE_TENSOR);

  const OrtTensorTypeAndShapeInfo* tensor_info;
  ort_api->CastTypeInfoToTensorInfo(input_type_info, &tensor_info);

  ONNXTensorElementDataType tensor_type;
  ort_api->GetTensorElementType(tensor_info, &tensor_type);
  WINML_EXPECT_EQUAL(tensor_type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  size_t dim_count;
  ort_api->GetDimensionsCount(tensor_info, &dim_count);
  WINML_EXPECT_EQUAL(dim_count, 4u);

  int64_t dim_values[4];
  ort_api->GetDimensions(tensor_info, dim_values, 4);
  WINML_EXPECT_EQUAL(dim_values[0], 1);
  WINML_EXPECT_EQUAL(dim_values[1], 3);
  WINML_EXPECT_EQUAL(dim_values[2], 224);
  WINML_EXPECT_EQUAL(dim_values[3], 224);

  ort_api->ReleaseTypeInfo(input_type_info);
}

static void ModelGetOutputTypeInfo() {
  OrtTypeInfo* output_type_info;
  winml_adapter_api->ModelGetOutputTypeInfo(squeezenet_model, 0, &output_type_info);

  ONNXType output_type;
  ort_api->GetOnnxTypeFromTypeInfo(output_type_info, &output_type);
  WINML_EXPECT_EQUAL(output_type, ONNX_TYPE_TENSOR);

  const OrtTensorTypeAndShapeInfo* tensor_info;
  ort_api->CastTypeInfoToTensorInfo(output_type_info, &tensor_info);

  ONNXTensorElementDataType tensor_type;
  ort_api->GetTensorElementType(tensor_info, &tensor_type);
  WINML_EXPECT_EQUAL(tensor_type, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT);

  size_t dim_count;
  ort_api->GetDimensionsCount(tensor_info, &dim_count);
  WINML_EXPECT_EQUAL(dim_count, 4u);

  int64_t dim_values[4];
  ort_api->GetDimensions(tensor_info, dim_values, 4);
  WINML_EXPECT_EQUAL(dim_values[0], 1);
  WINML_EXPECT_EQUAL(dim_values[1], 1000);
  WINML_EXPECT_EQUAL(dim_values[2], 1);
  WINML_EXPECT_EQUAL(dim_values[3], 1);

  ort_api->ReleaseTypeInfo(output_type_info);
}

static void ModelGetMetadataCount() {
  size_t metadata_count;
  winml_adapter_api->ModelGetMetadataCount(metadata_model, &metadata_count);
  WINML_EXPECT_EQUAL(metadata_count, 2u);
}

static void ModelGetMetadata() {
  const char* metadata_key;
  size_t metadata_key_len;
  const char* metadata_value;
  size_t metadata_value_len;

  winml_adapter_api->ModelGetMetadata(
    metadata_model, 0, &metadata_key, &metadata_key_len, &metadata_value, &metadata_value_len
  );
  WINML_EXPECT_EQUAL(std::string(metadata_key), "thisisalongkey");
  WINML_EXPECT_EQUAL(metadata_key_len, 14u);
  WINML_EXPECT_EQUAL(std::string(metadata_value), "thisisalongvalue");
  WINML_EXPECT_EQUAL(metadata_value_len, 16u);

  winml_adapter_api->ModelGetMetadata(
    metadata_model, 1, &metadata_key, &metadata_key_len, &metadata_value, &metadata_value_len
  );
  WINML_EXPECT_EQUAL(std::string(metadata_key), "key2");
  WINML_EXPECT_EQUAL(metadata_key_len, 4u);
  WINML_EXPECT_EQUAL(std::string(metadata_value), "val2");
  WINML_EXPECT_EQUAL(metadata_value_len, 4u);
}

static void ModelEnsureNoFloat16() {
  OrtStatus* float16_error_status;

  float16_error_status = winml_adapter_api->ModelEnsureNoFloat16(squeezenet_model);
  WINML_EXPECT_EQUAL(float16_error_status, nullptr);

  float16_error_status = winml_adapter_api->ModelEnsureNoFloat16(float16_Model);
  WINML_EXPECT_NOT_EQUAL(float16_error_status, nullptr);
  WINML_EXPECT_EQUAL(ort_api->GetErrorCode(float16_error_status), ORT_INVALID_GRAPH);
}

static void __stdcall TestLoggingCallback(
  void* param,
  OrtLoggingLevel severity,
  const char* category,
  const char* logger_id,
  const char* code_location,
  const char* message
) noexcept {
  UNREFERENCED_PARAMETER(param);
  UNREFERENCED_PARAMETER(severity);
  UNREFERENCED_PARAMETER(category);
  UNREFERENCED_PARAMETER(logger_id);
  UNREFERENCED_PARAMETER(code_location);
  UNREFERENCED_PARAMETER(message);
  logging_function_called = true;
}

static void __stdcall TestProfileEventCallback(const OrtProfilerEventRecord* profiler_record) noexcept {
  UNREFERENCED_PARAMETER(profiler_record);
  profiling_function_called = true;
}

static void EnvConfigureCustomLoggerAndProfiler() {
  OrtEnv* ort_env = nullptr;
  ort_api->CreateEnv(OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE, "Default", &ort_env);
  winml_adapter_api->EnvConfigureCustomLoggerAndProfiler(
    ort_env,
    &TestLoggingCallback,
    &TestProfileEventCallback,
    nullptr,
    OrtLoggingLevel::ORT_LOGGING_LEVEL_VERBOSE,
    "Default",
    &ort_env
  );
  logging_function_called = false;
  OrtSession* ort_session = nullptr;
  std::wstring squeezenet_path = FileHelpers::GetModulePath() + L"relu.onnx";
  ort_api->CreateSession(ort_env, squeezenet_path.c_str(), nullptr, &ort_session);
  WINML_EXPECT_TRUE(logging_function_called);

  size_t input_tensor_size = 5;
  int64_t input_dimensions[] = {5};

  std::vector<float> input_tensor_values(input_tensor_size);
  std::vector<const char*> input_node_names = {"X"};
  std::vector<const char*> output_node_names = {"Y"};

  // initialize input data with values in [0.0, 1.0]
  for (size_t i = 0; i < input_tensor_size; i++)
    input_tensor_values[i] = (float)i / (input_tensor_size + 1);

  // create input tensor object from data values
  OrtMemoryInfo* memory_info;
  ort_api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
  OrtValue* input_tensor = nullptr;
  ort_api->CreateTensorWithDataAsOrtValue(
    memory_info,
    input_tensor_values.data(),
    input_tensor_size * sizeof(float),
    input_dimensions,
    1,
    ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
    &input_tensor
  );
  int is_tensor;
  ort_api->IsTensor(input_tensor, &is_tensor);
  assert(is_tensor);
  ort_api->ReleaseMemoryInfo(memory_info);
  OrtValue* output_tensor = nullptr;
  winml_adapter_api->SessionStartProfiling(ort_env, ort_session);
  profiling_function_called = false;
  ort_api->Run(
    ort_session,
    nullptr,
    input_node_names.data(),
    (const OrtValue* const*)&input_tensor,
    1,
    output_node_names.data(),
    1,
    &output_tensor
  );
  WINML_EXPECT_TRUE(profiling_function_called);
  winml_adapter_api->SessionEndProfiling(ort_session);

  ort_api->ReleaseValue(output_tensor);
  ort_api->ReleaseValue(input_tensor);
  ort_api->ReleaseSession(ort_session);
  ort_api->ReleaseEnv(ort_env);
}

const AdapterTestApi& getapi() {
  static constexpr AdapterTestApi api = {
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

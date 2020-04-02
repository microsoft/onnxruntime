#include "testPch.h"
#include "adapter_model_test.h"
#include "fileHelpers.h"
#include "../../winml/adapter/winml_adapter_model.h"
#include "core/providers/winml/winml_provider_factory.h"



static void AdapterModelTestSetup() {
  winmlAdapter = OrtGetWinMLAdapter(OrtGetApiBase()->GetApi(2));
  std::wstring fullModelPath = FileHelpers::GetModulePath() + L"squeezenet_modifiedforruntimestests.onnx";
  std::string modelPath = std::wstring_convert<std::codecvt_utf8<wchar_t>>().to_bytes(fullModelPath);
  winmlAdapter->CreateModelFromPath(modelPath.c_str(), modelPath.size(), &ortModel);
}

static void CreateModelFromPath() {
  WINML_EXPECT_TRUE(ortModel != nullptr);
  WINML_EXPECT_TRUE(ortModel->UseModelProto() != nullptr);
}

static void CreateModelFromData() {
}

static void CloneModel() {
}

static void ModelGetAuthor() {
  const char* author;
  size_t* len;
  winmlAdapter->ModelGetAuthor(ortModel, author, len);
  std::string authorStr(author);
  WINML_EXPECT_EQUAL(authorStr, "onnx-caffe2");
}

static void ModelGetName() {
}

static void ModelGetDomain() {
}

static void ModelGetDescription() {
}

static void ModelGetVersion() {
}

static void ModelGetInputCount() {
}

static void ModelGetOutputCount() {
}

static void ModelGetInputName() {
}

static void ModelGetOutputName() {
}

static void ModelGetInputDescription() {
}

static void ModelGetOutputDescription() {
}

static void ModelGetInputTypeInfo() {
}

static void ModelGetOutputTypeInfo() {
}

static void ModelGetMetadataCount() {
}

static void ModelGetMetadata() {
}

static void ModelEnsureNoFloat16() {
}

const AdapterModelTestApi& getapi() {
  static constexpr AdapterModelTestApi api =
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
    ModelEnsureNoFloat16
  };
  return api;
}
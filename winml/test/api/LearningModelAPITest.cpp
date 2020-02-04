// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testPch.h"
#include "LearningModelAPITest.h"
#include "APITest.h"
#include <winrt/Windows.Graphics.Imaging.h>
#include <winrt/Windows.Media.h>
#include <winrt/Windows.Storage.h>
#include <winrt/Windows.Storage.Streams.h>

using namespace winrt;
using namespace winrt::Windows::AI::MachineLearning;
using namespace winrt::Windows::Foundation::Collections;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Storage::Streams;

static void LearningModelAPITestSetup() {
  init_apartment();
}

static void LearningModelAPITestGpuSetup() {
  GPUTEST;
  init_apartment();
}

static void CreateModelFromFilePath() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"squeezenet_modifiedforruntimestests.onnx", learningModel));
}

static void CreateModelFromIStorage() {
  std::wstring path = FileHelpers::GetModulePath() + L"squeezenet_modifiedforruntimestests.onnx";
  auto storageFile = winrt::Windows::Storage::StorageFile::GetFileFromPathAsync(path).get();
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(learningModel = LearningModel::LoadFromStorageFileAsync(storageFile).get());
  WINML_EXPECT_TRUE(learningModel != nullptr);

  // check the author so we know the model was populated correctly.
  std::wstring author(learningModel.Author());
  WINML_EXPECT_EQUAL(L"onnx-caffe2", author);
}

static void CreateModelFromIStorageOutsideCwd() {
  std::wstring path = FileHelpers::GetModulePath() + L"ModelSubdirectory\\ModelInSubdirectory.onnx";
  auto storageFile = winrt::Windows::Storage::StorageFile::GetFileFromPathAsync(path).get();
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(learningModel = LearningModel::LoadFromStorageFileAsync(storageFile).get());
  WINML_EXPECT_TRUE(learningModel != nullptr);

  // check the author so we know the model was populated correctly.
  std::wstring author(learningModel.Author());
  WINML_EXPECT_EQUAL(L"onnx-caffe2", author);
}

static void CreateModelFromIStream() {
  std::wstring path = FileHelpers::GetModulePath() + L"squeezenet_modifiedforruntimestests.onnx";
  auto storageFile = winrt::Windows::Storage::StorageFile::GetFileFromPathAsync(path).get();
  winrt::Windows::Storage::Streams::IRandomAccessStreamReference streamref;
  storageFile.as(streamref);
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(learningModel = LearningModel::LoadFromStreamAsync(streamref).get());
  WINML_EXPECT_TRUE(learningModel != nullptr);

  // check the author so we know the model was populated correctly.
  std::wstring author(learningModel.Author());
  WINML_EXPECT_EQUAL(L"onnx-caffe2", author);
}

static void ModelGetAuthor() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"squeezenet_modifiedforruntimestests.onnx", learningModel));
  std::wstring author(learningModel.Author());
  WINML_EXPECT_EQUAL(L"onnx-caffe2", author);
}

static void ModelGetName() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"squeezenet_modifiedforruntimestests.onnx", learningModel));
  std::wstring name(learningModel.Name());
  WINML_EXPECT_EQUAL(L"squeezenet_old", name);
}

static void ModelGetDomain() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"squeezenet_modifiedforruntimestests.onnx", learningModel));
  std::wstring domain(learningModel.Domain());
  WINML_EXPECT_EQUAL(L"test-domain", domain);
}

static void ModelGetDescription() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"squeezenet_modifiedforruntimestests.onnx", learningModel));
  std::wstring description(learningModel.Description());
  WINML_EXPECT_EQUAL(L"test-doc_string", description);
}

static void ModelGetVersion() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"squeezenet_modifiedforruntimestests.onnx", learningModel));
  int64_t version(learningModel.Version());
  (void)(version);
}

typedef std::vector<std::pair<std::wstring, std::wstring>> Metadata;

/*
class MetadataTest : public LearningModelAPITest, public testing::WithParamInterface<std::pair<std::wstring, Metadata>>
{};

TEST_P(MetadataTest, GetMetaData)
{
    std::wstring fileName;
    std::vector<std::pair<std::wstring, std::wstring>> keyValuePairs;

    tie(fileName, keyValuePairs) = GetParam();
    WINML_EXPECT_NO_THROW(LoadModel(fileName.c_str()));
    WINML_EXPECT_TRUE(m_model.Metadata() != nullptr);
    WINML_EXPECT_EQUAL(keyValuePairs.size(), m_model.Metadata().Size());

    auto iter = m_model.Metadata().First();
    for (auto& keyValue : keyValuePairs)
    {
        WINML_EXPECT_TRUE(iter.HasCurrent());
        WINML_EXPECT_EQUAL(keyValue.first, std::wstring(iter.Current().Key()));
        WINML_EXPECT_EQUAL(keyValue.second, std::wstring(iter.Current().Value()));
        iter.MoveNext();
    }
}

INSTANTIATE_TEST_SUITE_P(
    ModelMetadata,
    MetadataTest,
    ::testing::Values(
        std::pair(L"squeezenet_modifiedforruntimestests.onnx", Metadata{}),
        std::pair(L"modelWithMetaData.onnx", Metadata{{L"thisisalongkey", L"thisisalongvalue"}}),
        std::pair(L"modelWith2MetaData.onnx", Metadata{{L"thisisalongkey", L"thisisalongvalue"}, {L"key2", L"val2"}})
));
*/

static void EnumerateInputs() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"squeezenet_modifiedforruntimestests.onnx", learningModel));

  // purposely don't cache "InputFeatures" in order to exercise calling it multiple times
  WINML_EXPECT_TRUE(learningModel.InputFeatures().First().HasCurrent());

  std::wstring name(learningModel.InputFeatures().First().Current().Name());
  WINML_EXPECT_EQUAL(L"data_0", name);

  // make sure it's either tensor or image
  TensorFeatureDescriptor tensorDescriptor = nullptr;
  learningModel.InputFeatures().First().Current().try_as(tensorDescriptor);
  if (tensorDescriptor == nullptr) {
    ImageFeatureDescriptor imageDescriptor = nullptr;
    WINML_EXPECT_NO_THROW(learningModel.InputFeatures().First().Current().as(imageDescriptor));
  }

  auto modelDataKind = tensorDescriptor.TensorKind();
  WINML_EXPECT_EQUAL(TensorKind::Float, modelDataKind);

  WINML_EXPECT_TRUE(tensorDescriptor.IsRequired());

  std::vector<int64_t> expectedShapes = {1, 3, 224, 224};
  WINML_EXPECT_EQUAL(expectedShapes.size(), tensorDescriptor.Shape().Size());
  for (uint32_t j = 0; j < tensorDescriptor.Shape().Size(); j++) {
    WINML_EXPECT_EQUAL(expectedShapes.at(j), tensorDescriptor.Shape().GetAt(j));
  }

  auto first = learningModel.InputFeatures().First();
  first.MoveNext();
  WINML_EXPECT_FALSE(first.HasCurrent());
}

static void EnumerateOutputs() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"squeezenet_modifiedforruntimestests.onnx", learningModel));

  // purposely don't cache "OutputFeatures" in order to exercise calling it multiple times
  std::wstring name(learningModel.OutputFeatures().First().Current().Name());
  WINML_EXPECT_EQUAL(L"softmaxout_1", name);

  TensorFeatureDescriptor tensorDescriptor = nullptr;
  WINML_EXPECT_NO_THROW(learningModel.OutputFeatures().First().Current().as(tensorDescriptor));
  WINML_EXPECT_TRUE(tensorDescriptor != nullptr);

  auto tensorName = tensorDescriptor.Name();
  WINML_EXPECT_EQUAL(L"softmaxout_1", tensorName);

  auto modelDataKind = tensorDescriptor.TensorKind();
  WINML_EXPECT_EQUAL(TensorKind::Float, modelDataKind);

  WINML_EXPECT_TRUE(tensorDescriptor.IsRequired());

  std::vector<int64_t> expectedShapes = {1, 1000, 1, 1};
  WINML_EXPECT_EQUAL(expectedShapes.size(), tensorDescriptor.Shape().Size());
  for (uint32_t j = 0; j < tensorDescriptor.Shape().Size(); j++) {
    WINML_EXPECT_EQUAL(expectedShapes.at(j), tensorDescriptor.Shape().GetAt(j));
  }

  auto first = learningModel.OutputFeatures().First();
  first.MoveNext();
  WINML_EXPECT_FALSE(first.HasCurrent());
}

static void CloseModelCheckMetadata() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"squeezenet_modifiedforruntimestests.onnx", learningModel));
  WINML_EXPECT_NO_THROW(learningModel.Close());
  std::wstring author(learningModel.Author());
  WINML_EXPECT_EQUAL(L"onnx-caffe2", author);
  std::wstring name(learningModel.Name());
  WINML_EXPECT_EQUAL(L"squeezenet_old", name);
  std::wstring domain(learningModel.Domain());
  WINML_EXPECT_EQUAL(L"test-domain", domain);
  std::wstring description(learningModel.Description());
  WINML_EXPECT_EQUAL(L"test-doc_string", description);
  int64_t version(learningModel.Version());
  WINML_EXPECT_EQUAL(123456, version);
}

static void CloseModelCheckEval() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));
  LearningModelSession session = nullptr;
  WINML_EXPECT_NO_THROW(session = LearningModelSession(learningModel));
  WINML_EXPECT_NO_THROW(learningModel.Close());

  std::wstring fullImagePath = FileHelpers::GetModulePath() + L"kitten_224.png";
  StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
  IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
  SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
  VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);

  LearningModelBinding binding = nullptr;
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  WINML_EXPECT_NO_THROW(binding.Bind(learningModel.InputFeatures().First().Current().Name(), frame));

  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));
}

static void CloseModelNoNewSessions() {
  LearningModel learningModel = nullptr;
  WINML_EXPECT_NO_THROW(APITest::LoadModel(L"model.onnx", learningModel));
  WINML_EXPECT_NO_THROW(learningModel.Close());
  LearningModelSession session = nullptr;
  WINML_EXPECT_THROW_SPECIFIC(
      session = LearningModelSession(learningModel);,
      winrt::hresult_error,
      [](const winrt::hresult_error& e) -> bool {
            return e.code() == E_INVALIDARG;
      });
}

const LearningModelApiTestApi& getapi() {
  static constexpr LearningModelApiTestApi api =
  {
    LearningModelAPITestSetup,
    LearningModelAPITestGpuSetup,
    CreateModelFromFilePath,
    CreateModelFromIStorage,
    CreateModelFromIStorageOutsideCwd,
    CreateModelFromIStream,
    ModelGetAuthor,
    ModelGetName,
    ModelGetDomain,
    ModelGetDescription,
    ModelGetVersion,
    EnumerateInputs,
    EnumerateOutputs,
    CloseModelCheckMetadata,
    CloseModelCheckEval,
    CloseModelNoNewSessions
  };
  return api;
}
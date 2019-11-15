#include "pch.h"
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

namespace
{
    // void MetaDataVerifyHelper(
    //     std::wstring expectedKeyParam,
    //     std::wstring expectedValueParam,
    //     IIterator<IKeyValuePair<hstring, hstring>> iter)
    // {
    //     std::wstring expectedKey, expectedValue;
    //     if (FAILED(TestData::TryGetValue(expectedKeyParam, expectedKey))
    //         || FAILED(TestData::TryGetValue(expectedValueParam, expectedValue)))
    //     {
    //         throw_hresult(E_INVALIDARG);
    //     }

    //     EXPECT_TRUE(iter.HasCurrent());
    //     EXPECT_EQ(static_cast<const wchar_t*>(expectedKey), std::wstring(iter.Current().Key()));
    //     EXPECT_EQ(static_cast<const wchar_t*>(expectedValue), std::wstring(iter.Current().Value()));
    // }
}

class LearningModelAPITest : public APITest
{
protected:
    LearningModelAPITest() {
        init_apartment();
        m_model = nullptr;
        m_device = nullptr;
        m_session = nullptr;
    }
};

class LearningModelAPITestGpu : public APITest
{};

TEST_F(LearningModelAPITest, CreateModelFromFilePath)
{
    EXPECT_NO_THROW(LoadModel(L"squeezenet_modifiedforruntimestests.onnx"));
}

TEST_F(LearningModelAPITest, CreateModelFromIStorage)
{
    std::wstring path = FileHelpers::GetModulePath() + L"squeezenet_modifiedforruntimestests.onnx";
    auto storageFile = winrt::Windows::Storage::StorageFile::GetFileFromPathAsync(path).get();
    EXPECT_NO_THROW(m_model = LearningModel::LoadFromStorageFileAsync(storageFile).get());
    EXPECT_TRUE(m_model != nullptr);

    // check the author so we know the model was populated correctly.
    std::wstring author(m_model.Author());
    EXPECT_EQ(L"onnx-caffe2", author);
}

TEST_F(LearningModelAPITest, CreateModelFromIStorageOutsideCwd)
{
    std::wstring path = FileHelpers::GetModulePath() + L"ModelSubdirectory\\ModelInSubdirectory.onnx";
    auto storageFile = winrt::Windows::Storage::StorageFile::GetFileFromPathAsync(path).get();
    EXPECT_NO_THROW(m_model = LearningModel::LoadFromStorageFileAsync(storageFile).get());
    EXPECT_TRUE(m_model != nullptr);

    // check the author so we know the model was populated correctly.
    std::wstring author(m_model.Author());
    EXPECT_EQ(L"onnx-caffe2", author);
}

TEST_F(LearningModelAPITest, CreateModelFromIStream)
{
    std::wstring path = FileHelpers::GetModulePath() + L"squeezenet_modifiedforruntimestests.onnx";
    auto storageFile = winrt::Windows::Storage::StorageFile::GetFileFromPathAsync(path).get();
    winrt::Windows::Storage::Streams::IRandomAccessStreamReference streamref;
    storageFile.as(streamref);

    EXPECT_NO_THROW(m_model = LearningModel::LoadFromStreamAsync(streamref).get());
    EXPECT_TRUE(m_model != nullptr);

    // check the author so we know the model was populated correctly.
    std::wstring author(m_model.Author());
    EXPECT_EQ(L"onnx-caffe2", author);
}

TEST_F(LearningModelAPITest, GetAuthor)
{
    EXPECT_NO_THROW(LoadModel(L"squeezenet_modifiedforruntimestests.onnx"));
    std::wstring author(m_model.Author());
    EXPECT_EQ(L"onnx-caffe2", author);
}

TEST_F(LearningModelAPITest, GetName)
{
    EXPECT_NO_THROW(LoadModel(L"squeezenet_modifiedforruntimestests.onnx"));
    std::wstring name(m_model.Name());
    EXPECT_EQ(L"squeezenet_old", name);
}

TEST_F(LearningModelAPITest, GetDomain)
{
    EXPECT_NO_THROW(LoadModel(L"squeezenet_modifiedforruntimestests.onnx"));
    std::wstring domain(m_model.Domain());
    EXPECT_EQ(L"test-domain", domain);
}

TEST_F(LearningModelAPITest, GetDescription)
{
    EXPECT_NO_THROW(LoadModel(L"squeezenet_modifiedforruntimestests.onnx"));
    std::wstring description(m_model.Description());
    EXPECT_EQ(L"test-doc_string", description);
}

TEST_F(LearningModelAPITest, GetVersion)
{
    EXPECT_NO_THROW(LoadModel(L"squeezenet_modifiedforruntimestests.onnx"));
    int64_t version(m_model.Version());
}

// TEST_F(LearningModelAPITest, GetMetaData)
// {
//     // only checking cases for metaData counts 0,1,2
//     // because I can't figure out how to make a string array in the xml data file
//     String fileName;
//     int expectedEntries;
//     if (FAILED(TestData::TryGetValue(L"fileName", fileName)) || FAILED(TestData::TryGetValue(L"expectedEntries", expectedEntries)))
//     {
//         throw_hresult(E_INVALIDARG);
//     }

//     EXPECT_NO_THROW(LoadModel(static_cast<const wchar_t*>(fileName)));
//     EXPECT_TRUE(m_model != nullptr.Metadata());
//     EXPECT_EQ(expectedEntries, m_model.Metadata().Size());

//     auto first = m_model.Metadata().First();
//     if (expectedEntries > 0)
//     {
//         MetaDataVerifyHelper(L"expectedKey1", L"expectedValue1", first);
//         first.MoveNext();

//         if (expectedEntries > 1)
//         {
//             MetaDataVerifyHelper(L"expectedKey2", L"expectedValue2", first);
//             first.MoveNext();
//         }
//     }

//     EXPECT_FALSE(first.HasCurrent());
// }

TEST_F(LearningModelAPITest, EnumerateInputs)
{
    EXPECT_NO_THROW(LoadModel(L"squeezenet_modifiedforruntimestests.onnx"));

    // purposely don't cache "InputFeatures" in order to exercise calling it multiple times
    EXPECT_TRUE(m_model.InputFeatures().First().HasCurrent());

    std::wstring name(m_model.InputFeatures().First().Current().Name());
    EXPECT_EQ(L"data_0", name);

    // make sure it's either tensor or image
    TensorFeatureDescriptor tensorDescriptor = nullptr;
    m_model.InputFeatures().First().Current().try_as(tensorDescriptor);
    if (tensorDescriptor == nullptr)
    {
        ImageFeatureDescriptor imageDescriptor = nullptr;
        EXPECT_NO_THROW(m_model.InputFeatures().First().Current().as(imageDescriptor));
    }

    auto modelDataKind = tensorDescriptor.TensorKind();
    EXPECT_EQ(TensorKind::Float, modelDataKind);

    EXPECT_TRUE(tensorDescriptor.IsRequired());

    std::vector<int64_t> expectedShapes = { 1,3,224,224 };
    EXPECT_EQ(expectedShapes.size(), tensorDescriptor.Shape().Size());
    for (uint32_t j = 0; j < tensorDescriptor.Shape().Size(); j++)
    {
        EXPECT_EQ(expectedShapes.at(j), tensorDescriptor.Shape().GetAt(j));
    }

    auto first = m_model.InputFeatures().First();
    first.MoveNext();
    EXPECT_FALSE(first.HasCurrent());
}

TEST_F(LearningModelAPITest, EnumerateOutputs)
{
    EXPECT_NO_THROW(LoadModel(L"squeezenet_modifiedforruntimestests.onnx"));

    // purposely don't cache "OutputFeatures" in order to exercise calling it multiple times
    std::wstring name(m_model.OutputFeatures().First().Current().Name());
    EXPECT_EQ(L"softmaxout_1", name);

    TensorFeatureDescriptor tensorDescriptor = nullptr;
    EXPECT_NO_THROW(m_model.OutputFeatures().First().Current().as(tensorDescriptor));
    EXPECT_TRUE(tensorDescriptor != nullptr);

    auto tensorName = tensorDescriptor.Name();
    EXPECT_EQ(L"softmaxout_1", tensorName);

    auto modelDataKind = tensorDescriptor.TensorKind();
    EXPECT_EQ(TensorKind::Float, modelDataKind);

    EXPECT_TRUE(tensorDescriptor.IsRequired());

    std::vector<int64_t> expectedShapes = { 1, 1000, 1, 1 };
    EXPECT_EQ(expectedShapes.size(), tensorDescriptor.Shape().Size());
    for (uint32_t j = 0; j < tensorDescriptor.Shape().Size(); j++)
    {
        EXPECT_EQ(expectedShapes.at(j), tensorDescriptor.Shape().GetAt(j));
    }

    auto first = m_model.OutputFeatures().First();
    first.MoveNext();
    EXPECT_FALSE(first.HasCurrent());
}

TEST_F(LearningModelAPITest, CloseModelCheckMetadata)
{
    EXPECT_NO_THROW(LoadModel(L"squeezenet_modifiedforruntimestests.onnx"));
    EXPECT_NO_THROW(m_model.Close());
    std::wstring author(m_model.Author());
    EXPECT_EQ(L"onnx-caffe2", author);
    std::wstring name(m_model.Name());
    EXPECT_EQ(L"squeezenet_old", name);
    std::wstring domain(m_model.Domain());
    EXPECT_EQ(L"test-domain", domain);
    std::wstring description(m_model.Description());
    EXPECT_EQ(L"test-doc_string", description);
    int64_t version(m_model.Version());
    EXPECT_EQ(123456, version);
}

TEST_F(LearningModelAPITestGpu, CloseModelCheckEval)
{
    EXPECT_NO_THROW(LoadModel(L"model.onnx"));
    LearningModelSession session = nullptr;
    EXPECT_NO_THROW(session = LearningModelSession(m_model));
    EXPECT_NO_THROW(m_model.Close());

    std::wstring fullImagePath = FileHelpers::GetModulePath() + L"kitten_224.png";
    StorageFile imagefile = StorageFile::GetFileFromPathAsync(fullImagePath).get();
    IRandomAccessStream stream = imagefile.OpenAsync(FileAccessMode::Read).get();
    SoftwareBitmap softwareBitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
    VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);

    LearningModelBinding binding = nullptr;
    EXPECT_NO_THROW(binding = LearningModelBinding(session));
    EXPECT_NO_THROW(binding.Bind(m_model.InputFeatures().First().Current().Name(), frame));

    EXPECT_NO_THROW(session.Evaluate(binding, L""));
}

TEST_F(LearningModelAPITest, CloseModelNoNewSessions)
{
    EXPECT_NO_THROW(LoadModel(L"model.onnx"));
    EXPECT_NO_THROW(m_model.Close());
    LearningModelSession session = nullptr;
    EXPECT_THROW(
        try {
            session = LearningModelSession(m_model);
        } catch (const winrt::hresult_error& e) {
            EXPECT_EQ(E_INVALIDARG, e.code());
            throw;
        }
    , winrt::hresult_error);
}

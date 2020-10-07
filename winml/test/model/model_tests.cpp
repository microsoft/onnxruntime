#include "testPch.h"
#include "common.h"
#include "gtest/gtest.h"
#include "onnxruntime_cxx_api.h"
#include "test/onnx/TestCase.h"
#include "test/onnx/heap_buffer.h"
#include "test/util/include/test/compare_ortvalue.h"
#include "ort_value_helper.h"
#include <iostream>
/*
int main() {
  std::vector<ITestCase*> tests;
  std::vector<std::unique_ptr<ITestCase>> owned_tests;
  std::vector<std::basic_string<PATH_CHAR_TYPE>> whitelisted_test_cases;
  double per_sample_tolerance = 1e-3;
  double relative_per_sample_tolerance = 1e-3;
  std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests;
  std::vector<std::basic_string<PATH_CHAR_TYPE>> data_dirs;
  data_dirs.push_back(L"D:\\models\\models\\opset11");
  LoadTests(data_dirs, whitelisted_test_cases, per_sample_tolerance, relative_per_sample_tolerance,
            all_disabled_tests,
            [&owned_tests, &tests](std::unique_ptr<ITestCase> l) {
              tests.push_back(l.get());
              owned_tests.push_back(std::move(l));
            });
  onnxruntime::test::HeapBuffer holder;
  std::unordered_map<std::string, Ort::Value> feeds;
  owned_tests[5]->LoadTestData(1, holder, feeds, true);
  auto model = LearningModel::LoadFromFilePath(owned_tests[5]->GetModelUrl());
  auto device = LearningModelDevice(LearningModelDeviceKind::DirectX);
  auto session = LearningModelSession(model, device);
  LearningModelBinding binding(session);

  ITensor alternative_binding = OrtValueHelpers::LoadTensorFromOrtValue(feeds.at("input:0"));
  binding.Bind(L"input:0", alternative_binding);
  auto results = session.Evaluate(binding, L"Testing");
  auto resultTensor = results.Outputs().Lookup(L"resnet_v1_152/predictions/Reshape_1:0").as<TensorFloat>().GetAsVectorView();

  std::unordered_map<std::string, Ort::Value> output_feeds;
  owned_tests[5]->LoadTestData(1, holder, output_feeds, false);
  ITensor expected_output = OrtValueHelpers::LoadTensorFromOrtValue(output_feeds.at("resnet_v1_152/predictions/Reshape_1:0"));
  auto expectedTensor = expected_output.as<TensorFloat>().GetAsVectorView();
  for (uint32_t i = 0; i < resultTensor.Size(); i++) {
    std::cout << resultTensor.GetAt(i) << " " << expectedTensor.GetAt(i) << std::endl;
  }
}
*/
using namespace onnxruntime::test;
using namespace winml;
using namespace onnxruntime;
std::vector<std::unique_ptr<ITestCase>> owned_tests;

std::vector<ITestCase*> GetAllTestCases() {
  std::vector<ITestCase*> tests;
  std::vector<std::basic_string<PATH_CHAR_TYPE>> whitelisted_test_cases;
  double per_sample_tolerance = 1e-3;
  double relative_per_sample_tolerance = 1e-3;
  std::unordered_set<std::basic_string<ORTCHAR_T>> all_disabled_tests;
  std::vector<std::basic_string<PATH_CHAR_TYPE>> data_dirs;
  data_dirs.push_back(L"D:\\models\\models\\opset7");
  data_dirs.push_back(L"D:\\models\\models\\opset8");
  data_dirs.push_back(L"D:\\models\\models\\opset9");
  data_dirs.push_back(L"D:\\models\\models\\opset10");
  data_dirs.push_back(L"D:\\models\\models\\opset11");
  LoadTests(data_dirs, whitelisted_test_cases, per_sample_tolerance, relative_per_sample_tolerance,
            all_disabled_tests,
            [&tests](std::unique_ptr<ITestCase> l) {
              tests.push_back(l.get());
              owned_tests.push_back(std::move(l));
            });
  return tests;
}

namespace WinML {
class ModelTest : public testing::TestWithParam<std::tuple<ITestCase*, winml::LearningModelDeviceKind>> {
 protected:
  void SetUp() override {
    std::tie(testCase, deviceKind) = GetParam();
    testCase->GetPerSampleTolerance(&perSampleTolerance);
    testCase->GetRelativePerSampleTolerance(&relativePerSampleTolerance);
    testCase->GetPostProcessing(&postProcessing);
  }
  winml::LearningModelDeviceKind deviceKind;
  ITestCase* testCase;
  std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
  double perSampleTolerance = 1e-3;
  double relativePerSampleTolerance = 1e-3;
  bool postProcessing = false;

  void BindInputsFromFeed(LearningModelBinding& binding, std::unordered_map<std::string, Ort::Value>& feed) {
    for (auto it = feed.begin(); it != feed.end(); it++) {
      auto bindingName = it->first;
      auto bindingValue = OrtValueHelpers::LoadTensorFromOrtValue(it->second);
      binding.Bind(converter.from_bytes(bindingName), bindingValue);
    }
  }

  std::pair<COMPARE_RESULT, std::string> CompareEvaluationResults(LearningModelEvaluationResult& results,
      std::unordered_map<std::string, Ort::Value>& expectedOutputFeeds) {
     
    for (auto it = expectedOutputFeeds.begin(); it != expectedOutputFeeds.end(); it++) {
      std::wstring outputName = converter.from_bytes(it->first.c_str());
      auto actualOutputTensorValue = results.Outputs().Lookup(outputName).as<ITensorNative>();
      void* actualData;
      uint32_t actualSizeInBytes;
      actualOutputTensorValue->GetBuffer(reinterpret_cast<BYTE**>(&actualData), &actualSizeInBytes);

      auto expectedShapeAndTensorType = it->second.GetTensorTypeAndShapeInfo();
      auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
      Ort::Value actualOutput = Ort::Value::CreateTensor(
          memoryInfo,
          actualData,
          actualSizeInBytes,
          expectedShapeAndTensorType.GetShape().data(),
          expectedShapeAndTensorType.GetShape().size(),
          expectedShapeAndTensorType.GetElementType());
      std::pair<COMPARE_RESULT, std::string> ret = CompareOrtValue(*actualOutput, *it->second, perSampleTolerance, relativePerSampleTolerance, postProcessing);
      EXPECT_EQ(COMPARE_RESULT::SUCCESS, ret.first) << ret.second;
    }
    return std::make_pair(COMPARE_RESULT::SUCCESS, "");
  }
};

TEST_P(ModelTest, Run) {
  auto model = LearningModel::LoadFromFilePath(testCase->GetModelUrl());
  auto device = LearningModelDevice(deviceKind);
  auto session = LearningModelSession(model, device);
  auto binding = LearningModelBinding(session);
  for (int i = 0; i < testCase->GetDataCount(); i++) {
    onnxruntime::test::HeapBuffer inputHolder;
    std::unordered_map<std::string, Ort::Value> inputFeeds;
    testCase->LoadTestData(i, inputHolder, inputFeeds, true);
    BindInputsFromFeed(binding, inputFeeds);

    // evaluate
    auto results = session.Evaluate(binding, L"Testing");

    onnxruntime::test::HeapBuffer outputHolder;
    std::unordered_map<std::string, Ort::Value> outputFeeds;
    testCase->LoadTestData(i, outputHolder, outputFeeds, false);

    // compare results
    CompareEvaluationResults(results, outputFeeds);
  }
}

INSTANTIATE_TEST_SUITE_P(ModelTests, ModelTest, testing::Combine(testing::ValuesIn(GetAllTestCases()), testing::Values(winml::LearningModelDeviceKind::Cpu, winml::LearningModelDeviceKind::DirectX)),
                         [](const testing::TestParamInfo<ModelTest::ParamType>& info) {
                           std::string name = "";
                           auto modelPath = std::wstring(std::get<0>(info.param)->GetModelUrl());
                           auto modelPathStr = std::string(modelPath.begin(), modelPath.end());
                           std::vector<std::string> tokenizedModelPath;
                           std::istringstream ss(modelPathStr);
                           std::string token;
                           while (std::getline(ss, token, '\\')) {
                             tokenizedModelPath.push_back(token);
                           }
                           name += tokenizedModelPath[tokenizedModelPath.size() - 2] += "_";
                           name += tokenizedModelPath[tokenizedModelPath.size() - 3] += "_";

                           if (std::get<1>(info.param) == winml::LearningModelDeviceKind::Cpu) {
                             name += "CPU";
                           } else {
                             name += "GPU";
                           }

                           std::replace_if(name.begin(), name.end(), [&name](char c) { return !std::isalnum(c, std::locale()); }, '_');
                           return name;
                         });
}  // namespace WinML
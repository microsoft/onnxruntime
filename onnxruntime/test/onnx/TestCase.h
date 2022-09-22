// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <core/common/common.h>
#include <core/common/status.h>
#include <core/platform/path_lib.h>

namespace Ort {
struct Value;
}

namespace ONNX_NAMESPACE {
class ValueInfoProto;
}

namespace onnxruntime {
namespace test {
class HeapBuffer;
}
}  // namespace onnxruntime

// One test case is for one model file
// One test case can contain multiple test data(input/output pairs)
class ITestCase {
 public:
  virtual void LoadTestData(size_t id, onnxruntime::test::HeapBuffer& b,
                            std::unordered_map<std::string, Ort::Value>& name_data_map,
                            bool is_input) const = 0;
  virtual const PATH_CHAR_TYPE* GetModelUrl() const = 0;
  virtual const std::string& GetNodeName() const = 0;
  virtual const ONNX_NAMESPACE::ValueInfoProto* GetInputInfoFromModel(size_t i) const = 0;
  virtual const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const = 0;

  virtual const std::string& GetTestCaseName() const = 0;
  virtual std::string GetTestCaseVersion() const = 0;
  // a string to help identify the dataset
  virtual std::string GetDatasetDebugInfoString(size_t dataset_id) const = 0;
  // The number of input/output pairs
  virtual size_t GetDataCount() const = 0;
  virtual ~ITestCase() = default;
  virtual void GetPerSampleTolerance(double* value) const = 0;
  virtual void GetRelativePerSampleTolerance(double* value) const = 0;
  virtual void GetPostProcessing(bool* value) const = 0;
};

class TestModelInfo {
 public:
  virtual const PATH_CHAR_TYPE* GetModelUrl() const = 0;
  virtual std::basic_string<PATH_CHAR_TYPE> GetDir() const {
    std::basic_string<PATH_CHAR_TYPE> test_case_dir;
    auto st = onnxruntime::GetDirNameFromFilePath(GetModelUrl(), test_case_dir);
    if (!st.IsOK()) {
      ORT_THROW("GetDirNameFromFilePath failed");
    }
    return test_case_dir;
  }
  virtual const std::string& GetNodeName() const = 0;
  virtual const ONNX_NAMESPACE::ValueInfoProto* GetInputInfoFromModel(size_t i) const = 0;
  virtual const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const = 0;
  virtual int GetInputCount() const = 0;
  virtual int GetOutputCount() const = 0;
  virtual const std::string& GetInputName(size_t i) const = 0;
  virtual const std::string& GetOutputName(size_t i) const = 0;
  virtual std::string GetNominalOpsetVersion() const { return ""; }
  virtual ~TestModelInfo() = default;

#if !defined(ORT_MINIMAL_BUILD)
  static std::unique_ptr<TestModelInfo> LoadOnnxModel(_In_ const PATH_CHAR_TYPE* model_url);
#endif

  static std::unique_ptr<TestModelInfo> LoadOrtModel(_In_ const PATH_CHAR_TYPE* model_url);

  static const std::string unknown_version;
};

std::unique_ptr<ITestCase> CreateOnnxTestCase(const std::string& test_case_name,
                                              std::unique_ptr<TestModelInfo> model,
                                              double default_per_sample_tolerance,
                                              double default_relative_per_sample_tolerance);

class TestTolerances {
 public:
  typedef std::unordered_map<std::string, double> Map;
  TestTolerances(
      double absolute_default, double relative_default,
      const Map& absolute_overrides,
      const Map& relative_overrides);
  TestTolerances() = delete;
  double absolute(const std::string& test_name) const;
  double relative(const std::string& test_name) const;

 private:
  double absolute_default_;
  double relative_default_;
  const Map absolute_overrides_;
  const Map relative_overrides_;
};

void LoadTests(const std::vector<std::basic_string<PATH_CHAR_TYPE>>& input_paths,
               const std::vector<std::basic_string<PATH_CHAR_TYPE>>& whitelisted_test_cases,
               const TestTolerances& tolerances,
               const std::unordered_set<std::basic_string<ORTCHAR_T>>& disabled_tests,
               const std::function<void(std::unique_ptr<ITestCase>)>& process_function);

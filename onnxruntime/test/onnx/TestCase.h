// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <vector>
#include <mutex>
#include <unordered_map>
#include <core/common/status.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <core/framework/path_lib.h>
#include "heap_buffer.h"
namespace ONNX_NAMESPACE {
class ValueInfoProto;
}

//One test case is for one model file
//One test case can contain multiple test data(input/output pairs)
class ITestCase {
 public:
  virtual void LoadTestData(size_t id, HeapBuffer& b, std::unordered_map<std::string, OrtValue*>& name_data_map,
                            bool is_input) = 0;
  virtual const PATH_CHAR_TYPE* GetModelUrl() const = 0;
  virtual const std::string& GetNodeName() const = 0;
  virtual const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const = 0;

  virtual const std::string& GetTestCaseName() const = 0;
  //a string to help identify the dataset
  virtual std::string GetDatasetDebugInfoString(size_t dataset_id) = 0;
  //The number of input/output pairs
  virtual size_t GetDataCount() const = 0;
  virtual ~ITestCase() = default;
  virtual ::onnxruntime::common::Status GetPerSampleTolerance(double* value) = 0;
  virtual ::onnxruntime::common::Status GetRelativePerSampleTolerance(double* value) = 0;
  virtual ::onnxruntime::common::Status GetPostProcessing(bool* value) = 0;
};

class TestModelInfo {
 public:
  virtual const PATH_CHAR_TYPE* GetModelUrl() const = 0;
  virtual std::basic_string<PATH_CHAR_TYPE> GetDir() const = 0;
  virtual const std::string& GetNodeName() const = 0;
  virtual const ONNX_NAMESPACE::ValueInfoProto* GetOutputInfoFromModel(size_t i) const = 0;
  virtual int GetInputCount() const = 0;
  virtual int GetOutputCount() const = 0;
  virtual const std::string& GetInputName(size_t i) const = 0;
  virtual const std::string& GetOutputName(size_t i) const = 0;
  virtual ~TestModelInfo() = default;

  static TestModelInfo* LoadOnnxModel(_In_ const PATH_CHAR_TYPE* model_url);
};

ITestCase* CreateOnnxTestCase(const std::string& test_case_name, TestModelInfo* model,
                              double default_per_sample_tolerance, double default_relative_per_sample_tolerance);

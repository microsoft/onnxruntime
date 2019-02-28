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
  //must be called before calling the other functions
  virtual ::onnxruntime::common::Status SetModelPath(_In_ const PATH_CHAR_TYPE* path) ORT_ALL_ARGS_NONNULL = 0;
  virtual ::onnxruntime::common::Status LoadTestData(OrtSession* session, size_t id, HeapBuffer& b,
                                                     std::unordered_map<std::string, OrtValue*>& name_data_map,
                                                     bool is_input) = 0;
  virtual const PATH_CHAR_TYPE* GetModelUrl() const = 0;
  virtual const std::string& GetTestCaseName() const = 0;
  //a string to help identify the dataset
  virtual std::string GetDatasetDebugInfoString(size_t dataset_id) = 0;
  virtual ::onnxruntime::common::Status GetNodeName(std::string* out) = 0;
  //The number of input/output pairs
  virtual size_t GetDataCount() const = 0;
  virtual const ONNX_NAMESPACE::ValueInfoProto& GetOutputInfoFromModel(size_t i) const = 0;
  virtual ~ITestCase() {}
  virtual ::onnxruntime::common::Status GetPerSampleTolerance(double* value) = 0;
  virtual ::onnxruntime::common::Status GetRelativePerSampleTolerance(double* value) = 0;
  virtual ::onnxruntime::common::Status GetPostProcessing(bool* value) = 0;
};

ITestCase* CreateOnnxTestCase(const std::string& test_case_name);

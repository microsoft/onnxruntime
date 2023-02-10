// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef __ONNX_PREDICTION_H__
#define __ONNX_PREDICTION_H__
#include <iostream>
#include <fstream>
#include <algorithm>
#include <set>
#include <string>
#include <sstream>
#include <tuple>
#include <utility>
#include <vector>
#include <list>
#include <functional>
#include <ctime>
#include <string_view>
#include <cwchar>
#include <filesystem>

#include "BetaDistribution.h"
#include "onnx/onnx_pb.h"
#include "onnxruntime_cxx_api.h"

#include "testlog.h"

class OnnxPrediction {
  friend std::wostream& operator<<(std::wostream& out, OnnxPrediction& pred);
  friend Logger::TestLog& Logger::TestLog::operator<<(OnnxPrediction& pred);

 public:
  using InputGeneratorFunctionType = std::function<void(OnnxPrediction&,
                                                        size_t, const std::string&,
                                                        ONNXTensorElementDataType,
                                                        size_t, size_t)>;

 public:
  // Uses the onnxruntime to load the model
  // into a session.
  //
  OnnxPrediction(std::wstring& onnx_model_file, Ort::Env& env);

  // Uses the onnx model to create a prediction
  // environment
  //
  OnnxPrediction(onnx::ModelProto& onnx_model, Ort::Env& env);

  // The following constructor is meant for initializing using flatbuffer model.
  // Memory buffer pointing to the model
  //
  OnnxPrediction(const std::vector<char>& model_data, Ort::Env& env);

  // Data to run prediction on
  //
  template <typename T>
  void operator<<(std::vector<T>&& raw_data) {
    if (curr_input_index >= ptr_session->GetInputCount()) {
      return;
    }

    Ort::Value& input_value = input_values[curr_input_index];
    auto data_size_in_bytes = raw_data.size() * sizeof(T);
    // TODO The following allocation may be unnecessary
    // Copy the raw input data and control the lifetime.
    //
    input_data.emplace_back(alloc.Alloc(data_size_in_bytes),
                            [this](void* ptr1) {
                              this->GetAllocator().Free(ptr1);
                            });

    std::copy(raw_data.begin(), raw_data.end(), reinterpret_cast<T*>(input_data[curr_input_index].get()));
    auto input_type = ptr_session->GetInputTypeInfo(curr_input_index);
    auto shapeInfo = input_type.GetTensorTypeAndShapeInfo().GetShape();
    auto elem_type = input_type.GetTensorTypeAndShapeInfo().GetElementType();
    if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
      input_value = Ort::Value::CreateTensor(alloc.GetInfo(),
                                             input_data[curr_input_index].get(), data_size_in_bytes, shapeInfo.data(), shapeInfo.size(), elem_type);
    } else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
      input_value = Ort::Value::CreateTensor(alloc.GetInfo(),
                                             input_data[curr_input_index].get(), data_size_in_bytes, shapeInfo.data(), shapeInfo.size(), elem_type);
    } else {
      throw std::exception("only floats are implemented");
    }

    // Insert data into the next input type
    //
    curr_input_index++;
  }

  // Used to generate
  //
  void SetupInput(
      InputGeneratorFunctionType GenerateData,
      size_t seed = static_cast<unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));

  // Run the prediction
  //
  void RunInference();

  // Do operation on the output data
  //
  template <typename T>
  void ProcessOutputData(T process_function, Ort::Value& val) {
    if (val.IsTensor()) {
      if (val.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        auto ptr = val.GetTensorMutableData<float>();
        process_function(ptr, val);
      } else if (val.GetTensorTypeAndShapeInfo().GetElementType() == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32) {
        auto ptr = val.GetTensorMutableData<int32_t>();
        process_function(ptr, val);
      }
    }
  }

  // Print out the result of the predicition
  //
  void PrintOutputValues();

 private:
  // Common initilization amongst constructors
  //
  void init();

  // Get the allocator used by the runtime
  //
  Ort::AllocatorWithDefaultOptions& GetAllocator();

 private:
  // Create an allocator for the runtime to use
  //
  Ort::AllocatorWithDefaultOptions alloc;

  // Create Options for the Session
  //
  Ort::SessionOptions empty_session_option;

  // A pointer to the model
  //
  std::shared_ptr<void> raw_model;

  // Create RunOptions
  //
  Ort::RunOptions run_options;

  // Pointer to the current session object
  //
  std::unique_ptr<Ort::Session> ptr_session;

  // Create a list of input values
  //
  std::vector<Ort::Value> input_values{};

  // Stores the input names
  //
  std::vector<char*> input_names;

  std::vector<Ort::AllocatedStringPtr> input_names_ptrs;

  // Stores the output names
  //
  std::vector<char*> output_names;

  std::vector<Ort::AllocatedStringPtr> output_names_ptrs;

  // Create a list of output values
  //
  std::vector<Ort::Value> output_values{};

  // We own the lifetime of the input data
  //
  std::vector<std::shared_ptr<void>> input_data;

  // Keeps track of number of columns/dimensions data
  // given for predicition
  //
  size_t curr_input_index{0};
};

// OnnxPrediction console output format
// prints the output data.
//
std::ostream& operator<<(std::ostream& out, OnnxPrediction& pred);

// Used to Generate data for predict
//
void GenerateDataForInputTypeTensor(OnnxPrediction& predict,
                                    size_t input_index, const std::string& input_name,
                                    ONNXTensorElementDataType elem_type, size_t elem_count, size_t seed);
#endif

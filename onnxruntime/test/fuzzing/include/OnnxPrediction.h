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
#include "onnx/onnx-ml.pb.h"
#include "onnxruntime_cxx_api.h"

#include "testlog.h"

class OnnxPrediction
{
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
  OnnxPrediction(std::wstring& onnx_model_file);

  // Uses the onnx model to create a prediction
  // environment
  //
  OnnxPrediction(onnx::ModelProto& onnx_model);

  // Deletes the prediction object 
  //
  ~OnnxPrediction();

  // Data to run prediction on
  //
  template<typename T>
  void operator<<(std::vector<T>&& raw_data)
  {
    if( currInputIndex >= ptrSession->GetInputCount())
    {
      return;
    }

    Ort::Value& inputValue = inputValues[currInputIndex];
    auto data_size_in_bytes = raw_data.size() * sizeof(T);

    // Copy the raw input data and control the lifetime.
    //
    inputData.emplace_back(alloc.Alloc(data_size_in_bytes), 
    [this](void * ptr1)
        { 
          this->GetAllocator().Free(ptr1); 
        }
    );
    
    std::copy(raw_data.begin(), raw_data.end(), reinterpret_cast<T*>(inputData[currInputIndex].get()));
    auto inputType = ptrSession->GetInputTypeInfo(currInputIndex);
    auto shapeInfo = inputType.GetTensorTypeAndShapeInfo().GetShape();
    auto elem_type = inputType.GetTensorTypeAndShapeInfo().GetElementType();
    if ( elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
    {
      inputValue = Ort::Value::CreateTensor(alloc.GetInfo(),
          inputData[currInputIndex].get(), data_size_in_bytes, shapeInfo.data(), shapeInfo.size(), elem_type);
    }
    else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
    {
      inputValue = Ort::Value::CreateTensor(alloc.GetInfo(),
          inputData[currInputIndex].get(), data_size_in_bytes, shapeInfo.data(), shapeInfo.size(), elem_type);
    }
    else
    {
      throw std::exception("only floats are implemented");
    }

    // Insert data into the next input type
    //
    currInputIndex++;
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
  template<typename T>
  void ProcessOutputData(T process_function, Ort::Value& val)
  {
    if ( val.IsTensor() )
    {
      if (val.GetTensorTypeAndShapeInfo().GetElementType() 
            == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
      {
        auto ptr = val.GetTensorMutableData<float>();
        process_function(ptr, val);
      }
      else if (val.GetTensorTypeAndShapeInfo().GetElementType() 
            == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
      {
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

  // Create Environment
  //
  Ort::Env env;

  // Create Options for the Session
  //
  Ort::SessionOptions emptySessionOption;

  // A pointer to the model
  //
  std::shared_ptr<void> rawModel;

  // Create RunOptions
  //
  Ort::RunOptions runOptions;

  // Create a Session to run
  //
  Ort::Session session;

  // Pointer to the current session object
  //
  std::unique_ptr<Ort::Session> ptrSession;

  // Create a list of input values
  //
  std::vector<Ort::Value> inputValues{};

  // Stores the input names
  //
  std::vector<char *> inputNames;

  // Stores the output names
  //
  std::vector<char *> outputNames;

  // Create a list of output values
  //
  std::vector<Ort::Value> outputValues{};

  // We own the lifetime of the input data
  //
  std::vector<std::shared_ptr<void>> inputData;

  // Keeps track of number of columns/dimensions data
  // given for predicition
  //
  size_t currInputIndex{0};
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
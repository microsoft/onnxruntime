// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "OnnxPrediction.h"

// Uses the onnxruntime to load the model
// into a session.
//
OnnxPrediction::OnnxPrediction(std::wstring& onnx_model_file)
:
rawModel{nullptr},
ptrSession{nullptr},
session{ env, onnx_model_file.c_str(), emptySessionOption},
inputNames{session.GetInputCount()},
outputNames{session.GetOutputCount()}
{
  init();
}

// Uses the onnx to seri
//
OnnxPrediction::OnnxPrediction(onnx::ModelProto& onnx_model)
:
session{nullptr}
{
  rawModel = std::shared_ptr<void>{alloc.Alloc(onnx_model.ByteSizeLong()),   
  [this](void * ptr)
      { 
      this->GetAllocator().Free(ptr); 
      }
  };

  onnx_model.SerializeToArray(rawModel.get(), static_cast<int>(onnx_model.ByteSizeLong()));

  ptrSession = std::make_unique<Ort::Session>(env, 
      rawModel.get(), 
      onnx_model.ByteSizeLong(), 
      emptySessionOption),

  inputNames.resize(ptrSession->GetInputCount());
  outputNames.resize(ptrSession->GetOutputCount());

  init();
}

// Destructor
//
OnnxPrediction::~OnnxPrediction()
{
  if (ptrSession.get() == &session)
  {
      // Ensure the session is not deleted
      // by the unique_ptr. Because it will be deleting the stack
      //
      ptrSession.release();
  }
}

// OnnxPrediction console output format
// prints the output data.
//
std::wostream& operator<<(std::wostream& out, OnnxPrediction& pred)
{
  auto pretty_print = [&out](auto ptr, Ort::Value& val)
  {
      out << L"[";
      std::wstring msg = L"";
      for(int i=0; i < val.GetTensorTypeAndShapeInfo().GetElementCount(); i++)
      {
        out << msg << ptr[i];
        msg = L", ";
      }
      out << L"]\n"; 
  };

  size_t index {0};
  for(auto& val : pred.outputValues)
  {
    out << pred.outputNames[index++] << L" = ";
    pred.ProcessOutputData(pretty_print, val);
  }
  
  return out;
}

// Used to Generate data for predict
//
void GenerateDataForInputTypeTensor(OnnxPrediction& predict,
  size_t input_index, const std::string& input_name,
  ONNXTensorElementDataType elem_type, size_t elem_count, size_t seed)
{
  (void) input_name;
  (void) input_index;
  
  auto pretty_print = [&input_name](auto raw_data)
  {
    Logger::testLog << input_name << L" = ";
    Logger::testLog << L"[";
    std::wstring msg = L"";
    for(int i=0; i < raw_data.size(); i++)
    {
      Logger::testLog << msg << raw_data[i];
      msg = L", ";
    }
    Logger::testLog << L"]\n"; 
  };

  if ( elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT)
  {
    auto raw_data = GenerateRandomData(0.0f, elem_count,seed);
    pretty_print(raw_data);
    predict << std::move(raw_data);
  }
  else if (elem_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32)
  {
    int32_t initial = 0;
    auto raw_data = GenerateRandomData(initial, elem_count,seed);
    pretty_print(raw_data);
    predict << std::move(raw_data);
  }
  else
  {
    throw std::exception("only floats are implemented");
  }
}

// Run the prediction
//
void OnnxPrediction::RunInference()
{
  Logger::testLog << L"inference starting " << Logger::endl;
  Logger::testLog.flush();

  try
  {
    ptrSession->Run(runOptions, 
      inputNames.data(), inputValues.data(), 
      inputValues.size(), outputNames.data(), outputValues.data(),
      outputNames.size());
  }
  catch(...)
  {
    Logger::testLog << L"Something went wrong in inference " << Logger::endl;
    Logger::testLog.flush();
    throw;
  }

  Logger::testLog << L"inference completed " << Logger::endl;
  Logger::testLog.flush();
}

// Print the output values of the prediction.
//
void OnnxPrediction::PrintOutputValues()
{
  Logger::testLog << L"output data:\n";
  Logger::testLog << *this;
  Logger::testLog << Logger::endl;
}

// Common initilization amongst constructors
//
void OnnxPrediction::init()
{
  // Enable telemetry events
  //
  env.EnableTelemetryEvents();

  if (!ptrSession)
  {
      // To use one consistent value
      // across the class
      //
      ptrSession.reset(&session);
  }

  // Initialize model input names
  //
  for(int i=0; i < ptrSession->GetInputCount(); i++)
  {
      inputNames[i] = ptrSession->GetInputName(i, alloc);
      inputValues.emplace_back(nullptr);
  }

  // Initialize model output names
  //
  for(int i=0; i < ptrSession->GetOutputCount(); i++)
  {
      outputNames[i] = ptrSession->GetOutputName(i, alloc);
      outputValues.emplace_back(nullptr);
  }
}

// Get the allocator used by the runtime
//
Ort::AllocatorWithDefaultOptions& OnnxPrediction::GetAllocator()
{
  return alloc;
}

void OnnxPrediction::SetupInput(
    InputGeneratorFunctionType GenerateData, 
    size_t seed)
{
  Logger::testLog << L"input data:\n";
  for(int i=0; i < ptrSession->GetInputCount(); i++)
  {
    auto inputType = ptrSession->GetInputTypeInfo(i);

    if (inputType.GetONNXType() == ONNX_TYPE_TENSOR)
    {
      auto elem_type = inputType.GetTensorTypeAndShapeInfo().GetElementType();
      auto elem_count = inputType.GetTensorTypeAndShapeInfo().GetElementCount();
      
      // This can be any generic function to generate inputs
      //
      GenerateData(*this, i , std::string(inputNames[i]), elem_type, elem_count, seed);

      // Update the seed in a predicatable way to get other values for different inputs
      //
      seed++;
    }
    else
    {
      std::cout << "Unsupported \n";
    }
  }
  Logger::testLog << Logger::endl;
}
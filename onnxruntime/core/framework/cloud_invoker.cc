// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_AZURE
#include "http_client.h"
#include "core/common/common.h"
#include "core/framework/cloud_invoker.h"
#include "core/framework/ort_value.h"

#define CHECK_TRITON_ERR(ret, msg)                                                           \
  if (!ret.IsOk()) {                                                                         \
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, msg, ", triton err: ", ret.Message().c_str()); \
  }

using namespace onnxruntime::common;

namespace onnxruntime {

namespace tc = triton::client;

const char* kAzureUri = "azure.uri";
const char* kAzureModelName = "azure.model_name";
const char* kAzureModelVer = "azure.model_version";
const char* kAzureVerbose = "azure.verbose";
const char* kAzureEndpointType = "azure.endpoint_type";
const char* kAzureAuthKey = "azure.auth_key";
const char* kAzureTriton = "triton";

CloudEndPointInvoker::CloudEndPointInvoker(const CloudEndPointConfig& config,
                                           const AllocatorPtr& allocator) : config_(config), allocator_(allocator) {
  if (!allocator_) {
    ORT_THROW("Cannot create invoker on invalid allocator");
  }
}

class AzureTritonInvoker : public CloudEndPointInvoker {
 public:
  AzureTritonInvoker(const CloudEndPointConfig& config, const AllocatorPtr& allocator);
  onnxruntime::Status Send(const CloudEndPointConfig& run_options,
                           const InlinedVector<std::string>& input_names,
                           gsl::span<const OrtValue> ort_inputs,
                           const InlinedVector<std::string>& output_names,
                           std::vector<OrtValue>& ort_outputs) const override;

 private:
  static std::string MapDataType(int32_t ort_data_type);
  onnxruntime::TensorPtr CreateTensor(const std::string& data_type, const onnxruntime::VectorInt64& dim) const;

  std::string uri_;
  std::string model_name_;
  std::string model_ver_ = "1";
  std::string verbose_ = "0";
  std::unique_ptr<triton::client::InferenceServerHttpClient> triton_client_;
};

std::string AzureTritonInvoker::MapDataType(int32_t ort_data_type) {
  std::string triton_data_type;
  switch (ort_data_type) {
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT:
      triton_data_type = "FP32";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT8:
      triton_data_type = "UINT8";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT8:
      triton_data_type = "INT8";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT16:
      triton_data_type = "UINT16";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT16:
      triton_data_type = "INT16";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT32:
      triton_data_type = "INT32";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_INT64:
      triton_data_type = "INT64";
      break;
    //do we need to support string?
    //case ONNX_NAMESPACE::TensorProto_DataType_STRING:
    //  triton_data_type = "BYTES";
    //  break;
    case ONNX_NAMESPACE::TensorProto_DataType_BOOL:
      triton_data_type = "BOOL";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_FLOAT16:
      triton_data_type = "FP16";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_DOUBLE:
      triton_data_type = "FP64";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT32:
      triton_data_type = "UINT32";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_UINT64:
      triton_data_type = "UINT64";
      break;
    case ONNX_NAMESPACE::TensorProto_DataType_BFLOAT16:
      triton_data_type = "BF16";
      break;
    default:
      break;
  }
  return triton_data_type;
}

onnxruntime::TensorPtr AzureTritonInvoker::CreateTensor(const std::string& data_type, const onnxruntime::VectorInt64& dim) const {
  if (data_type == "FP32") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<float>(), TensorShape{dim}, allocator_);
  } else if (data_type == "UINT8") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<uint8_t>(), TensorShape{dim}, allocator_);
  } else if (data_type == "INT8") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<int8_t>(), TensorShape{dim}, allocator_);
  } else if (data_type == "UINT16") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<uint16_t>(), TensorShape{dim}, allocator_);
  } else if (data_type == "INT16") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<int16_t>(), TensorShape{dim}, allocator_);
  } else if (data_type == "INT32") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<int32_t>(), TensorShape{dim}, allocator_);
  } else if (data_type == "INT64") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<int64_t>(), TensorShape{dim}, allocator_);
  } else if (data_type == "BOOL") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<bool>(), TensorShape{dim}, allocator_);
  } else if (data_type == "FP16") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<MLFloat16>(), TensorShape{dim}, allocator_);
  } else if (data_type == "FP64") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<double>(), TensorShape{dim}, allocator_);
  } else if (data_type == "UINT32") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<uint32_t>(), TensorShape{dim}, allocator_);
  } else if (data_type == "UINT64") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<uint64_t>(), TensorShape{dim}, allocator_);
  } else if (data_type == "BF16") {
    return std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<BFloat16>(), TensorShape{dim}, allocator_);
  } else {
    return {};
  }
}

AzureTritonInvoker::AzureTritonInvoker(const CloudEndPointConfig& config,
                             const AllocatorPtr& allocator) : CloudEndPointInvoker(config, allocator) {
  ReadConfig(kAzureUri, uri_);
  ReadConfig(kAzureModelName, model_name_);
  ReadConfig(kAzureModelVer, model_ver_, false);
  ReadConfig(kAzureVerbose, verbose_, false);

  auto err = tc::InferenceServerHttpClient::Create(&triton_client_, uri_, verbose_ != "0");
  if (!err.IsOk()) {
    ORT_THROW("Failed to initialize triton client, triton err: " + err.Message());
  }
}

onnxruntime::Status AzureTritonInvoker::Send(const CloudEndPointConfig& run_options,
                                        const InlinedVector<std::string>& input_names,
                                        gsl::span<const OrtValue> ort_inputs,
                                        const InlinedVector<std::string>& output_names,
                                        std::vector<OrtValue>& ort_outputs) const {
  const auto auth_key_iter = run_options.find(kAzureAuthKey);
  if (run_options.end() == auth_key_iter) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "auth key must be specified for triton client");
  }

  if (ort_inputs.size() != input_names.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Number of inputs mismatch with number of input names for triton invoker: ",
                           ort_inputs.size(), " != ", input_names.size());
  }

  auto tensor_type = DataTypeImpl::GetType<Tensor>();
  std::vector<std::unique_ptr<tc::InferInput>> triton_input_vec;
  std::vector<tc::InferInput*> triton_inputs;
  std::vector<std::unique_ptr<const tc::InferRequestedOutput>> triton_output_vec;
  std::vector<const tc::InferRequestedOutput*> triton_outputs;
  tc::Error err;

  try {
    //assemble triton inputs
    auto iter = input_names.begin();
    for (int i = 0; i < static_cast<int>(ort_inputs.size()); i++) {
      const OrtValue& ort_input = ort_inputs[i];
      if (!ort_input.IsTensor()) {
        //do we need to support tensor sequence and sparse tensor?
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Triton client only accept tensor(s) as input");
      }

      const auto& input_tensor = ort_input.Get<Tensor>();
      const auto& ort_input_shape = input_tensor.Shape();

      tc::InferInput* triton_input = {};
      std::string triton_data_type = MapDataType(input_tensor.GetElementType());
      if (triton_data_type.empty()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Triton client does not support data type: ",
                               ONNX_NAMESPACE::TensorProto_DataType_Name(input_tensor.GetElementType()));
      }

      onnxruntime::VectorInt64 dims(ort_input_shape.NumDimensions());
      ort_input_shape.CopyDims(dims.data(), ort_input_shape.NumDimensions());

      err = tc::InferInput::Create(&triton_input, *iter, dims, MapDataType(input_tensor.GetElementType()));
      triton_input_vec.emplace_back(triton_input);
      CHECK_TRITON_ERR(err, (std::string{"Failed to create triton input for "} + *iter).c_str());

      triton_inputs.push_back(triton_input);
      triton_input->AppendRaw(static_cast<const uint8_t*>(input_tensor.DataRaw()), input_tensor.SizeInBytes());
      ++iter;
    }  //for

    iter = output_names.begin();
    while (iter != output_names.end()) {
      tc::InferRequestedOutput* triton_output;
      err = tc::InferRequestedOutput::Create(&triton_output, *iter);
      triton_output_vec.emplace_back(triton_output);
      CHECK_TRITON_ERR(err, (std::string{"Failed to create triton output for "} + *iter).c_str());
      triton_outputs.push_back(triton_output);
      ++iter;
    }

    std::unique_ptr<tc::InferResult> results_ptr;
    tc::InferResult* results = {};
    tc::InferOptions options(model_name_);
    options.model_version_ = model_ver_;
    options.client_timeout_ = 0;

    tc::Headers http_headers;
    http_headers["Authorization"] = std::string{"Bearer "} + auth_key_iter->second;

    err = triton_client_->Infer(&results, options, triton_inputs, triton_outputs,
                                http_headers, tc::Parameters(),
                                tc::InferenceServerHttpClient::CompressionType::NONE,  //support compression in config?
                                tc::InferenceServerHttpClient::CompressionType::NONE);
    results_ptr.reset(results);
    CHECK_TRITON_ERR(err, "Triton client failed to do inference");

    if (ort_outputs.empty()) {
      ort_outputs.resize(output_names.size());
    }

    int output_index = 0;
    iter = output_names.begin();

    while (iter != output_names.end()) {
      std::vector<int64_t> dims;
      err = results_ptr->Shape(*iter, &dims);
      CHECK_TRITON_ERR(err, (std::string{"Failed to get shape for output "} + *iter).c_str());

      std::string type;
      err = results_ptr->Datatype(*iter, &type);
      CHECK_TRITON_ERR(err, (std::string{"Failed to get type for output "} + *iter).c_str());

      const uint8_t* raw_data = {};
      size_t raw_size;
      err = results_ptr->RawData(*iter, &raw_data, &raw_size);
      CHECK_TRITON_ERR(err, (std::string{"Failed to get raw data for output "} + *iter).c_str());

      auto output_tensor = CreateTensor(type, dims);
      if (!output_tensor) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create output tensor for output", *iter);
      }
      //how to skip memcpy?
      memcpy(output_tensor->MutableDataRaw(), raw_data, raw_size);
      ort_outputs[output_index++].Init(output_tensor.release(), tensor_type, tensor_type->GetDeleteFunc());
      ++iter;
    }
  } catch (const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Caught exception in TritonInvokder::Send", ex.what());
  }
  return Status::OK();
}

void CloudEndPointInvoker::ReadConfig(const char* config_name, std::string& config_val, bool required) {
  const auto iter = config_.find(config_name);
  if (config_.end() != iter) {
    config_val = iter->second;
  } else if (required) {
    ORT_THROW("Triton invoker failed to initialize due to missed config: ", config_name);
  }
}

Status CloudEndPointInvoker::CreateInvoker(const CloudEndPointConfig& config,
                                           const AllocatorPtr& allocator,
                                           std::unique_ptr<CloudEndPointInvoker>& invoker) {
  auto status = Status::OK();
  ORT_TRY {
    const auto iter = config.find(kAzureEndpointType);
    if (config.end() != iter) {
      if (iter->second == kAzureTriton) {
        invoker = std::make_unique<AzureTritonInvoker>(config, allocator);
        return status;
      } // else other endpoint types ...
    }
    status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Cannot create azure invoker due to missed or mismatched endpoint type.");
  }
  ORT_CATCH(const std::exception& ex) {
    ORT_HANDLE_EXCEPTION([&]() {
      status = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, ex.what());
    });
  }
  return status;
}

}  // namespace onnxruntime
#endif
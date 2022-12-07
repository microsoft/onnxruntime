// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifdef USE_CLOUD
#include "http_client.h"
#include "core/common/common.h"
#include "core/framework/cloud_invoker.h"
#include "core/framework/ort_value.h"

namespace onnxruntime {

namespace tc = triton::client;

class TritonInvoker : public CloudEndPointInvoker {
 public:
  TritonInvoker(const CloudEndPointConfig& config);
  onnxruntime::Status Send(const CloudEndPointConfig& run_options,
                           const InlinedVector<std::string>& input_names,
                           gsl::span<const OrtValue> ort_inputs,
                           const InlinedVector<std::string>& output_names,
                           std::vector<OrtValue>& ort_outputs) const noexcept override;

 private:
  static std::string MapDataType(int32_t ort_data_type);
  std::string uri_;
  std::string model_name_;
  std::string model_ver_ = "0";
  bool verbose_ = false;
  std::shared_ptr<CPUAllocator> cpu_allocator_;
  std::unique_ptr<triton::client::InferenceServerHttpClient> triton_client_;
};

std::string TritonInvoker::MapDataType(int32_t ort_data_type) {
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
    //todo - do we need to support string?
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

TritonInvoker::TritonInvoker(const CloudEndPointConfig& config) : CloudEndPointInvoker(config) {
  if (ReadConfig("cloud.uri", uri_) &&
      ReadConfig("cloud.model_name", model_name_) &&
      ReadConfig("cloud.model_ver", model_ver_, false) &&
      ReadConfig("cloud.verbose", verbose_, false)) {
    if (tc::InferenceServerHttpClient::Create(&triton_client_, uri_, verbose_).IsOk()) {
      cpu_allocator_ = std::make_shared<CPUAllocator>();
    } else {
      status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to initialize triton client");
    }
  }
}

onnxruntime::Status TritonInvoker::Send(const CloudEndPointConfig& run_options,
                                        const InlinedVector<std::string>& input_names,
                                        gsl::span<const OrtValue> ort_inputs,
                                        const InlinedVector<std::string>& output_names,
                                        std::vector<OrtValue>& ort_outputs) const noexcept {
  if (!status_.IsOK()) return status_;
  if (run_options.count("auth_token") == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "auth_token must be specified for triton client");
  }

  tc::Headers http_headers;
  http_headers["Authorization"] = std::string{"Bearer "} + run_options.at("auth_token");

  if (ort_inputs.size() != input_names.size()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Number of inputs mismatch with number of input names for triton invoker: ",
                           ort_inputs.size(), " != ", input_names.size());
  }

  auto tensor_type = DataTypeImpl::GetType<Tensor>();
  std::vector<std::unique_ptr<tc::InferInput>> triton_inputs_uptr;
  std::vector<tc::InferInput*> triton_inputs;
  std::vector<std::unique_ptr<const tc::InferRequestedOutput>> triton_outputs_uptr;
  std::vector<const tc::InferRequestedOutput*> triton_outputs;

  try {
    //assemble triton inputs
    auto iter = input_names.begin();
    for (int i = 0; i < static_cast<int>(ort_inputs.size()); i++) {
      const OrtValue& ort_input = ort_inputs[i];
      if (!ort_input.IsTensor()) {
        //todo - do we need to support tensor sequence and sparse tensor?
        return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Triton client only accept tensor(s) as input");
      }
      const auto& input_tensor = ort_input.Get<Tensor>();
      const auto& ort_input_shape = input_tensor.Shape();
      std::vector<int64_t> dims;
      dims.reserve(ort_input_shape.NumDimensions());
      for (auto dim : ort_input_shape.GetDims()) {
        dims.push_back(dim);
      }
      tc::InferInput* triton_input{};
      std::string triton_data_type = MapDataType(input_tensor.GetElementType());
      if (triton_data_type.empty()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Triton client does not support data type: ",
                               ONNX_NAMESPACE::TensorProto_DataType_Name(input_tensor.GetElementType()));
      }
      if (!tc::InferInput::Create(&triton_input, *iter, dims, MapDataType(input_tensor.GetElementType())).IsOk()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create triton input for ", *iter);
      }
      triton_inputs_uptr.emplace_back(triton_input);
      triton_inputs.push_back(triton_input);
      triton_input->AppendRaw(static_cast<const uint8_t*>(input_tensor.DataRaw()), input_tensor.SizeInBytes());
      ++iter;
    }  //for

    iter = output_names.begin();
    while (iter != output_names.end()) {
      tc::InferRequestedOutput* triton_output;
      if (!tc::InferRequestedOutput::Create(&triton_output, *iter).IsOk()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create triton output for ", *iter);
      }
      triton_outputs_uptr.emplace_back(triton_output);
      triton_outputs.push_back(triton_output);
      ++iter;
    }

    tc::InferResult* results;
    tc::InferOptions options(model_name_);
    options.model_version_ = model_ver_;
    options.client_timeout_ = 0;

    auto request_compression_algorithm = tc::InferenceServerHttpClient::CompressionType::NONE;
    auto response_compression_algorithm = tc::InferenceServerHttpClient::CompressionType::NONE;

    if (!triton_client_->Infer(&results, options, triton_inputs, triton_outputs,
                               http_headers, tc::Parameters(), request_compression_algorithm, response_compression_algorithm)
             .IsOk()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to infer inputs by triton client");
    }

    if (ort_outputs.empty()) {
      ort_outputs.resize(output_names.size());
    }
    int output_index = 0;
    std::unique_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);
    iter = output_names.begin();
    while (iter != output_names.end()) {
      std::vector<int64_t> dims;
      if (!results_ptr->Shape(*iter, &dims).IsOk()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get shape for output: ", *iter);
      }

      int32_t* output0_data;
      size_t output0_byte_size;
      if (!results_ptr->RawData(*iter, (const uint8_t**)&output0_data, &output0_byte_size).IsOk()) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get raw data for output", *iter);
      }

      TensorShape tensor_shape(dims);
      auto output_tensor = std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<float>(), tensor_shape, cpu_allocator_);
      //todo - can we skip memcpy?
      memcpy(output_tensor->MutableDataRaw(), output0_data, output0_byte_size);
      ort_outputs[output_index].Init(output_tensor.get(), tensor_type, tensor_type->GetDeleteFunc());
      output_tensor.release();
      ++iter;
      ++output_index;
    }
  } catch (const std::exception& ex) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Caught exception in TritonInvokder::Send", ex.what());
  }
  return Status::OK();
}

bool CloudEndPointInvoker::ReadConfig(const char* config_name, bool& config_val, bool required) {
  if (config_.count(config_name)) {
    config_val = config_[config_name] == "true" || config_[config_name] == "True" || config_[config_name] == "TRUE" || config_[config_name] == "1";
  } else if (required) {
    status_ = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Triton invoker failed to initialize due to missing config: ", config_name);
  }
  return status_.IsOK();
}

bool CloudEndPointInvoker::ReadConfig(const char* config_name, std::string& config_val, bool required) {
  if (config_.count(config_name)) {
    config_val = config_[config_name];
  } else if (required) {
    status_ = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Triton invoker failed to initialize due to missing config: ", config_name);
  }
  return status_.IsOK();
}

bool CloudEndPointInvoker::ReadConfig(const char* config_name, onnxruntime::InlinedVector<std::string>& config_vals, bool required) {
  if (config_.count(config_name)) {
    std::stringstream ss;
    ss << config_[config_name];
    std::string tmp;
    while (std::getline(ss, tmp, ',')) {
      config_vals.push_back(std::move(tmp));
    }
  } else if (required) {
    status_ = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Triton invoker failed to initialize due to missing config: ", config_name);
  }
  return status_.IsOK();
}

std::unique_ptr<CloudEndPointInvoker> CloudEndPointInvoker::CreateInvoker(const CloudEndPointConfig& config) {
  static const std::string endpoint_type = "cloud.endpoint_type";
  static const std::string triton_type = "triton";
  if (config.count(endpoint_type)) {
    if (config.at(endpoint_type) == triton_type) {
      return std::make_unique<TritonInvoker>(config);
    }
  }
  return {};
}
}  // namespace onnxruntime
#endif
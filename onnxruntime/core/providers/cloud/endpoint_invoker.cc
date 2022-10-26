// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "http_client.h"
#include "core/common/common.h"
#include "core/providers/cloud/endpoint_invoker.h"

namespace onnxruntime {
namespace cloud {

EndPointInvoker::~EndPointInvoker() {}

namespace tc = triton::client;

class TritonInvokder : public EndPointInvoker {
 public:
  TritonInvokder(const EndPointConfig& config);
  TensorPtrArray Send(ConstTensorPtrArray input_tensors) const override;

 private:
  bool ReadConfig(const char* config_name, std::string& config_val);
  bool ReadConfig(const char* config_name, onnxruntime::InlinedVector<std::string>& config_vals);

  std::string uri_;
  std::string key_; // access token for bearer authentication
  std::string model_name_;
  std::string model_ver_;
  onnxruntime::InlinedVector<std::string> input_names_;
  onnxruntime::InlinedVector<std::string> output_names_;
  std::shared_ptr<CPUAllocator> cpu_allocator_;
  std::unique_ptr<triton::client::InferenceServerHttpClient> triton_client_;
};

TritonInvokder::TritonInvokder(const EndPointConfig& config) : EndPointInvoker(config) {
  if (ReadConfig("uri", uri_) &&
      ReadConfig("key", key_) &&
      ReadConfig("model_name", model_name_) &&
      ReadConfig("model_ver", model_ver_) &&
      ReadConfig("input_names", input_names_) &&
      ReadConfig("output_names", output_names_)) {
    if (tc::InferenceServerHttpClient::Create(&triton_client_, uri_, true).IsOk()) {
      cpu_allocator_ = std::make_shared<CPUAllocator>();
    } else {
      status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to initialize triton client");
    }
  }
}

TensorPtrArray TritonInvokder::Send(ConstTensorPtrArray ort_inputs) const {
  if (!status_.IsOK()) return {};

  if (ort_inputs.size() != input_names_.size()) {
    status_ = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                              "Number of inputs mismatch with number of input names for triton invoker: ",
                              ort_inputs.size(), " != ", input_names_.size());
    return {};
  }

  std::vector<std::unique_ptr<tc::InferInput>> triton_inputs_uptr;
  std::vector<tc::InferInput*> triton_inputs;
  std::vector<std::unique_ptr<const tc::InferRequestedOutput>> triton_outputs_uptr;
  std::vector<const tc::InferRequestedOutput*> triton_outputs;

  try {
    //assemble triton inputs
    auto iter = input_names_.begin();
    for (const auto* ort_input : ort_inputs) {
      const auto& ort_input_shape = ort_input->Shape();
      std::vector<int64_t> dims;
      dims.reserve(ort_input_shape.NumDimensions());
      for (auto dim : ort_input_shape.GetDims()) {
        dims.push_back(dim);
      }
      tc::InferInput* triton_input{};
      //todo - support types other than fp32
      if (!tc::InferInput::Create(&triton_input, *iter, dims, "FP32").IsOk()) {
        status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create triton input for ", *iter);
        return {};
      }
      triton_inputs_uptr.emplace_back(triton_input);
      triton_inputs.push_back(triton_input);
      triton_input->AppendRaw(static_cast<const uint8_t*>(ort_input->DataRaw()), ort_input->SizeInBytes());
      ++iter;
    }  //for
    // assemble trition outputs
    iter = output_names_.begin();
    while (iter != output_names_.end()) {
      tc::InferRequestedOutput* triton_output;
      if (!tc::InferRequestedOutput::Create(&triton_output, *iter).IsOk()) {
        status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to create triton output for ", *iter);
        return {};
      }
      triton_outputs_uptr.emplace_back(triton_output);
      triton_outputs.push_back(triton_output);
      ++iter;
    }

    tc::Headers http_headers;  //todo - move this out of Send()
    http_headers["Authorization"] = std::string{"Bearer "} + key_;

    tc::InferResult* results;
    tc::InferOptions options(model_name_);
    options.model_version_ = model_ver_;
    options.client_timeout_ = 0;

    auto request_compression_algorithm = tc::InferenceServerHttpClient::CompressionType::NONE;
    auto response_compression_algorithm = tc::InferenceServerHttpClient::CompressionType::NONE;

    if (!triton_client_->Infer(&results, options, triton_inputs, triton_outputs,
                               http_headers, tc::Parameters(), request_compression_algorithm, response_compression_algorithm)
             .IsOk()) {
      status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to infer inputs by triton client");
      return {};
    }

    TensorPtrArray output_tensor_ptr;
    std::unique_ptr<tc::InferResult> results_ptr;
    results_ptr.reset(results);
    iter = output_names_.begin();
    while (iter != output_names_.end()) {
      std::vector<int64_t> dims;
      if (!results_ptr->Shape(*iter, &dims).IsOk()) {
        status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get shape for output: ", *iter);
        return {};
      }

      int32_t* output0_data;
      size_t output0_byte_size;
      if (!results_ptr->RawData(iter->c_str(), (const uint8_t**)&output0_data, &output0_byte_size).IsOk()) {
        status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Failed to get raw data for output", *iter);
      }
      TensorShape tensor_shape(dims);
      output_tensor_ptr.push_back(std::make_unique<Tensor>(onnxruntime::DataTypeImpl::GetType<float>(), tensor_shape, cpu_allocator_));
      //todo - can we skip data copy?
      memcpy(output_tensor_ptr.back()->MutableDataRaw(), output0_data, output0_byte_size);
      ++iter;
    }
    return output_tensor_ptr;
  } catch (...) {
    status_ = ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Caught exception in TritonInvokder::Send");
  }
  return {};
}

bool TritonInvokder::ReadConfig(const char* config_name, std::string& config_val) {
  if (config_.count(config_name)) {
    config_val = config_[config_name];
  } else {
    status_ = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Triton invoker failed to initialize due to missing config: ", config_name);
  }
  return status_.IsOK();
}

bool TritonInvokder::ReadConfig(const char* config_name, onnxruntime::InlinedVector<std::string>& config_vals) {
  if (config_.count(config_name)) {
    std::stringstream ss;
    ss << config_[config_name];
    std::string tmp;
    while (std::getline(ss, tmp, ',')) {
      config_vals.push_back(std::move(tmp));
    }
  } else {
    status_ = ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Triton invoker failed to initialize due to missing config: ", config_name);
  }
  return status_.IsOK();
}

std::unique_ptr<EndPointInvoker> EndPointInvoker::CreateInvoker(const EndPointConfig& config) {
  static const std::string endpoint_type = "endpoint_type";
  static const std::string triton_type = "triton";
  if (config.count(endpoint_type)) {
    if (config.at(endpoint_type) == triton_type) {
      return std::make_unique<TritonInvokder>(config);
    }
  }
  return {};
}

}  // namespace cloud
}  // namespace onnxruntime
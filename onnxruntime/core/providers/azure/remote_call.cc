// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "curl/curl.h"
#include "remote_call.h"
#include "core/framework/tensorprotoutils.h"
#include "azure_execution_provider.h"
#include "http_client.h"

namespace tc = triton::client;

namespace onnxruntime {

ONNX_OPERATOR_KERNEL_EX(
    RemoteCall,  //name
    kMSDomain,
    1,
    kAzureExecutionProvider,
    KernelDefBuilder(),
    azure::RemoteCall);

namespace azure {

/*
struct RecvData {
  char* recv_data = {};
  size_t recv_size = {};
};

static size_t RecvCallback(void* data, size_t size, size_t nmemb, void* userp) {
  size_t realsize = size * nmemb;
  struct RecvData* recv_data = (struct RecvData*)userp;

  char* ptr = (char*)realloc(recv_data->recv_data, recv_data->recv_size + realsize + 1);
  if (ptr == NULL) {
    return 0;
  }

  recv_data->recv_data = ptr;
  memcpy(&(recv_data->recv_data[recv_data->recv_size]), data, realsize);
  recv_data->recv_size += realsize;
  recv_data->recv_data[recv_data->recv_size] = 0;

  return realsize;
}

struct SendData {
  const char* send_data{};
  size_t send_size{};
};

static size_t SendCallback(char* dest, size_t size, size_t nmemb, void* userp) {
  struct SendData* send_data = (struct SendData*)userp;
  size_t buffer_size = size * nmemb;

  if (send_data->send_size) {
    size_t copy_this_much = send_data->send_size;
    if (copy_this_much > buffer_size)
      copy_this_much = buffer_size;
    memcpy(dest, send_data->send_data, copy_this_much);

    send_data->send_size += copy_this_much;
    send_data->send_size -= copy_this_much;
    return copy_this_much;
  }
  return 0;
}

void InvokeHttpEndPoint(const char* uri, const char*, const char* input, size_t input_len, void** output, size_t& output_len) {
  CURL* curl = {};
  CURLcode res = {};
  SendData sd;
  sd.send_data = input;
  sd.send_size = input_len;
  curl = curl_easy_init();

  if (curl) {
    curl_easy_setopt(curl, CURLOPT_URL, uri);
    //curl_easy_setopt(curl, CURLOPT_XOAUTH2_BEARER, key);
    //curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BEARER);
    curl_easy_setopt(curl, CURLOPT_POST, 1L);
    curl_easy_setopt(curl, CURLOPT_READFUNCTION, SendCallback);
    curl_easy_setopt(curl, CURLOPT_READDATA, &sd);
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
    curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, input_len);

    struct RecvData rd;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, RecvCallback);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &rd);

    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      throw std::runtime_error(curl_easy_strerror(res));
    } else {
      *output = rd.recv_data;
      output_len = rd.recv_size;
    }
  } else {
    throw std::runtime_error("Failed to initialize http client for AML EP.");
  }
}

common::Status RemoteCall::Compute(OpKernelContext* context) const {
  std::string input_string;
  for (int i = 0; i < context->InputCount(); i++) {
    const auto* input_value = context->GetInputOrtValue(i);
    ORT_ENFORCE(input_value->IsTensor());  // todo - suppport other types
    const auto& input_tensor = input_value->Get<Tensor>();
    ONNX_NAMESPACE::TensorProto input_proto = onnxruntime::utils::TensorToTensorProto(input_tensor, this->Node().InputDefs()[i]->Name());
    input_proto.AppendToString(&input_string);
  }
  size_t output_len{};
  void* output{};
  
  auto azure_ep = static_cast<const AzureExecutionProvider*>(this->Info().GetExecutionProvider());

  InvokeHttpEndPoint(azure_ep->info.end_point.c_str(),
                     azure_ep->info.access_token.c_str(),
                     input_string.c_str(),
                     input_string.size(),
                     &output, output_len);

  ORT_ENFORCE(output && output_len);  // todo - capture http error code and return
  std::string output_string{static_cast<char*>(output), output_len};
  std::stringstream in_stream{output_string};
  ONNX_NAMESPACE::TensorProto output_proto;
  //std::stringstream in_stream {
  //  std::string {
  //        static_cast<char*>(input_string.data()), input_string.size()}};
  int output_index{0};
  while (output_proto.ParsePartialFromIstream(&in_stream)) {
    TensorShape shape{output_proto.dims()};
    size_t total_bytes = shape.Size() << 2;
    std::cout << "total bytes: " << total_bytes << std::endl;
    auto* output_tensor = context->Output(output_index++, shape);
    memcpy(output_tensor->MutableDataRaw(), output_proto.raw_data().c_str(), output_proto.ByteSizeLong());
    break; //todo - process multiple tensor outputs
  }
  output_len = 0;
  free(output);
  return common::Status::OK();
}*/

common::Status RemoteCall::Compute(OpKernelContext* context) const {
  std::vector<std::unique_ptr<tc::InferInput>> triton_inputs;
  std::vector<tc::InferInput*> triton_raw_inputs;
  for (int i = 0; i < context->InputCount(); i++) {
    //todo - replace ORT_ENFORCE to allow for failures here?
    ORT_ENFORCE(i < input_names_.size(), "Invalid input index");
    const auto* input_value = context->GetInputOrtValue(i);
    ORT_ENFORCE(input_value->IsTensor(), "Remote call only support tensor for now");
    const auto& input_tensor = input_value->Get<Tensor>();
    const auto& input_dims = input_tensor.Shape().GetDims();
    tc::InferInput* triton_input = {};
    ORT_ENFORCE(tc::InferInput::Create(&triton_input, input_names_[i], {input_dims.begin(), input_dims.end()}, "FLOAT32").IsOk(),
                "Failed to construct triton input for tenstor: ", input_names_[i]);
    triton_inputs.emplace_back(triton_input);
    triton_raw_inputs.emplace_back(triton_input);
    triton_input->AppendRaw(static_cast<const uint8_t*>(input_tensor.DataRaw()), input_tensor.SizeInBytes());
  }

  tc::HttpSslOptions ssl_options;
 /* ssl_options.verify_peer = verify_peer;
  ssl_options.verify_host = verify_host;
  ssl_options.ca_info = cacerts;
  ssl_options.cert = certfile;
  ssl_options.key = keyfile;*/
  // Create a InferenceServerHttpClient instance to communicate with the
  // server using HTTP protocol.
  auto azure_ep = static_cast<const AzureExecutionProvider*>(this->Info().GetExecutionProvider());
  std::unique_ptr<tc::InferenceServerHttpClient> client;
  ORT_ENFORCE(tc::InferenceServerHttpClient::Create(&client,
                                                    azure_ep->info.end_point,
                                                    /*verbose*/ false,
                                                    ssl_options)
                  .IsOk(),
              "unable to create triton http client");

  tc::Headers http_headers;
  http_headers["Authorization"] = std::string{"Bearer "} + azure_ep->info.access_token;
  std::vector<const tc::InferRequestedOutput*> outputs = {};
  tc::InferResult* results;
  tc::InferOptions options("model7"); //todo - this has to be configurable
  options.model_version_ = "1";//todo - this has to be configurable
  options.client_timeout_ = 8192;
  auto request_compression_algorithm = tc::InferenceServerHttpClient::CompressionType::NONE;
  auto response_compression_algorithm = tc::InferenceServerHttpClient::CompressionType::NONE;

  ORT_ENFORCE(client->Infer(&results, options, triton_raw_inputs, outputs, http_headers, tc::Parameters(),
                            request_compression_algorithm, response_compression_algorithm)
                  .IsOk(),
              "unable to call triton server");

  std::unique_ptr<tc::InferResult> results_ptr{results};
  int output_index = 0;
  for (const auto& output_name : output_names_) {
    std::vector<int64_t> output_shape;
    ORT_ENFORCE(results_ptr->Shape(output_name, &output_shape).IsOk(), "failed to fetch output shape for: ", output_name);
    auto* output_tensor = context->Output(output_index++, output_shape);
    uint8_t* output_raw_data{};
    size_t output_raw_data_byte_size{};
    ORT_ENFORCE(
        results_ptr->RawData(
                       output_name.c_str(),
                       (const uint8_t**)(&output_raw_data),
                       &output_raw_data_byte_size)
            .IsOk(),
        "unable to get result data for: ", output_name);
    memcpy(output_tensor->MutableDataRaw(), output_raw_data, output_raw_data_byte_size);
  }
  return common::Status::OK();
}

}  // namespace azure
}  // namespace onnxruntime
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//#include "curl/curl.h"
#include "remote.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    RemoteCall,  //name
    kMSDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder(),
    RemoteCall);

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
  InvokeHttpEndPoint(uri_.c_str(), key_.c_str(), input_string.c_str(), input_string.size(), &output, output_len);
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
      memcpy(output_tensor->MutableDataRaw(), output_proto.raw_data().c_str(), total_bytes);
  }
  output_len = 0;
  free(output);
  return common::Status::OK();
}*/

common::Status RemoteCall::Compute(OpKernelContext*) const {
  return common::Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
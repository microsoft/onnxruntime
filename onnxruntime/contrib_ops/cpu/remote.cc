// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "curl/curl.h"
#include "remote.h"
#include "core/framework/tensorprotoutils.h"

namespace onnxruntime {
namespace contrib {

struct memory {
  char* response = {};
  size_t size = {};
};

static size_t cb(void* data, size_t size, size_t nmemb, void* userp) {
  size_t realsize = size * nmemb;
  struct memory* mem = (struct memory*)userp;

  char* ptr = (char*)realloc(mem->response, mem->size + realsize + 1);
  if (ptr == NULL) {
    return 0; /* out of memory! */
  }

  mem->response = ptr;
  memcpy(&(mem->response[mem->size]), data, realsize);
  mem->size += realsize;
  mem->response[mem->size] = 0;

  return realsize;
}

void InvokeHttpEndPoint(const char* uri, const char* key, const char* input, size_t input_len, void** output, size_t& output_len) {
  CURL* curl = {};
  CURLcode res = {};
  curl_mime* form = {};
  curl_mimepart* field = {};
  curl = curl_easy_init();
  if (curl) {
    curl_easy_setopt(curl, CURLOPT_VERBOSE, 0L);
    curl_easy_setopt(curl, CURLOPT_URL, uri);
    curl_easy_setopt(curl, CURLOPT_XOAUTH2_BEARER, key);
    curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BEARER);

    form = curl_mime_init(curl);

    /* Fill in the file upload field */
    field = curl_mime_addpart(form);
    curl_mime_name(field, "data");
    curl_mime_filename(field, "data");
    curl_mime_data(field, input, input_len);
    curl_mime_type(field, "application/octet-stream");

    curl_easy_setopt(curl, CURLOPT_MIMEPOST, form);

    struct memory chunk;
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, cb);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void*)&chunk);
    res = curl_easy_perform(curl);
    if (res != CURLE_OK) {
      throw std::runtime_error(curl_easy_strerror(res));
    } else {
      *output = chunk.response;
      output_len = chunk.size;
    }
    curl_easy_cleanup(curl);
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
  std::stringstream in_stream{std::string{
      static_cast<char*>(output), output_len}};

  int output_index{0};
  while (true) {
    ONNX_NAMESPACE::TensorProto output_proto;
    if (output_proto.ParseFromIstream(&in_stream)) {
      auto* output_tensor = context->Output(output_index++, TensorShape{output_proto.dims()});
      memcpy(output_tensor->MutableDataRaw(), output_proto.raw_data().c_str(), output_proto.ByteSizeLong());
    } else {
      break;
    }
  }
  output_len = 0;
  free(output);
  return common::Status::OK();
}

}  // namespace contrib
}  // namespace onnxruntime
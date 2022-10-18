// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "curl/curl.h"
#include "core/common/common.h"
#include "core/providers/cloud/endpoint_invoker.h"

namespace onnxruntime {
namespace cloud {

EndPointType EndPointInvoker::MapType(const std::string& type) {
  if (type == "Rest") return EndPointType::Rest;
  return EndPointType::Unknown;
}

std::unique_ptr<EndPointInvoker> EndPointInvoker::CreateInvoker(EndPointType type, const EndPointConfig& config) {
  std::unique_ptr<EndPointInvoker> invoker;
  switch (type) {
    case onnxruntime::cloud::Rest:
      invoker = std::make_unique<RestInvoker>(config);
      break;
    default:
      break;
  }
  return invoker;
}

static size_t RecvCallback(void* raw_data, size_t size, size_t nmemb, void* userp) {
  size_t realsize = size * nmemb;
  Data* recv_data = static_cast<Data*>(userp);

  char* ptr = (char*)realloc(recv_data->content, recv_data->size_in_byte + realsize + 1);
  if (ptr == NULL) {
    return 0;
  }

  recv_data->content = ptr;
  memcpy(&(recv_data->content[recv_data->size_in_byte]), raw_data, realsize);
  recv_data->size_in_byte += realsize;
  recv_data->content[recv_data->size_in_byte] = 0;

  return realsize;
}

static size_t SendCallback(char* dest, size_t size, size_t nmemb, void* userp) {
  auto send_data = static_cast<Data*>(userp);
  size_t buffer_size = size * nmemb;

  if (send_data->size_in_byte) {
    size_t copy_this_much = send_data->size_in_byte;
    if (copy_this_much > buffer_size) {
      copy_this_much = buffer_size;
    }
    memcpy(dest, send_data->content, copy_this_much);
    send_data->size_in_byte -= copy_this_much;
    return copy_this_much;
  }
  return 0;
}

RestInvoker::RestInvoker(const EndPointConfig& config) : config_(config) {}

Data RestInvoker::Send(Data request) const {
  CURL* curl = {};
  CURLcode res = {};
  ORT_ENFORCE(config_.contains("uri"), "must specify uri for sending the data");
  curl_easy_setopt(curl, CURLOPT_URL, config_.at("uri").c_str());
  if (config_.contains("key")) {
    curl_easy_setopt(curl, CURLOPT_XOAUTH2_BEARER, config_.at("key").c_str());
    //todo - make this configurable
    curl_easy_setopt(curl, CURLOPT_HTTPAUTH, CURLAUTH_BEARER);
  }
  curl_easy_setopt(curl, CURLOPT_POST, 1L);
  curl_easy_setopt(curl, CURLOPT_READFUNCTION, SendCallback);
  curl_easy_setopt(curl, CURLOPT_READDATA, &request);
  curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);
  curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, request.size_in_byte);

  Data response;
  curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, RecvCallback);
  curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response);
  //todo - allow for http failures
  ORT_ENFORCE(curl_easy_perform(curl) == CURLE_OK,
              "Failed to send http request, curl report error:",
              curl_easy_strerror(res));
  return response;
}

}  // namespace cloud
}  // namespace onnxruntime
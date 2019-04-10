// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <google/protobuf/stubs/status.h>
#include "gtest/gtest.h"
#include "hosting/http/core/context.h"
#include "hosting/http/util.h"

namespace onnxruntime {
namespace hosting {
namespace test {

namespace protobufutil = google::protobuf::util;

TEST(PositiveTests, GetRequestContentTypeJson) {
  HttpContext context;
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  request.set(http::field::content_type, "application/json");
  context.request = request;

  auto result = GetRequestContentType(context);
  EXPECT_EQ(result, SupportedContentType::Json);
}

TEST(PositiveTests, GetRequestContentTypeRawData) {
  HttpContext context;
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  request.set(http::field::content_type, "application/octet-stream");
  context.request = request;

  auto result = GetRequestContentType(context);
  EXPECT_EQ(result, SupportedContentType::PbByteArray);

  context.request.set(http::field::content_type, "application/vnd.google.protobuf");
  result = GetRequestContentType(context);
  EXPECT_EQ(result, SupportedContentType::PbByteArray);

  context.request.set(http::field::content_type, "application/x-protobuf");
  result = GetRequestContentType(context);
  EXPECT_EQ(result, SupportedContentType::PbByteArray);
}

TEST(NegativeTests, GetRequestContentTypeUnknown) {
  HttpContext context;
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  request.set(http::field::content_type, "text/plain");
  context.request = request;

  auto result = GetRequestContentType(context);
  EXPECT_EQ(result, SupportedContentType::Unknown);
}

TEST(NegativeTests, GetRequestContentTypeMissing) {
  HttpContext context;
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  context.request = request;

  auto result = GetRequestContentType(context);
  EXPECT_EQ(result, SupportedContentType::Unknown);
}

TEST(PositiveTests, GetResponseContentTypeJson) {
  HttpContext context;
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  request.set(http::field::accept, "application/json");
  context.request = request;

  auto result = GetResponseContentType(context);
  EXPECT_EQ(result, SupportedContentType::Json);
}

TEST(PositiveTests, GetResponseContentTypeRawData) {
  HttpContext context;
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  request.set(http::field::accept, "application/octet-stream");
  context.request = request;

  auto result = GetResponseContentType(context);
  EXPECT_EQ(result, SupportedContentType::PbByteArray);

  context.request.set(http::field::accept, "application/vnd.google.protobuf");
  result = GetResponseContentType(context);
  EXPECT_EQ(result, SupportedContentType::PbByteArray);

  context.request.set(http::field::accept, "application/x-protobuf");
  result = GetResponseContentType(context);
  EXPECT_EQ(result, SupportedContentType::PbByteArray);
}

TEST(NegativeTests, GetResponseContentTypeAny) {
  HttpContext context;
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  request.set(http::field::accept, "*/*");
  context.request = request;

  auto result = GetResponseContentType(context);
  EXPECT_EQ(result, SupportedContentType::PbByteArray);
}

TEST(NegativeTests, GetResponseContentTypeUnknown) {
  HttpContext context;
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  request.set(http::field::accept, "text/plain");
  context.request = request;

  auto result = GetResponseContentType(context);
  EXPECT_EQ(result, SupportedContentType::Unknown);
}

TEST(NegativeTests, GetResponseContentTypeMissing) {
  HttpContext context;
  http::request<http::string_body, http::basic_fields<std::allocator<char>>> request{};
  context.request = request;

  auto result = GetResponseContentType(context);
  EXPECT_EQ(result, SupportedContentType::PbByteArray);
}

}  // namespace test
}  // namespace hosting
}  // namespace onnxruntime
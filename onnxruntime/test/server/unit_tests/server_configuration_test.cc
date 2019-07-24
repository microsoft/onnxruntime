// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "gmock/gmock.h"
#include "server/server_configuration.h"

namespace onnxruntime {
namespace server {
namespace test {

TEST(ConfigParsingTests, AllArgs) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("--model_path"), const_cast<char*>("testdata/mul_1.pb"),
      const_cast<char*>("--address"), const_cast<char*>("4.4.4.4"),
      const_cast<char*>("--http_port"), const_cast<char*>("80"),
      const_cast<char*>("--num_http_threads"), const_cast<char*>("1"),
      const_cast<char*>("--log_level"), const_cast<char*>("info")};

  onnxruntime::server::ServerConfiguration config{};
  Result res = config.ParseInput(11, test_argv);
  EXPECT_EQ(res, Result::ContinueSuccess);
  EXPECT_EQ(config.model_path, "testdata/mul_1.pb");
  EXPECT_EQ(config.address, "4.4.4.4");
  EXPECT_EQ(config.http_port, 80);
  EXPECT_EQ(config.num_http_threads, 1);
  EXPECT_EQ(config.logging_level, onnxruntime::logging::Severity::kINFO);
}

TEST(ConfigParsingTests, Defaults) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("--model"), const_cast<char*>("testdata/mul_1.pb"),
      const_cast<char*>("--num_http_threads"), const_cast<char*>("3")};

  onnxruntime::server::ServerConfiguration config{};
  Result res = config.ParseInput(5, test_argv);
  EXPECT_EQ(res, Result::ContinueSuccess);
  EXPECT_EQ(config.model_path, "testdata/mul_1.pb");
  EXPECT_EQ(config.address, "0.0.0.0");
  EXPECT_EQ(config.http_port, 8001);
  EXPECT_EQ(config.num_http_threads, 3);
  EXPECT_EQ(config.logging_level, onnxruntime::logging::Severity::kINFO);
}

TEST(ConfigParsingTests, Help) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("--help")};

  onnxruntime::server::ServerConfiguration config{};
  auto res = config.ParseInput(2, test_argv);
  EXPECT_EQ(res, Result::ExitSuccess);
}

TEST(ConfigParsingTests, NoModelArg) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("--num_http_threads"), const_cast<char*>("3")};

  onnxruntime::server::ServerConfiguration config{};
  Result res = config.ParseInput(3, test_argv);
  EXPECT_EQ(res, Result::ExitFailure);
}

TEST(ConfigParsingTests, ModelNotFound) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("--model_path"), const_cast<char*>("does/not/exist"),
      const_cast<char*>("--address"), const_cast<char*>("4.4.4.4"),
      const_cast<char*>("--http_port"), const_cast<char*>("80"),
      const_cast<char*>("--num_http_threads"), const_cast<char*>("1")};

  onnxruntime::server::ServerConfiguration config{};
  Result res = config.ParseInput(9, test_argv);
  EXPECT_EQ(res, Result::ExitFailure);
}

TEST(ConfigParsingTests, WrongLoggingLevel) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("--log_level"), const_cast<char*>("not a logging level"),
      const_cast<char*>("--model_path"), const_cast<char*>("testdata/mul_1.pb"),
      const_cast<char*>("--address"), const_cast<char*>("4.4.4.4"),
      const_cast<char*>("--http_port"), const_cast<char*>("80"),
      const_cast<char*>("--num_http_threads"), const_cast<char*>("1")};

  onnxruntime::server::ServerConfiguration config{};
  Result res = config.ParseInput(11, test_argv);
  EXPECT_EQ(res, Result::ExitFailure);
}

}  // namespace test
}  // namespace server
}  // namespace onnxruntime
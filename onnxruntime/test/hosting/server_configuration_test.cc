// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"

#include "gmock/gmock.h"
#include "hosting/server_configuration.h"

namespace onnxruntime {
namespace hosting {
namespace test {

TEST(PositiveTests, ConfigParsingFullArgs) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("--model_path"), const_cast<char*>("testdata/mul_1.pb"),
      const_cast<char*>("--address"), const_cast<char*>("4.4.4.4"),
      const_cast<char*>("--http_port"), const_cast<char*>("80"),
      const_cast<char*>("--num_http_threads"), const_cast<char*>("1")};

  onnxruntime::hosting::ServerConfiguration config{};
  Result res = config.ParseInput(9, test_argv);
  EXPECT_EQ(res, Result::ContinueSuccess);
  EXPECT_EQ(config.model_path, "testdata/mul_1.pb");
  EXPECT_EQ(config.address, "4.4.4.4");
  EXPECT_EQ(config.http_port, 80);
  EXPECT_EQ(config.num_http_threads, 1);
}

TEST(PositiveTests, ConfigParsingShortArgs) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("-m"), const_cast<char*>("testdata/mul_1.pb"),
      const_cast<char*>("-a"), const_cast<char*>("4.4.4.4"),
      const_cast<char*>("--http_port"), const_cast<char*>("5001"),
      const_cast<char*>("--num_http_threads"), const_cast<char*>("2")};

  onnxruntime::hosting::ServerConfiguration config{};
  Result res = config.ParseInput(9, test_argv);
  EXPECT_EQ(res, Result::ContinueSuccess);
  EXPECT_EQ(config.model_path, "testdata/mul_1.pb");
  EXPECT_EQ(config.address, "4.4.4.4");
  EXPECT_EQ(config.http_port, 5001);
  EXPECT_EQ(config.num_http_threads, 2);
}

TEST(PositiveTests, ConfigParsingDefaults) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("-m"), const_cast<char*>("testdata/mul_1.pb"),
      const_cast<char*>("--num_http_threads"), const_cast<char*>("3")};

  onnxruntime::hosting::ServerConfiguration config{};
  Result res = config.ParseInput(5, test_argv);
  EXPECT_EQ(res, Result::ContinueSuccess);
  EXPECT_EQ(config.model_path, "testdata/mul_1.pb");
  EXPECT_EQ(config.address, "0.0.0.0");
  EXPECT_EQ(config.http_port, 8001);
  EXPECT_EQ(config.num_http_threads, 3);
}

TEST(PositiveTests, ConfigParsingHelp) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("--help")};

  onnxruntime::hosting::ServerConfiguration config{};
  auto res = config.ParseInput(2, test_argv);
  EXPECT_EQ(res, Result::ExitSuccess);
}

TEST(NegativeTests, ConfigParsingNoModelArg) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("--num_http_threads"), const_cast<char*>("3")};

  onnxruntime::hosting::ServerConfiguration config{};
  Result res = config.ParseInput(3, test_argv);
  EXPECT_EQ(res, Result::ExitFailure);
}

TEST(PositiveTests, ConfigParsingModelNotFound) {
  char* test_argv[] = {
      const_cast<char*>("/path/to/binary"),
      const_cast<char*>("--model_path"), const_cast<char*>("does/not/exist"),
      const_cast<char*>("--address"), const_cast<char*>("4.4.4.4"),
      const_cast<char*>("--http_port"), const_cast<char*>("80"),
      const_cast<char*>("--num_http_threads"), const_cast<char*>("1")};

  onnxruntime::hosting::ServerConfiguration config{};
  Result res = config.ParseInput(9, test_argv);
  EXPECT_EQ(res, Result::ExitFailure);
}

}  // namespace test
}  // namespace hosting
}  // namespace onnxruntime
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <atomic>
#include <vector>
#include <core/common/common.h>
#include <core/session/inference_session.h>

#include <experimental/filesystem>
#ifdef _MSC_VER
#include <filesystem>
#endif

class ITestCase;
class TestCaseResult;
template <typename T>
class FixedCountFinishCallbackImpl;
using FixedCountFinishCallback = FixedCountFinishCallbackImpl<TestCaseResult>;

class SessionFactory {
 private:
  const std::vector<std::string> providers_;
  bool enable_mem_pattern_ = true;
  bool enable_cpu_mem_arena_ = true;

 public:
  SessionFactory(std::vector<std::string>&& providers, bool enable_mem_pattern, bool enable_cpu_mem_arena)
      : providers_(std::move(providers)),
        enable_mem_pattern_(enable_mem_pattern),
        enable_cpu_mem_arena_(enable_cpu_mem_arena) {}

  //Create an initialized session from a given model url
  onnxruntime::common::Status Create(std::shared_ptr<::onnxruntime::InferenceSession>& sess,
                                     const std::experimental::filesystem::v1::path& model_url,
                                     const std::string& logid) const;

  bool enable_sequential_execution = true;
  int session_thread_pool_size = 0;
};

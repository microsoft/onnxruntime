// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Tests for the CUDA plugin EP profiling integration.
// Uses InferenceSessionWrapper to directly query the Profiler's EP start status,
// which propagates the OrtStatus* returned by the plugin's StartProfiling C API.
// This distinguishes "profiling not compiled in / CUPTI unavailable" (skip)
// from "profiling started but no kernel events appeared" (regression/fail).

#if defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP) && defined(ENABLE_CUDA_PROFILING)

#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <cuda_runtime_api.h>
#include <gsl/gsl>
#include <gtest/gtest.h>
#include "nlohmann/json.hpp"

#include "core/framework/data_types.h"
#include "core/framework/error_code_helper.h"
#include "core/framework/run_options.h"
#include "core/framework/tensor.h"
#include "core/session/abi_session_options_impl.h"
#include "core/session/inference_session.h"
#include "core/session/onnxruntime_cxx_api.h"
#include "core/session/ort_env.h"
#include "core/session/utils.h"
#include "test/util/include/asserts.h"
#include "test/util/include/file_util.h"
#include "test/util/include/inference_session_wrapper.h"

extern std::unique_ptr<Ort::Env> ort_env;

namespace onnxruntime {
namespace test {
namespace {

constexpr const char* kCudaPluginEpName = "CudaPluginExecutionProvider";
constexpr const char* kRegistrationName = "CudaPluginProfilingTest";

std::filesystem::path GetCudaPluginLibraryPath() {
  return GetSharedLibraryFileName(ORT_TSTR("onnxruntime_providers_cuda_plugin"));
}

// Get the internal OrtEnv from the C++ Ort::Env wrapper.
OrtEnv& GetOrtEnv() {
  return *static_cast<OrtEnv*>(*ort_env);
}

// RAII handle that registers/unregisters the CUDA plugin EP library.
// Uses the C API directly to avoid exceptions (the plugin DLL may fail to load
// if CUPTI is not on PATH due to the hard import dependency).
// OrtStatus* returns are wrapped in Ort::Status for leak-safe RAII.
class ScopedCudaPluginRegistration {
 public:
  ScopedCudaPluginRegistration(Ort::Env& env, const char* registration_name)
      : env_(env), name_(registration_name) {
    auto lib_path = GetCudaPluginLibraryPath();
    if (!std::filesystem::exists(lib_path)) {
      load_error_ = "Plugin library not found: " + lib_path.string();
      return;
    }
    Ort::Status status(Ort::GetApi().RegisterExecutionProviderLibrary(
        env_, name_.c_str(), lib_path.c_str()));
    if (!status.IsOK()) {
      load_error_ = status.GetErrorMessage();
      return;
    }
    available_ = true;
  }

  ~ScopedCudaPluginRegistration() {
    if (available_) {
      Ort::Status status(Ort::GetApi().UnregisterExecutionProviderLibrary(
          env_, name_.c_str()));
      ORT_UNUSED_PARAMETER(status);  // intentionally ignore unregister errors during teardown
    }
  }

  bool IsAvailable() const { return available_; }
  const std::string& LoadError() const { return load_error_; }

  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(ScopedCudaPluginRegistration);

 private:
  Ort::Env& env_;
  std::string name_;
  std::string load_error_;
  bool available_ = false;
};

// Find the CUDA plugin EP device after registration.
Ort::ConstEpDevice FindCudaPluginDevice(Ort::Env& env) {
  auto ep_devices = env.GetEpDevices();
  for (const auto& device : ep_devices) {
    if (strcmp(device.EpName(), kCudaPluginEpName) == 0) {
      return device;
    }
  }
  return Ort::ConstEpDevice{nullptr};
}

}  // namespace

class CudaPluginProfilingTest : public ::testing::Test {
 protected:
  void SetUp() override {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
      GTEST_SKIP() << "No CUDA device available.";
    }

    registration_ = std::make_unique<ScopedCudaPluginRegistration>(
        *ort_env, kRegistrationName);
    if (!registration_->IsAvailable()) {
      GTEST_SKIP() << "CUDA plugin EP library not available: "
                   << registration_->LoadError();
    }

    cuda_device_ = FindCudaPluginDevice(*ort_env);
    if (!cuda_device_) {
      GTEST_SKIP() << "No CUDA plugin EP device found after registration.";
    }
  }

  void TearDown() override {
    registration_.reset();
    cudaDeviceSynchronize();
  }

  std::unique_ptr<ScopedCudaPluginRegistration> registration_;
  Ort::ConstEpDevice cuda_device_{nullptr};
};

// Test that session-level profiling produces valid JSON with GPU Kernel events
// when CUPTI is functional. Uses InferenceSessionWrapper to directly query the
// profiler's EP start status for definitive pass/skip/fail decisions.
TEST_F(CudaPluginProfilingTest, SessionProfiling_ProducesValidProfile) {
  const ORTCHAR_T* model_path = ORT_TSTR("testdata/matmul_1.onnx");

  // Set up session options with the plugin EP and profiling enabled.
  OrtSessionOptions ort_options;

  const OrtEpDevice* device_ptr = static_cast<const OrtEpDevice*>(cuda_device_);
  auto ep_devices_span = gsl::make_span(&device_ptr, 1);

  std::unique_ptr<IExecutionProviderFactory> factory;
  ASSERT_STATUS_OK(CreateIExecutionProviderFactoryForEpDevices(
      GetOrtEnv().GetEnvironment(), ep_devices_span, factory));
  ort_options.provider_factories.push_back(std::move(factory));

  auto profile_prefix = std::filesystem::temp_directory_path() / ORT_TSTR("cuda_plugin_profiling_test");
  ort_options.value.enable_profiling = true;
  ort_options.value.profile_file_prefix = profile_prefix.native();

  // Create session via InferenceSessionWrapper for internal access.
  InferenceSessionWrapper session(ort_options.value, GetOrtEnv().GetEnvironment());
  ASSERT_STATUS_OK(session.Load(model_path));

  OrtStatus* init_status = InitializeSession(&ort_options, session);
  ASSERT_STATUS_OK(ToStatusAndRelease(init_status));

  // Check EP profiling status. Three scenarios:
  // 1. No EP profiler registered (plugin not built with profiling) → skip
  // 2. EP profiler registered but StartProfiling failed (CUPTI unavailable/blocked) → skip
  // 3. EP profiler started successfully → kernel events MUST appear
  const auto& profiler = session.GetProfiling();
  if (!profiler.HasEpProfilers()) {
    GTEST_SKIP() << "Plugin EP did not register a profiler. "
                 << "It may have been built without ENABLE_CUDA_PROFILING.";
  }

  const auto& ep_profiling_status = profiler.GetEpProfilingStatus();
  if (!ep_profiling_status.IsOK()) {
    GTEST_SKIP() << "EP profiling did not start: " << ep_profiling_status.ErrorMessage();
  }

  // Profiling started successfully — run inference.
  // Input X:[3,2] float, output Y:[3,1].
  std::vector<float> x_data(6, 1.0f);
  int64_t x_shape[] = {3, 2};

  OrtValue input_tensor;
  Tensor::InitOrtValue(DataTypeImpl::GetType<float>(),
                       TensorShape(x_shape, 2),
                       x_data.data(), OrtMemoryInfo(),
                       input_tensor);

  std::vector<std::string> feed_names = {"X"};
  std::vector<OrtValue> feeds = {input_tensor};
  std::vector<std::string> output_names = {"Y"};
  std::vector<OrtValue> fetches;

  RunOptions run_options;
  ASSERT_STATUS_OK(session.Run(run_options, feed_names, feeds, output_names, &fetches));

  // End profiling and read the output file.
  std::string profile_file_path = session.EndProfiling();

  auto cleanup_profile = gsl::finally([&profile_file_path] {
    std::error_code ec;
    std::filesystem::remove(profile_file_path, ec);
  });

  ASSERT_TRUE(std::filesystem::exists(profile_file_path))
      << "Profile file not found: " << profile_file_path;

  std::ifstream profile_stream(profile_file_path);
  ASSERT_TRUE(profile_stream.is_open()) << "Could not open: " << profile_file_path;

  std::string content(std::istreambuf_iterator<char>{profile_stream},
                      std::istreambuf_iterator<char>{});
  profile_stream.close();

  auto profile_json = nlohmann::json::parse(content);
  ASSERT_TRUE(profile_json.is_array()) << "Profile JSON is not an array";
  ASSERT_GT(profile_json.size(), 0u) << "Profile JSON is empty";

  // Validate standard fields on all entries.
  for (const auto& entry : profile_json) {
    if (!entry.is_object() || !entry.contains("name")) {
      continue;
    }
    EXPECT_TRUE(entry.contains("pid")) << "Missing 'pid': " << entry;
    EXPECT_TRUE(entry.contains("ts")) << "Missing 'ts': " << entry;
    EXPECT_TRUE(entry.contains("dur")) << "Missing 'dur': " << entry;
    EXPECT_TRUE(entry.contains("ph")) << "Missing 'ph': " << entry;
    EXPECT_TRUE(entry.contains("args")) << "Missing 'args': " << entry;
  }

  // Since EP profiling started OK, GPU Kernel events MUST be present.
  // Their absence indicates a regression in the profiling wiring.
  std::vector<nlohmann::json> kernel_events;
  for (const auto& entry : profile_json) {
    if (entry.is_object() && entry.contains("cat") && entry["cat"] == "Kernel") {
      kernel_events.push_back(entry);
    }
  }

  ASSERT_FALSE(kernel_events.empty())
      << "EP profiling started successfully (CUPTI tracing is active) but no GPU "
      << "Kernel events were found in the profile output. This is a regression.\n"
      << "Profile content (first 2000 chars): " << content.substr(0, 2000);

  // Validate kernel event metadata.
  for (const auto& event : kernel_events) {
    EXPECT_TRUE(event.contains("ts")) << event;
    EXPECT_TRUE(event.contains("dur")) << event;
    EXPECT_GE(event["dur"].get<int64_t>(), 0) << event;
    ASSERT_TRUE(event.contains("args")) << "Kernel event missing 'args': " << event;
    const auto& args = event["args"];
    EXPECT_TRUE(args.contains("stream")) << "Kernel missing 'stream': " << event;
    EXPECT_TRUE(args.contains("block_x")) << "Kernel missing 'block_x': " << event;
  }

  // Timeline plausibility: kernel timestamps should fall within the session
  // profiling window (derived from CPU-side events).
  int64_t session_end_us = 0;
  for (const auto& entry : profile_json) {
    if (!entry.is_object() || !entry.contains("cat")) continue;
    std::string cat = entry["cat"].get<std::string>();
    if ((cat == "Session" || cat == "Node" || cat == "Api") &&
        entry.contains("ts") && entry.contains("dur")) {
      int64_t end = entry["ts"].get<int64_t>() + entry["dur"].get<int64_t>();
      session_end_us = std::max(session_end_us, end);
    }
  }

  if (session_end_us > 0) {
    constexpr int64_t kMarginUs = 10'000;  // 10ms margin for GPU clock skew
    for (const auto& event : kernel_events) {
      int64_t ts = event["ts"].get<int64_t>();
      EXPECT_GE(ts, -kMarginUs)
          << "GPU kernel ts before profiling start (domain mismatch?): " << event;
      EXPECT_LE(ts, session_end_us + kMarginUs)
          << "GPU kernel ts beyond session end (domain mismatch?): " << event;
    }
  }
}

}  // namespace test
}  // namespace onnxruntime

#endif  // defined(ORT_UNIT_TEST_HAS_CUDA_PLUGIN_EP) && defined(ENABLE_CUDA_PROFILING)

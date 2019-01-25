// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testenv.h"
#include <core/common/logging/logging.h>
#include <core/graph/constants.h>
#include <core/framework/allocator.h>
#include <core/framework/execution_provider.h>
#include <core/session/onnxruntime_cxx_api.h>
#include <experimental/filesystem>
#ifdef _MSC_VER
#include <filesystem>
#endif
#include "providers.h"
#include "default_providers.h"

using namespace std::experimental::filesystem::v1;
using onnxruntime::Status;

inline void RegisterExecutionProvider(onnxruntime::InferenceSession* sess, std::unique_ptr<onnxruntime::IExecutionProvider>&& f) {
  auto status = sess->RegisterExecutionProvider(std::move(f));
  if (!status.IsOK()) {
    throw std::runtime_error(status.ErrorMessage().c_str());
  }
}

Status SessionFactory::create(std::shared_ptr<::onnxruntime::InferenceSession>& sess, const path& model_url, const std::string& logid) const {
  ::onnxruntime::SessionOptions so;
  so.session_logid = logid;
  so.enable_cpu_mem_arena = enable_cpu_mem_arena_;
  so.enable_mem_pattern = enable_mem_pattern_;
  so.enable_sequential_execution = enable_sequential_execution;
  so.session_thread_pool_size = session_thread_pool_size;
  sess.reset(new ::onnxruntime::InferenceSession(so));

  Status status;
  for (const std::string& provider : providers_) {
    if (provider == onnxruntime::kCudaExecutionProvider) {
#ifdef USE_CUDA
      RegisterExecutionProvider(sess.get(), onnxruntime::test::DefaultCudaExecutionProvider());
#else
      ORT_THROW("CUDA is not supported in this build");
#endif
    } else if (provider == onnxruntime::kMklDnnExecutionProvider) {
#ifdef USE_MKLDNN
      RegisterExecutionProvider(sess.get(), onnxruntime::test::DefaultMkldnnExecutionProvider(enable_cpu_mem_arena_ ? 1 : 0));
#else
      ORT_THROW("CUDA is not supported in this build");
#endif
    } else if (provider == onnxruntime::kNupharExecutionProvider) {
#ifdef USE_NUPHAR
      RegisterExecutionProvider(sess.get(), onnxruntime::test::DefaultNupharExecutionProvider());
#else
      ORT_THROW("CUDA is not supported in this build");
#endif
    } else if (provider == onnxruntime::kBrainSliceExecutionProvider) {
#if USE_BRAINSLICE
      RegisterExecutionProvider(sess.get(), onnxruntime::test::DefaultBrainsliceExecutionProvider());
#else
      ORT_THROW("This executable was not built with BrainSlice");
#endif
    } else if (provider == onnxruntime::kTRTExecutionProvider) {
#if USE_TRT
      OrtProviderFactoryInterface** f;
      ORT_THROW_ON_ERROR(OrtCreateTRTExecutionProviderFactory(0, &f));
      RegisterExecutionProvider(sess.get(), f);
      FACTORY_PTR_HOLDER;
#else
      ORT_THROW("TensorRT is not supported in this build");
#endif
    }
    // TODO: add more
  }

  status = sess->Load(model_url.string());
  ORT_RETURN_IF_ERROR(status);
  LOGS_DEFAULT(INFO) << "successfully loaded model from " << model_url;
  status = sess->Initialize();
  if (status.IsOK())
    LOGS_DEFAULT(INFO) << "successfully initialized model from " << model_url;
  return status;
}

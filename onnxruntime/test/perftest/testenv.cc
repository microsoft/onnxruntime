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

using namespace std::experimental::filesystem::v1;
using onnxruntime::Status;

inline void RegisterExecutionProvider(onnxruntime::InferenceSession* sess, OrtProviderFactory* f) {
  OrtProvider* p;
  f->CreateProvider(f, &p);
  std::unique_ptr<onnxruntime::IExecutionProvider> q((onnxruntime::IExecutionProvider*)p);
  auto status = sess->RegisterExecutionProvider(std::move(q));
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
      OrtProviderFactory* f;
      ORT_THROW_ON_ERROR(OrtCreateCUDAExecutionProviderFactory(0, &f));
      std::unique_ptr<OrtProviderFactory> ptr_holder_(f);
      RegisterExecutionProvider(sess.get(), f);
#else
      ORT_THROW("CUDA is not supported in this build");
#endif
    } else if (provider == onnxruntime::kMklDnnExecutionProvider) {
#ifdef USE_MKLDNN
      OrtProviderFactory* f;
      ORT_THROW_ON_ERROR(OrtCreateMkldnnExecutionProviderFactory(enable_cpu_mem_arena_ ? 1 : 0, &f));
      std::unique_ptr<OrtProviderFactory> ptr_holder_(f);
      RegisterExecutionProvider(sess.get(), f);
#else
      ORT_THROW("CUDA is not supported in this build");
#endif
    } else if (provider == onnxruntime::kNupharExecutionProvider) {
#ifdef USE_NUPHAR
      OrtProviderFactory* f;
      ORT_THROW_ON_ERROR(OrtCreateNupharExecutionProviderFactory(0, "", &f));
      std::unique_ptr<OrtProviderFactory> ptr_holder_(f);
      RegisterExecutionProvider(sess.get(), f);
#else
      ORT_THROW("CUDA is not supported in this build");
#endif
    } else if (provider == onnxruntime::kBrainSliceExecutionProvider) {
#if USE_BRAINSLICE
      OrtProviderFactory* f;
      ORT_THROW_ON_ERROR(OrtCreateBrainSliceExecutionProviderFactory(0, true, "testdata/firmwares/onnx_rnns/instructions.bin", "testdata/firmwares/onnx_rnns/data.bin", "testdata/firmwares/onnx_rnns/schema.bin", &f));
      std::unique_ptr<OrtProviderFactory> ptr_holder_(f);
      RegisterExecutionProvider(sess.get(), f);
#else
      ORT_THROW("This executable was not built with BrainSlice");
#endif
    }
    //TODO: add more
  }

  status = sess->Load(model_url.string());
  ORT_RETURN_IF_ERROR(status);
  LOGS_DEFAULT(INFO) << "successfully loaded model from " << model_url;
  status = sess->Initialize();
  if (status.IsOK())
    LOGS_DEFAULT(INFO) << "successfully initialized model from " << model_url;
  return status;
}

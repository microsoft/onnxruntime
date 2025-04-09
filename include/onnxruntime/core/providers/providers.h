// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

struct OrtSessionOptions;
struct OrtLogger;

namespace onnxruntime {
class IExecutionProvider;

struct IExecutionProviderFactory {
  virtual ~IExecutionProviderFactory() = default;
  virtual std::unique_ptr<IExecutionProvider> CreateProvider() = 0;
  virtual std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions* session_options,
	                                                         const OrtLogger* logger) = 0;
};
}  // namespace onnxruntime

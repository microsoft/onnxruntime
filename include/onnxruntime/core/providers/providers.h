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

  /// <summary>
  /// Creates an IExecutionProvider instance. Allows using session-level options such as
  /// session configs (string key/value pairs), graph optimization level, etc. to initialize the EP.
  ///
  /// An IExecutionProviderFactory that does not need to extract information from the session options should return
  /// the result from the above CreateProvider() function.
  ///
  /// This version of CreateProvider() is used by InferenceSession when registering EPs.
  /// </summary>
  /// <param name="session_options"></param>
  /// <param name="logger"></param>
  /// <returns></returns>
  virtual std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions* session_options,
                                                             const OrtLogger* logger) = 0;
};
}  // namespace onnxruntime

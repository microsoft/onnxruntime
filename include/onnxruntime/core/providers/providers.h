// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <memory>

struct OrtSessionOptions;
struct OrtLogger;

namespace onnxruntime {
class IExecutionProvider;

struct IExecutionProviderFactory {
  virtual ~IExecutionProviderFactory() = default;
  virtual std::unique_ptr<IExecutionProvider> CreateProvider() = 0;

  /// <summary>
  /// Creates an IExecutionProvider instance. Enables initialization of an EP instance using session-level options
  /// such as session configs (string key/value pairs), graph optimization level, etc.
  ///
  /// The default implementation ignores the arguments and calls the above CreateProvider() function,
  /// which does not take in any arguments.
  ///
  /// This version of CreateProvider() is used by InferenceSession when registering EPs.
  /// </summary>
  /// <param name="session_options">Options for the session in which the IExecutionProvider is used</param>
  /// <param name="session_logger">Session logger that should be used by the IExecutionProvider.</param>
  /// <returns>An IExecutionProvider</returns>
  virtual std::unique_ptr<IExecutionProvider> CreateProvider(const OrtSessionOptions& session_options,
                                                             const OrtLogger& session_logger);
};
}  // namespace onnxruntime

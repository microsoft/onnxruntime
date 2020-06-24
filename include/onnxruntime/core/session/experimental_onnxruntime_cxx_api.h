// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Summary: The experimental Ort C++ API is a wrapper around the Ort C++ API.
//
// The experimental C++ API further simplifies usage and provides support for modern C++ syntax/features
// at the cost of some overhead (for example, using std::string over char *).
//
// Where possible, certain default values are defined by the session unless otherwise specified by the API caller.
//
// NOTE: Experimental API components are subject to change and provide no guarantee of backwards compatibility in future releases.

#pragma once
#include "onnxruntime_cxx_api.h"

namespace Ort {

struct ExperimentalSession : Session {
  ExperimentalSession(Env& env, ORTCHAR_T* model_path, SessionOptions& options)
      : Session(env, model_path, options){};
  ExperimentalSession(Env& env, void* model_data, size_t model_data_length, SessionOptions& options)
      : Session(env, model_data, model_data_length, options){};

  // overloaded Run() with sensible defaults
  std::vector<Value> Run(const std::vector<std::string>& input_names,
                         const std::vector<Value>& input_values,
                         const std::vector<std::string>& output_names,
                         const RunOptions& run_options = RunOptions());
  void Run(const std::vector<std::string>& input_names,
           const std::vector<Value>& input_values,
           const std::vector<std::string>& output_names,
           std::vector<Value>& output_values,
           const RunOptions& run_options = RunOptions());

  // convenience methods that simplify common lower-level API calls
  std::vector<std::string> GetInputNames() const;
  std::vector<std::string> GetOutputNames() const;
  std::vector<std::string> GetOverridableInitializerNames() const;

  // NOTE: shape dimensions may have a negative value to indicate a symbolic/unknown dimension.
  // This is typically used to denote batch size.
  std::vector<std::vector<int64_t> > GetInputShapes() const;
  std::vector<std::vector<int64_t> > GetOutputShapes() const;
  std::vector<std::vector<int64_t> > GetOverridableInitializerShapes() const;
};

}  // namespace Ort

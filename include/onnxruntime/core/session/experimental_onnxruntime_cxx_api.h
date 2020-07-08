// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// Summary: The experimental Ort C++ API is a wrapper around the Ort C++ API.
//
// This C++ API further simplifies usage and provides support for modern C++ syntax/features
// at the cost of some overhead (i.e. using std::string over char *).
//
// Experimental components are designed as drop-in replacements of the regular API, requiring
// minimal code modifications to use.
//
// Example:  Ort::Session -> Ort::Experimental::Session
//
// NOTE: Experimental API components are subject to change based on feedback and provide no
// guarantee of backwards compatibility in future releases.

#pragma once
#include "onnxruntime_cxx_api.h"

namespace Ort::Experimental {

struct Session : Ort::Session {
  Session(Env& env, ORTCHAR_T* model_path, SessionOptions& options)
      : Ort::Session(env, model_path, options){};
  Session(Env& env, void* model_data, size_t model_data_length, SessionOptions& options)
      : Ort::Session(env, model_data, model_data_length, options){};

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
  std::vector<std::vector<int64_t> > GetInputShapes() const;
  std::vector<std::vector<int64_t> > GetOutputShapes() const;
  std::vector<std::vector<int64_t> > GetOverridableInitializerShapes() const;
};

}  // namespace Ort::Experimental

#include "experimental_onnxruntime_cxx_inline.h"

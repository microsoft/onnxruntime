// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains the implementation of TuningResultsManager. At the moment, there is no necessity to expose these
// methods as OrtApis. This will cause missing symbols when loading provider dynamic libraries, because the libraries
// are not whole-archive linked and these symbols are not referenced at framework level. To circumvent this problem,
// the EP must has and only has one translation unit include this file.
#ifndef TUNING_CONTEXT_IMPL
#error define TUNING_CONTEXT_IMPL to use this header (impl) file
#endif

#pragma once

#include "core/framework/tunable.h"
#include "core/framework/tuning_context.h"
#include "core/framework/tuning_results.h"

namespace onnxruntime {

KernelMap TuningResultsManager::Lookup(const std::string& op_signature) const {
  std::scoped_lock l{lock_};
  auto it = results_.find(op_signature);
  if (it == results_.cend()) {
    return {};
  }
  return it->second;  // copied
}

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
int TuningResultsManager::Lookup(const std::string& op_signature, const std::string& params_signature) const {
  std::scoped_lock l{lock_};
  auto kernel_map_it = results_.find(op_signature);
  if (kernel_map_it == results_.cend()) {
    return -1;
  }

  const auto& km = kernel_map_it->second;
  auto it = km.find(params_signature);
  if (it == km.cend()) {
    return -1;
  }
  return it->second;
}

inline void AddImpl(const std::string& op_signature,
                    const std::string& params_signature,
                    int best_id,
                    KernelMap& kernel_map) {
  auto it = kernel_map.find(params_signature);
  if (it != kernel_map.end()) {
    if (it->second != best_id) {
      LOGS_DEFAULT(WARNING) << op_signature << "(" << params_signature << ") already have a best kernel "
                            << "id=" << it->second << " selected, want to add a different best kernel id=" << best_id
                            << ", the new kernel id will be ignored.";
    }
    return;
  }

  kernel_map[params_signature] = best_id;
}

void TuningResultsManager::Add(const std::string& op_signature, const std::string& params_signature, int best_id) {
  std::scoped_lock l{lock_};

  auto it = results_.find(op_signature);
  if (it == results_.end()) {
    it = results_.insert({op_signature, {}}).first;
  }

  AddImpl(op_signature, params_signature, best_id, it->second);
}

std::unordered_map<std::string, KernelMap> TuningResultsManager::Dump() const {
  std::scoped_lock l{lock_};
  return results_;
}

void MergeImpl(
    const std::string& op_signature,
    const KernelMap& kernel_map,
    /*out*/ std::unordered_map<std::string, KernelMap>& results) {
  auto it = results.find(op_signature);
  if (it == results.end()) {
    results[op_signature] = kernel_map;
    return;
  }

  for (const auto& [params_signature, best_id] : kernel_map) {
    AddImpl(op_signature, params_signature, best_id, it->second);
  }
}

void TuningResultsManager::Load(const std::unordered_map<std::string, KernelMap>& results_to_load) {
  std::scoped_lock l{lock_};
  for (const auto& [op_signature, kernel_map] : results_to_load) {
    MergeImpl(op_signature, kernel_map, results_);
  }
}

void TuningResultsManager::Merge(const std::string& op_signature, const KernelMap& kernel_map) {
  std::scoped_lock l{lock_};
  MergeImpl(op_signature, kernel_map, results_);
}

void TuningResultsManager::Clear() {
  results_ = {};
}

}  // namespace onnxruntime

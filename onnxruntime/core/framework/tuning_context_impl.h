// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file contains the implementation of TuningContext. At the moment, there is no necessity to expose these
// methods as OrtApis. This will cause missing symbols when loading provider dynamic libraries, because the libraries
// are not whole-archive linked and these symbols are not referenced at framework level. To circumvent this problem,
// the EP must has and only has one translation unit include this file.
#ifndef TUNING_CONTEXT_IMPL
#error define TUNING_CONTEXT_IMPL to use this header (impl) file
#endif

#pragma once

#include <functional>
#include <unordered_set>
#include <utility>

#include "core/framework/tunable.h"
#include "core/framework/tuning_context.h"
#include "core/framework/tuning_results.h"

namespace onnxruntime {

TuningResults ITuningContext::GetTuningResults() const {
  TuningResults tr;
  tr.ep = ep_->Type();
  tr.validators = GetTuningResultsValidator().GetAllValidators();
  tr.results = GetTuningResultsManager().Dump();
  return tr;
}

Status ITuningContext::LoadTuningResults(const TuningResults& tr) {
  ORT_RETURN_IF(tr.ep != ep_->Type(), "EP mismatch");
  LOGS_DEFAULT(VERBOSE) << "Loading tuning results for " << tr.ep;
  ORT_RETURN_IF_ERROR(GetTuningResultsValidator().ValidateAll(tr.validators));
  GetTuningResultsManager().Load(tr.results);
  return Status::OK();
}

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
      LOGS_DEFAULT(WARNING) << op_signature << "(" << params_signature << ") already has a best kernel "
                            << "id=" << it->second << " selected, want to add a different best kernel id=" << best_id
                            << ", the new kernel id will be ignored.";
    }
    return;
  }

  LOGS_DEFAULT(VERBOSE) << op_signature << "(" << params_signature << ") -> " << best_id;
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

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void TuningResultsManager::Delete(const std::string& op_signature, const std::string& params_signature) {
  std::scoped_lock l{lock_};

  auto it = results_.find(op_signature);
  if (it == results_.end()) {
    return;
  }

  auto it2 = it->second.find(params_signature);
  if (it2 == it->second.end()) {
    return;
  }

  LOGS_DEFAULT(VERBOSE) << op_signature << "(" << params_signature << ")";
  it->second.erase(it2);
}

std::unordered_map<std::string, KernelMap> TuningResultsManager::Dump() const {
  std::scoped_lock l{lock_};
  return results_;
}

void DisjointMergeImpl(
    const std::string& op_signature,
    const KernelMap& kernel_map,
    /*out*/ std::unordered_map<std::string, KernelMap>& results) {
  auto it = results.find(op_signature);
  if (it == results.end()) {
    for (const auto& [param_sig, kernel_id] : kernel_map) {
      LOGS_DEFAULT(VERBOSE) << op_signature << "(" << param_sig << ") -> " << kernel_id;
    }
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
    DisjointMergeImpl(op_signature, kernel_map, results_);
  }
}

void TuningResultsManager::DisjointMerge(const std::string& op_signature, const KernelMap& kernel_map) {
  std::scoped_lock l{lock_};
  DisjointMergeImpl(op_signature, kernel_map, results_);
}

void TuningResultsManager::Clear() {
  results_ = {};
}

static Status CheckMandatoryKeys(
    const TuningResultsValidator::GetValidateFuncs& gv_funcs,
    const std::unordered_map<std::string, std::string>& to_check) {
  bool passed = true;
  std::ostringstream oss;
  for (const auto& k : TuningResultsValidator::mandatory_keys) {
    if (gv_funcs.find(k) == gv_funcs.end()) {
      passed = false;
      oss << "key=\"" << k << "\" is not registered for Get and Validate. ";
    }

    if (to_check.find(k) == to_check.end()) {
      passed = false;
      oss << "key=\"" << k << "\" is not provided for validation. ";
    }
  }
  ORT_RETURN_IF(!passed, oss.str());
  return Status::OK();
}

static Status CheckKeysMatching(
    const TuningResultsValidator::GetValidateFuncs& gv_funcs,
    const std::unordered_map<std::string, std::string>& to_check) {
  auto get_keys = [](const auto& it) -> std::string { return it.first; };
  std::vector<std::string> required_keys;
  std::vector<std::string> provided_keys;
  std::transform(gv_funcs.cbegin(), gv_funcs.cend(), std::back_inserter(required_keys), get_keys);
  std::transform(to_check.cbegin(), to_check.cend(), std::back_inserter(provided_keys), get_keys);
  std::sort(required_keys.begin(), required_keys.end());
  std::sort(provided_keys.begin(), provided_keys.end());

  std::unordered_set<std::string> intersection;
  std::set_intersection(required_keys.cbegin(), required_keys.cend(),
                        provided_keys.cbegin(), provided_keys.cend(),
                        std::inserter(intersection, intersection.end()));
  bool matched = true;
  std::ostringstream oss;
  if (intersection.size() != required_keys.size()) {
    matched = false;
    for (const auto& k : required_keys) {
      if (intersection.find(k) == intersection.end()) {
        oss << "Unmatched validator: \"" << k << "\" is required, but the tuning results does not provide it. ";
      }
    }
  }
  if (intersection.size() != provided_keys.size()) {
    matched = false;
    for (const auto& k : provided_keys) {
      if (intersection.find(k) == intersection.end()) {
        oss << "Unmatched validator: \"" << k << "\" is provided, but onnxruntime is unable to consume it. ";
      }
    }
  }
  ORT_RETURN_IF(!matched, oss.str());
  return Status::OK();
}

std::string TuningResultsValidator::GetOrtVersion() const {
  return ORT_VERSION;
}

Status TuningResultsValidator::ValidateOrtVersion(const std::string& value) const {
  ORT_RETURN_IF(value != ORT_VERSION, "onnxruntime version mismatch");
  return Status::OK();
}

std::string TuningResultsValidator::GetOrtGitCommit() const {
  // TODO:
  return "";
}

Status TuningResultsValidator::ValidateOrtGitCommit(const std::string& value) const {
  // TODO:
  ORT_UNUSED_PARAMETER(value);
  return Status::OK();
}

std::string TuningResultsValidator::GetOrtBuildConfig() const {
  return "";
}

Status TuningResultsValidator::ValidateOrtBuildConfig(const std::string& value) const {
  auto current = GetOrtBuildConfig();
  ORT_RETURN_IF(current != value,
                "onnxruntime building configuration mismatch: tuning results produced with library \"",
                value, "\", current library built with \"", current, "\"");
  return Status::OK();
}

TuningResultsValidator::TuningResultsValidator() {
  RegisterValidator(
      "ORT_VERSION",
      [this]() { return GetOrtVersion(); },
      [this](auto&& k) { return ValidateOrtVersion(std::forward<decltype(k)>(k)); });

  RegisterValidator(
      "ORT_GIT_COMMIT",
      [this]() { return GetOrtGitCommit(); },
      [this](auto&& k) { return ValidateOrtGitCommit(std::forward<decltype(k)>(k)); });

  RegisterValidator(
      "ORT_BUILD_CONFIG",
      [this]() { return GetOrtBuildConfig(); },
      [this](auto&& k) { return ValidateOrtBuildConfig(std::forward<decltype(k)>(k)); });
}

Status TuningResultsValidator::ValidateAll(const std::unordered_map<std::string, std::string>& to_validate) const {
  ORT_RETURN_IF_ERROR(CheckMandatoryKeys(validators_, to_validate));
  ORT_RETURN_IF_ERROR(CheckKeysMatching(validators_, to_validate));

  for (const auto& [key, value] : to_validate) {
    const auto& it = validators_.find(key);
    ORT_ENFORCE(it != validators_.cend());
    const ValidateFunc& validator = it->second.second;
    ORT_RETURN_IF_ERROR(validator(value));
  }

  return Status::OK();
}

std::unordered_map<std::string, std::string> TuningResultsValidator::GetAllValidators() const {
  std::unordered_map<std::string, std::string> ret;
  for (const auto& [key, get_validate_func_pair] : validators_) {
    const GetFunc& getter = get_validate_func_pair.first;
    ret[key] = getter();
  }
  return ret;
}

void TuningResultsValidator::RegisterValidator(const std::string& key, const GetFunc& gf, const ValidateFunc& vf) {
  ORT_ENFORCE(validators_.find(key) == validators_.end());
  validators_[key] = std::make_pair(gf, vf);
}

}  // namespace onnxruntime

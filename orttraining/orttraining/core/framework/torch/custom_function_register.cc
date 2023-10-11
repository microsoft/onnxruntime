// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "orttraining/core/framework/torch/custom_function_register.h"
#include "core/common/common.h"
#include "orttraining/core/framework/torch/refcount_tracker.h"
#include "core/platform/env.h"
#include <cstdio>
#include <sstream>
#include <string>

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

// Perform a thread-safe registration for "pool" (type: map).
// It creates a new ownership to the Python object "obj" and that
// ownership is stored in "pool".
template <typename TKey>
static void RegisterEntry(
    std::mutex& mutex,  // The mutex uniquely associated with "pool".
    TKey key,           // used in move-constructor of tuple below.
    PyObject* obj,
    std::unordered_map<TKey, PythonObjectPtr>& pool) {
  std::lock_guard<std::mutex> lock(mutex);
  // Get iterator to the existing entry, if exists.
  auto it = pool.find(key);

  // Don't overwrite existing registered function.
  if (it != pool.end()) {
    return;
  }

  // Own the Python object.
  Py_INCREF(obj);
  PythonObjectPtr ptr(obj, PythonObjectDeleter);

  if (it != pool.end()) {
    // If an obj has been registered for the key, we release
    // the ownership of the old one.
    it->second = std::move(ptr);
  } else {
    // Add new entry if key hasn't been registered.
    pool.emplace(key, std::move(ptr));
  }
}

static bool EnsureTorchAutogradFunction(PyObject* obj) {
  // New reference.
  PythonObjectPtr bases(PyObject_GetAttrString(obj, "__bases__"), PythonObjectDeleter);
  // New reference.
  PythonObjectPtr expected_base_name(PyUnicode_FromString("Function"), PythonObjectDeleter);
  // New reference.
  PythonObjectPtr expected_module_name(PyUnicode_FromString("torch.autograd.function"), PythonObjectDeleter);
  const auto n_bases = static_cast<size_t>(PyTuple_GET_SIZE(bases.get()));

  bool correct = false;

  for (size_t i = 0; i < n_bases; ++i) {
    // Borrow reference.
    PyObject* base = PyTuple_GetItem(bases.get(), i);
    // New reference.
    PythonObjectPtr base_name(PyObject_GetAttrString(base, "__name__"), PythonObjectDeleter);
    // New reference.
    PythonObjectPtr module_name(PyObject_GetAttrString(base, "__module__"), PythonObjectDeleter);

    const bool correct_base_name = PyObject_RichCompareBool(
                                       base_name.get(), expected_base_name.get(), Py_EQ) == 1;
    const bool correct_module_name = PyObject_RichCompareBool(
                                         module_name.get(), expected_module_name.get(), Py_EQ) == 1;

    if (correct_base_name && correct_module_name) {
      correct = true;
      break;
    }
  }

  return correct;
}

void OrtTorchFunctionPool::RegisterTorchAutogradFunction(
    const std::string& key,
    PyObject* obj) {
  auto correct = EnsureTorchAutogradFunction(obj);
  ORT_ENFORCE(correct, "Only torch.autograd.Function is allowed to be registered with key ", key);

  // New reference.
  PythonObjectPtr forward(PyObject_GetAttrString(obj, "apply"), PythonObjectDeleter);
  PythonObjectPtr backward(PyObject_GetAttrString(obj, "backward"), PythonObjectDeleter);

  ORT_ENFORCE(forward.get(), "apply attribute not found when registering ", key);
  ORT_ENFORCE(backward.get(), "backward attribute not found when registering ", key);

  RegisterEntry(mutex_, key, forward.get(), forward_core_pool_);
  RegisterEntry(mutex_, key, backward.get(), backward_core_pool_);
}

void OrtTorchFunctionPool::RegisterShapeInferenceFunction(const std::string& key,
                                                          PyObject* obj) {
  RegisterEntry(mutex_, key, obj, shape_inference_function_pool_);
}

void OrtTorchFunctionPool::RegisterInputAliasFunction(const std::string& key,
                                                      PyObject* obj) {
  RegisterEntry(mutex_, key, obj, input_alias_function_pool_);
}

static void RegisterEntry(
    std::mutex& mutex,
    PyObject* obj,
    PythonObjectPtr& storage) {
  std::lock_guard<std::mutex> lock(mutex);
  // Basic checks.
  ORT_ENFORCE(obj, "Cannot register NULL PyObject*.");

  // Skip registration if storage already stores a Python object.
  if (storage.get() != nullptr) {
    return;
  }

  // Own the Python object.
  Py_INCREF(obj);
  PythonObjectPtr ptr(obj, PythonObjectDeleter);

  // If an obj has been registered, this old ownership is automatically released
  // after this move-assignment. Then, the "storage" owns the new object.
  storage = std::move(ptr);
}

void OrtTorchFunctionPool::RegisterForwardRunner(PyObject* obj) {
  RegisterEntry(mutex_, obj, forward_runner_);
}

void OrtTorchFunctionPool::RegisterBackwardRunner(PyObject* obj) {
  RegisterEntry(mutex_, obj, backward_runner_);
}

PyObject* OrtTorchFunctionPool::GetForwardRunner() {
  std::lock_guard<std::mutex> lock(mutex_);
  ORT_ENFORCE(forward_runner_.get(), "Forward runner cannot be NULL. Do you forget register it by calling RegisterForwardRunner(...)?");
  return forward_runner_.get();
}

PyObject* OrtTorchFunctionPool::GetBackwardRunner() {
  std::lock_guard<std::mutex> lock(mutex_);
  ORT_ENFORCE(backward_runner_.get(), "backward runner cannot be NULL. Do you forget register it by calling RegisterBackwardRunner(...)?");
  return backward_runner_.get();
}

PyObject* OrtTorchFunctionPool::GetForwardCore(const std::string& key) {
  ORT_ENFORCE(!key.empty(), "Cannot be empty string.");
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = forward_core_pool_.find(key);
  ORT_ENFORCE(iter != forward_core_pool_.end(), "No forward registered for ", key);
  return iter->second.get();
}

PyObject* OrtTorchFunctionPool::GetBackwardCore(const std::string& key) {
  ORT_ENFORCE(!key.empty(), "Cannot be empty string.");
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = backward_core_pool_.find(key);
  ORT_ENFORCE(iter != backward_core_pool_.end(), "No backward registered for ", key);
  return iter->second.get();
}

std::optional<PyObject*> OrtTorchFunctionPool::TryGettingShapeInferenceFunction(const std::string& key) {
  ORT_ENFORCE(!key.empty(), "Cannot be empty string.");
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = shape_inference_function_pool_.find(key);
  if (iter != shape_inference_function_pool_.end()) {
    return iter->second.get();
  }
  return std::nullopt;
}

std::optional<PyObject*> OrtTorchFunctionPool::TryGettingInputAliasFunction(const std::string& key) {
  ORT_ENFORCE(!key.empty(), "Cannot be empty string.");
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = input_alias_function_pool_.find(key);
  if (iter != input_alias_function_pool_.end()) {
    return iter->second.get();
  }
  return std::nullopt;
}

void OrtTorchFunctionPool::RegisterMiscellaneousConstInput(PyObject* obj) {
  ORT_ENFORCE(obj, "Cannot register NULL reference input.");
  const void* address = static_cast<const void*>(obj);
  std::stringstream ss;
  ss << address;
  std::string key = ss.str();
  RegisterEntry(mutex_, key, obj, miscellaneous_const_input_pool_);
}

int64_t OrtTorchFunctionPool::RegisterContext(PyObject* autograd_context) {
  static int64_t index_ = 0x1000000;
  std::lock_guard<std::mutex> lock(mutex_);
  index_++;

  RefCountTracker::GetInstance().TrackPyObject(RefCountTracker::ObjCategory::AutoGradContext,
                                               autograd_context, "autograd_context_register");

  ORT_ENFORCE(autograd_context, "Cannot register NULL autograd context.");
  Py_INCREF(autograd_context);

  func_context_pool_.insert({index_, PythonObjectPtr(autograd_context, PythonObjectDeleter)});
  // We don't need increase the context refcnt because PyTorch already did it during .apply().
  return index_;
}

void OrtTorchFunctionPool::UnregisterContext(int64_t context_index) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = func_context_pool_.find(context_index);

  // We just need remove the context key value pair, the context itself
  // will be removed, when forward outputs are destoyed.
  ORT_ENFORCE(it != func_context_pool_.end(),
              "Cannot unregister unexisting key: ", context_index);
  func_context_pool_.erase(it);
}

PyObject* OrtTorchFunctionPool::GetContext(int64_t context_index) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = func_context_pool_.find(context_index);
  ORT_ENFORCE(iter != func_context_pool_.end(), "No context registered for ", context_index);
  return iter->second.get();
}

void OrtTorchFunctionPool::UnRegisterGlobalFunctions() {
  forward_runner_.reset();
  backward_runner_.reset();
  func_context_pool_.clear();
}

void OrtTorchFunctionPool::UnRegisterModelSpecificFunctions() {
  forward_core_pool_.clear();
  backward_core_pool_.clear();
  shape_inference_function_pool_.clear();
  input_alias_function_pool_.clear();
  miscellaneous_const_input_pool_.clear();
}

void OrtTorchFunctionPool::UnRegisterFunctions() {
  UnRegisterGlobalFunctions();
  UnRegisterModelSpecificFunctions();
}

}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime

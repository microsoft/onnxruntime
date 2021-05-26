// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/common.h"
#include "core/language_interop_ops/torch/custom_function_register.h"
#include "core/language_interop_ops/torch/refcount_tracker.h"
#include "core/platform/env.h"

namespace onnxruntime {
namespace language_interop_ops {
namespace torch {

// Perform a thread-safe registration for "pool" (type: map).
template <typename TKey>
static void RegisterEntry(
    std::mutex& mutex,  // The mutex uniquely associated with "pool".
    TKey key,           // used in move-constructor of tuple below.
    PyObject* obj,
    std::unordered_map<TKey, PyObject*>& pool,
    const bool overwrite) {
  std::lock_guard<std::mutex> lock(mutex);
  // Get iterator to the existing entry, if exists.
  auto it = pool.find(key);
  if (!overwrite) {
    // Cannot overwrite existing registered function.
    ORT_ENFORCE(it == pool.end(), "Duplicated registration found: ", key);
  }

  // Own the Python object.
  Py_INCREF(obj);

  if (it != pool.end()) {
    // If an obj has been registered for the key, we release
    // the ownership of the old one.
    Py_DECREF(it->second);
    it->second = obj;
  } else {
    // Add new entry if key hasn't been registered.
    pool.emplace(key, obj);
  }
}

static bool EnsureTorchAutogradFunction(PyObject* obj) {
  PyObject* bases = PyObject_GetAttrString(obj, "__bases__");
  PyObject* expected_base_name = PyUnicode_FromString("Function");
  PyObject* expected_module_name = PyUnicode_FromString("torch.autograd.function");
  const auto n_bases = static_cast<size_t>(PyTuple_GET_SIZE(bases));

  bool correct = false;

  for (size_t i = 0; i < n_bases; ++i) {
    // Borrow reference.
    PyObject* base = PyTuple_GetItem(bases, i);
    // New reference.
    PyObject* base_name = PyObject_GetAttrString(base, "__name__");
    // New reference.
    PyObject* module_name = PyObject_GetAttrString(base, "__module__");

    const bool correct_base_name = PyObject_RichCompareBool(
                                       base_name, expected_base_name, Py_EQ) == 1;
    const bool correct_module_name = PyObject_RichCompareBool(
                                         module_name, expected_module_name, Py_EQ) == 1;

    if (correct_base_name && correct_module_name) {
      correct = true;
      break;
    }

    Py_DECREF(base_name);
    Py_DECREF(module_name);
  }

  Py_DECREF(bases);
  Py_DECREF(expected_base_name);
  Py_DECREF(expected_module_name);

  return correct;
}

void OrtTorchFunctionPool::RegisterTorchAutogradFunction(
    const std::string& key,
    PyObject* obj,
    const bool overwrite) {
  auto correct = EnsureTorchAutogradFunction(obj);
  ORT_ENFORCE(correct, "Only torch.autograd.Function is allowed to be registered with key ", key);

  PyObject* forward = PyObject_GetAttrString(obj, "apply");
  PyObject* backward = PyObject_GetAttrString(obj, "backward");

  ORT_ENFORCE(forward, "apply attribute not found when registering ", key);
  ORT_ENFORCE(backward, "backward attribute not found when registering ", key);

  RegisterEntry(mutex_, key, forward, forward_core_pool, overwrite);
  RegisterEntry(mutex_, key, backward, backward_core_pool, overwrite);

  Py_DECREF(forward);
  Py_DECREF(backward);
}

template <typename TKey>
static void UnregisterEntry(
    std::mutex& mutex,
    const TKey& key,
    std::unordered_map<TKey, PyObject*>& pool) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = pool.find(key);

  ORT_ENFORCE(it != pool.end(), "Cannot unregister unexisting key: ", key);
  // Release the ownership.
  Py_DECREF(it->second);
  pool.erase(it);
}

void OrtTorchFunctionPool::UnregisterTorchAutogradFunction(const std::string& key) {
  UnregisterEntry(mutex_, key, forward_core_pool);
  UnregisterEntry(mutex_, key, backward_core_pool);
}

static void RegisterEntry(
    std::mutex& mutex,
    PyObject* obj,
    PyObject** storage,
    const bool overwrite) {
  std::lock_guard<std::mutex> lock(mutex);
  // Basic checks.
  ORT_ENFORCE(storage, "Cannot store PyObject* on NULL pointer.");
  ORT_ENFORCE(obj, "Cannot register NULL PyObject*.");

  // Get iterator to the existing entry, if exists.
  if (!overwrite) {
    // Cannot overwrite existing registered function.
    ORT_ENFORCE(*storage == nullptr, "Duplicated registration found.");
  }

  // Own the Python object.
  Py_INCREF(obj);

  if (*storage) {
    // If an obj has been registered, we release
    // the ownership of the old one.
    Py_DECREF(*storage);
  }

  // Point to the new registered object.
  *storage = obj;
}

void OrtTorchFunctionPool::RegisterForwardRunner(PyObject* obj, bool overwrite) {
  RegisterEntry(mutex_, obj, &forward_runner, overwrite);
}

void OrtTorchFunctionPool::RegisterBackwardRunner(PyObject* obj, bool overwrite) {
  RegisterEntry(mutex_, obj, &backward_runner, overwrite);
}

static void UnregisterEntry(
    std::mutex& mutex,
    PyObject** storage) {
  std::lock_guard<std::mutex> lock(mutex);
  // Basic checks.
  ORT_ENFORCE(storage, "Cannot unregister PyObject* on NULL storage.");
  ORT_ENFORCE(*storage, "Cannot unregister NULL PyObject*.");

  // Release the ownership.
  Py_DECREF(*storage);
  // Avoid accessing the released object.
  *storage = nullptr;
}

void OrtTorchFunctionPool::UnregisterForwardRunner() {
  UnregisterEntry(mutex_, &forward_runner);
}

void OrtTorchFunctionPool::UnregisterBackwardRunner() {
  UnregisterEntry(mutex_, &backward_runner);
}

PyObject* OrtTorchFunctionPool::GetForwardRunner() {
  std::lock_guard<std::mutex> lock(mutex_);
  ORT_ENFORCE(forward_runner, "Forward runner cannot be NULL. Do you forget register it by calling RegisterForwardRunner(...)?");
  return forward_runner;
}

PyObject* OrtTorchFunctionPool::GetBackwardRunner() {
  std::lock_guard<std::mutex> lock(mutex_);
  ORT_ENFORCE(backward_runner, "backward runner cannot be NULL. Do you forget register it by calling RegisterBackwardRunner(...)?");
  return backward_runner;
}

PyObject* OrtTorchFunctionPool::GetForwardCore(const std::string& key) {
  ORT_ENFORCE(!key.empty(), "Cannot be empty string.");
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = forward_core_pool.find(key);
  ORT_ENFORCE(iter != forward_core_pool.end(), "No forward registered for ", key);
  return iter->second;
}

PyObject* OrtTorchFunctionPool::GetBackwardCore(const std::string& key) {
  ORT_ENFORCE(!key.empty(), "Cannot be empty string.");
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = backward_core_pool.find(key);
  ORT_ENFORCE(iter != backward_core_pool.end(), "No backward registered for ", key);
  return iter->second;
}

int64_t OrtTorchFunctionPool::RegisterContext(PyObject* auto_grad_context) {
  static int64_t index_ = 0x1000000;
  std::lock_guard<std::mutex> lock(mutex_);
  index_++;

  RefCountTracker::GetInstance().TrackPyObject(RefCountTracker::ObjCategory::AutoGradContext,
                                               auto_grad_context, "autograd_context_register");

  func_context_pool.insert({index_, auto_grad_context});
  // We don't need increase the context refcnt because PyTorch already did it during .apply().
  return index_;
}

void OrtTorchFunctionPool::UnregisterContext(int64_t context_index) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = func_context_pool.find(context_index);

  // We just need remove the context key value pair, the context itself
  // will be removed, when forward outputs are destoyed.
  ORT_ENFORCE(it != func_context_pool.end(),
              "Cannot unregister unexisting key: ", context_index);
  func_context_pool.erase(it);
}

PyObject* OrtTorchFunctionPool::GetContext(int64_t context_index) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto iter = func_context_pool.find(context_index);
  ORT_ENFORCE(iter != func_context_pool.end(), "No context registered for ", context_index);
  return iter->second;
}

template <typename TKey>
void UnregisterAllEntries(
    std::mutex& mutex,
    std::unordered_map<TKey, PyObject*>& pool) {
  std::lock_guard<std::mutex> lock(mutex);
  auto it = pool.begin();
  while (it != pool.end()) {
    Py_DECREF(it->second);
    it = pool.erase(it);
  }
}

OrtTorchFunctionPool::OrtTorchFunctionPool() : forward_runner(nullptr), backward_runner(nullptr){};
OrtTorchFunctionPool::~OrtTorchFunctionPool() {
  UnregisterEntry(mutex_, &forward_runner);
  UnregisterEntry(mutex_, &backward_runner);

  UnregisterAllEntries(mutex_, forward_core_pool);
  UnregisterAllEntries(mutex_, backward_core_pool);

  UnregisterAllEntries(mutex_, func_context_pool);
};

}  // namespace torch
}  // namespace language_interop_ops
}  // namespace onnxruntime

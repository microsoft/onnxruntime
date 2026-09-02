// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_cxx_api.h"

#include <memory>
#include <mutex>
#include <napi.h>

// class InferenceSessionWrap is a N-API object wrapper for native InferenceSession.
class InferenceSessionWrap : public Napi::ObjectWrap<InferenceSessionWrap> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  InferenceSessionWrap(const Napi::CallbackInfo& info);
  ~InferenceSessionWrap();

 private:
  class RunAsyncWorker;

  /**
   * [sync] initialize ONNX Runtime once.
   *
   * This function must be called before any other functions.
   *
   * @param arg0 a number specifying the log level.
   *
   * @returns undefined
   */
  static Napi::Value InitOrtOnce(const Napi::CallbackInfo& info);

  /**
   * [sync] list supported backend list
   * @returns array with objects { "name": "cpu", requirementsInstalled: true }
   */
  static Napi::Value ListSupportedBackends(const Napi::CallbackInfo& info);

  /**
   * [sync] create the session.
   * @param arg0 either a string (file path) or a Uint8Array
   * @returns nothing
   * @throw error if status code != 0
   */
  Napi::Value LoadModel(const Napi::CallbackInfo& info);

  // following functions have to be called after model is loaded.

  /**
   * [sync] get metadata of the model's inputs or outputs.
   * @param nothing
   * @returns an array of objects with keys: name, isTensor, type, symbolicDimensions, shape
   * @throw nothing
   */
  Napi::Value GetMetadata(const Napi::CallbackInfo& info);

  /**
   * [async] run the model.
   * @param arg0 input object: all keys must present, value is object
   * @param arg1 output object: at least one key must present, value can be null.
   * @returns a Promise resolving to an object that contains every specified output
   * @throw error if status code != 0
   */
  Napi::Value Run(const Napi::CallbackInfo& info);

  /**
   * [sync] dispose the session.
   * @param nothing
   * @returns nothing
   * @throw nothing
   */
  Napi::Value Dispose(const Napi::CallbackInfo& info);

  /**
   * [sync] end the profiling.
   * @param nothing
   * @returns nothing
   * @throw nothing
   */
  Napi::Value EndProfiling(const Napi::CallbackInfo& info);

  // Run bookkeeping. Every access happens on the JS thread: Run(), Dispose() and EndProfiling() are
  // called from Javascript, and RunAsyncWorker::Complete() runs from the AsyncWorker completion
  // callback, which node-addon-api also invokes there. Execute() never touches these.
  void BeginRun();
  void EndRun();
  // Settle and drain a run that failed during preparation, without double-counting or double-settling
  // a worker that already owns them.
  void FailRun(const std::unique_ptr<RunAsyncWorker>& worker, Napi::Promise::Deferred& deferred, Napi::Value error);
  // Release the ORT objects. Deferred until the last in-flight run finishes if one is outstanding.
  void TeardownSession();

  // private members

  // session objects
  bool initialized_;
  bool disposed_;
  // Set when dispose() is called while runs are still in flight; EndRun() completes the teardown.
  bool teardown_pending_{false};
  size_t active_runs_{0};
  std::unique_ptr<Ort::Session> session_;
  // Serializes the IoBinding path. ORT guards graph execution itself for execution providers whose
  // ConcurrentRunSupported() is false (WebGPU is one), but binding inputs and outputs performs
  // device work before Run() is reached, outside that guard. Two concurrent runs would then encode
  // onto the same command encoder and the provider would fail with "command encoding already
  // finished". The plain Run path needs no lock: it does no device work outside ORT's own.
  std::mutex io_binding_mutex_;

  // input/output metadata
  std::vector<std::string> inputNames_;
  std::vector<Ort::TypeInfo> inputTypes_;
  std::vector<std::string> outputNames_;
  std::vector<Ort::TypeInfo> outputTypes_;

  // preferred output locations
  std::vector<int> preferredOutputLocations_;
};

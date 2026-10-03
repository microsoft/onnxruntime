// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_cxx_api.h"

#include <memory>
#include <napi.h>

// class InferenceSessionWrap is a N-API object wrapper for native InferenceSession.
class InferenceSessionWrap : public Napi::ObjectWrap<InferenceSessionWrap> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  InferenceSessionWrap(const Napi::CallbackInfo& info);
  ~InferenceSessionWrap();

 private:
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
   * @param arg0 input object: keys present in the object that match model input names are forwarded; missing keys are
   *             silently ignored (ORT will error if a required input is absent).
   * @param arg1 output object: at least one key must match a model output name; values must be null (preallocated
   *             output tensors are not supported for async run — use runSync() if preallocated outputs are needed).
   * @param arg2 (optional) run options object
   * @returns a Promise that resolves to an object where every requested output is present and value must be object.
   *          The Promise rejects on inference error or invalid arguments detected after Promise creation.
   *          Argument validation errors (wrong types, IO binding active) throw synchronously before the Promise is created.
   * @note Callers must not mutate or transfer input/output ArrayBuffers while the Promise is pending.
   */
  Napi::Value Run(const Napi::CallbackInfo& info);

  /**
   * [sync] run the model.
   * @param arg0 input object: all keys must present, value is object
   * @param arg1 output object: at least one key must present, value can be null.
   * @returns an object that every output specified will present and value must be object
   * @throw error if status code != 0
   */
  Napi::Value RunSync(const Napi::CallbackInfo& info);

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

  // private members

  // session objects
  bool initialized_;
  bool disposed_;
  int inFlightCount_;
  std::unique_ptr<Ort::Session> session_;

  // input/output metadata
  std::vector<std::string> inputNames_;
  std::vector<Ort::TypeInfo> inputTypes_;
  std::vector<std::string> outputNames_;
  std::vector<Ort::TypeInfo> outputTypes_;

  // preferred output locations
  std::vector<int> preferredOutputLocations_;
  std::unique_ptr<Ort::IoBinding> ioBinding_;
};

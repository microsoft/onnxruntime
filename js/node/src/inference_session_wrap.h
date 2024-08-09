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
  InferenceSessionWrap(const Napi::CallbackInfo &info);

private:
  /**
   * [sync] list supported backend list
   * @returns array with objects { "name": "cpu", requirementsInstalled: true }
   */
  static Napi::Value ListSupportedBackends(const Napi::CallbackInfo &info);

  /**
   * [sync] create the session.
   * @param arg0 either a string (file path) or a Uint8Array
   * @returns nothing
   * @throw error if status code != 0
   */
  Napi::Value LoadModel(const Napi::CallbackInfo &info);

  // following functions have to be called after model is loaded.

  /**
   * [sync] get input names.
   * @param nothing
   * @returns a string array.
   * @throw nothing
   */
  Napi::Value GetInputNames(const Napi::CallbackInfo &info);
  /**
   * [sync] get output names.
   * @param nothing
   * @returns a string array.
   * @throw nothing
   */
  Napi::Value GetOutputNames(const Napi::CallbackInfo &info);

  /**
   * [sync] run the model.
   * @param arg0 input object: all keys must present, value is object
   * @param arg1 output object: at least one key must present, value can be null.
   * @returns an object that every output specified will present and value must be object
   * @throw error if status code != 0
   */
  Napi::Value Run(const Napi::CallbackInfo &info);

  /**
   * [sync] dispose the session.
   * @param nothing
   * @returns nothing
   * @throw nothing
   */
  Napi::Value Dispose(const Napi::CallbackInfo &info);

  // private members

  // persistent constructor
  static Napi::FunctionReference constructor;

  // session objects
  bool initialized_;
  bool disposed_;
  std::unique_ptr<Ort::Session> session_;
  std::unique_ptr<Ort::RunOptions> defaultRunOptions_;

  // input/output metadata
  std::vector<std::string> inputNames_;
  std::vector<ONNXType> inputTypes_;
  std::vector<ONNXTensorElementDataType> inputTensorElementDataTypes_;
  std::vector<std::string> outputNames_;
  std::vector<ONNXType> outputTypes_;
  std::vector<ONNXTensorElementDataType> outputTensorElementDataTypes_;
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "onnxruntime_cxx_api.h"

#include <memory>
#include <napi.h>
#include <string>
#include <vector>

#include "ep_context_data_read_helper.h"
#include "ort_singleton_data.h"

// LoadModelWorker performs the native session construction off the JavaScript thread.
class LoadModelWorker;

/**
 * The native session and everything bound to its lifetime.
 *
 * It is shared by the wrapper, by every in-flight run and by every gpu-buffer Tensor the session produced, and is
 * destroyed when the last of them lets go: dispose() only drops the wrapper's reference. Nothing here changes once
 * the session is published, so runs read it from worker threads without synchronization.
 */
struct SessionCore {
  // Creates an empty core, deleted under the DeviceLock if `requiresDeviceSerialization`.
  static std::shared_ptr<SessionCore> Create(bool requiresDeviceSerialization,
                                             std::shared_ptr<EpContextDataReadState> epContextDataReadState);

  // Keeps ORT initialized for as long as the session can be released, which may be after the environment cleanup
  // hook has run. Declared first so that it is released last.
  OrtSingletonData::Reference ort;
  // The session may call back into this state, so it must outlive the session.
  std::shared_ptr<EpContextDataReadState> epContextDataReadState;
  bool requiresDeviceSerialization{false};

  Ort::Session session{nullptr};
  // The session's own CPU allocator, used for input copies. Values allocated from it are released before the core.
  Ort::Allocator cpuAllocator{nullptr};

  std::vector<std::string> inputNames;
  std::vector<Ort::TypeInfo> inputTypes;
  std::vector<std::string> outputNames;
  std::vector<Ort::TypeInfo> outputTypes;
  // Empty unless the session binds its outputs through IoBinding.
  std::vector<int> preferredOutputLocations;

  // Reads names, types and the CPU allocator from `session`.
  void ReadSessionMetadata();
};

// class InferenceSessionWrap is a N-API object wrapper for native InferenceSession.
class InferenceSessionWrap : public Napi::ObjectWrap<InferenceSessionWrap> {
 public:
  static Napi::Object Init(Napi::Env env, Napi::Object exports);

  InferenceSessionWrap(const Napi::CallbackInfo& info);
  ~InferenceSessionWrap();

 private:
  friend class ::LoadModelWorker;

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
   * Create the session and return a Promise.
   *
   * Sessions with `sessionOptions.epContextDataRead` are constructed on a worker thread so that the
   * JavaScript event loop stays available for the callback. Other sessions retain the synchronous,
   * zero-copy native construction path and return an already-settled Promise.
   *
   * @param arg0 either a string (file path) or an ArrayBuffer
   * @returns a Promise that resolves when the session is created
   * @throw error if the arguments are invalid or the session options cannot be parsed
   */
  Napi::Value LoadModel(const Napi::CallbackInfo& info);

  Napi::Value LoadModelSynchronously(const Napi::CallbackInfo& info, bool isModelPath,
                                     const Napi::Object& options, Ort::SessionOptions&& sessionOptions,
                                     bool requiresDeviceSerialization);

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
   * @param arg2 run options object (optional)
   * @returns a Promise resolving to an object that contains every specified output
   * @throw error if the session is not usable or an argument is invalid
   */
  Napi::Value Run(const Napi::CallbackInfo& info);

  /**
   * [sync] dispose the session. Runs still in flight complete normally; the native session is released after them.
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

  // Take over a loaded session. Runs on the JavaScript thread.
  // @throw Napi::Error if the preferred output locations are invalid.
  void AdoptLoadedSession(std::shared_ptr<SessionCore> core, const Napi::Object options);

  // Reset the session state after a failed load. Runs on the JavaScript thread.
  void ResetAfterFailedLoad() noexcept;

  // Give up the EPContext data read callback state. Must run after the native session was released.
  void ReleaseEpContextDataReadState() noexcept;

  // private members

  // The EPContext data read callback state is declared first so that it is destroyed last: the native
  // session must always be released before the state that it can call into.
  std::shared_ptr<EpContextDataReadState> epContextDataReadState_;

  // session objects
  bool initialized_;
  bool loading_;
  bool disposed_;
  std::shared_ptr<SessionCore> core_;
};

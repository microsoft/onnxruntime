// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

"use strict";

//
// This file contains the pre-run code for the ORT WebAssembly module. The code in this file will be injected into the
// final module using Emscripten's `--pre-js` option.
//
// This file will only be used in build with flag `--use_jsep`.

// This is a wrapper for OrtRun() and OrtRunWithBinding() to ensure that Promises are handled correctly.
const jsepRunAsync = (runAsyncFunc) => {
  return async (...args) => {
    try {
      // Module.jsepSessionState should be null, unless we are in the middle of a session.
      // If it is not null, it means that the previous session has not finished yet.
      if (Module.jsepSessionState) {
        throw new Error("Session already started");
      }
      const state = (Module.jsepSessionState = {
        sessionHandle: args[0],
        errors: [],
      });

      // Run the acyncified function: OrtRun() or OrtRunWithBinding()
      const ret = await runAsyncFunc(...args);

      // Check if the session is still valid. this object should be the same as the one we set above.
      if (Module.jsepSessionState !== state) {
        throw new Error("Session mismatch");
      }

      // Flush the backend. This will submit all pending commands to the GPU.
      Module.jsepBackend?.["flush"]();

      // Await all pending promises. This includes GPU validation promises for diagnostic purposes.
      const errorPromises = state.errors;
      if (errorPromises.length > 0) {
        let errors = await Promise.all(errorPromises);
        errors = errors.filter((e) => e);
        if (errors.length > 0) {
          throw new Error(errors.join("\n"));
        }
      }

      return ret;
    } finally {
      Module.jsepSessionState = null;
    }
  };
};

/**
 * initialize JSEP for WebGPU and WebNN.
 */
Module["jsepInit"] = (name, params) => {
  if (name === "webgpu") {
    [
      Module.jsepBackend,
      Module.jsepAlloc,
      Module.jsepFree,
      Module.jsepCopy,
      Module.jsepCopyAsync,
      Module.jsepCreateKernel,
      Module.jsepReleaseKernel,
      Module.jsepRunKernel,
      Module.jsepCaptureBegin,
      Module.jsepCaptureEnd,
      Module.jsepReplay,
    ] = params;

    // expose webgpu backend functions
    const backend = Module.jsepBackend;
    Module["jsepRegisterBuffer"] = (sessionId, index, buffer, size) => {
      return backend["registerBuffer"](sessionId, index, buffer, size);
    };
    Module["jsepGetBuffer"] = (dataId) => {
      return backend["getBuffer"](dataId);
    };
    Module["jsepCreateDownloader"] = (gpuBuffer, size, type) => {
      return backend["createDownloader"](gpuBuffer, size, type);
    };
    Module["jsepOnCreateSession"] = (sessionId) => {
      backend["onCreateSession"](sessionId);
    };
    Module["jsepOnReleaseSession"] = (sessionId) => {
      backend["onReleaseSession"](sessionId);
    };
    Module["jsepOnRunStart"] = (sessionId) => {
      return backend["onRunStart"](sessionId);
    };

    Module.jsepUploadExternalBuffer = (dataId, buffer) => {
      backend["upload"](dataId, buffer);
    };
  } else if (name === "webnn") {
    // Functions called from EM_ASM need to be assigned in a way that can be minified.
    // Functions called via emscripten::val::module_property need to be assigned by name so that the minifier doesn't
    // change the name.

    [
      Module.jsepBackend,
      Module.jsepReserveTensorId,
      Module.jsepReleaseTensorId,
      Module["jsepEnsureTensor"],
      Module.jsepUploadTensor,
      Module["jsepDownloadTensor"],
    ] = params;

    // This function is called from both JS and an EM_ASM block, it needs both a minifiable name and an explicit name.
    Module["jsepReleaseTensorId"] = Module.jsepReleaseTensorId;
    Module["jsepUploadTensor"] = Module.jsepUploadTensor;

    // Functions called from JS also need to have explicit names.
    const backend = Module.jsepBackend;
    Module["jsepOnRunStart"] = (sessionId) => {
      return backend["onRunStart"](sessionId);
    };
    Module["jsepOnRunEnd"] = backend["onRunEnd"].bind(backend);
    Module["jsepRegisterMLContext"] = (sessionId, mlContext) => {
      backend["registerMLContext"](sessionId, mlContext);
    };
    Module["jsepOnReleaseSession"] = (sessionId) => {
      backend["onReleaseSession"](sessionId);
    };
    Module["jsepCreateMLTensorDownloader"] = (tensorId, type) => {
      return backend["createMLTensorDownloader"](tensorId, type);
    };
    Module["jsepRegisterMLTensor"] = (sessionId, tensor, dataType, shape) => {
      return backend["registerMLTensor"](sessionId, tensor, dataType, shape);
    };
    Module["jsepCreateMLContext"] = (optionsOrGpuDevice) => {
      return backend["createMLContext"](optionsOrGpuDevice);
    };
    Module["jsepRegisterMLConstant"] = (
      externalFilePath,
      dataOffset,
      dataLength,
      builder,
      desc
    ) => {
      return backend["registerMLConstant"](
        externalFilePath,
        dataOffset,
        dataLength,
        builder,
        desc,
        Module.MountedFiles
      );
    };
    Module["jsepRegisterGraphInput"] =
      backend["registerGraphInput"].bind(backend);
    Module["jsepIsGraphInput"] = backend["isGraphInput"].bind(backend);

    Module["jsepCreateTemporaryTensor"] =
      backend["createTemporaryTensor"].bind(backend);
  }
};

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// init JSEP
Module['jsepInit'] = (backend, alloc, free, copy, copyAsync, createKernel, releaseKernel, runKernel) => {
  Module.jsepBackend = backend;
  Module.jsepAlloc = alloc;
  Module.jsepFree = free;
  Module.jsepCopy = copy;
  Module.jsepCopyAsync = copyAsync;
  Module.jsepCreateKernel = createKernel;
  Module.jsepReleaseKernel = releaseKernel;
  Module.jsepRunKernel = runKernel;

  // This is a simplified version of cwrap() with options.async === true (-sASYNCIFY=1)
  // It removes some overhead in cwarp() and ccall() that we don't need.
  //
  // Currently in JSEP build, we only use this for the following functions:
  // - OrtRun()
  // - OrtRunWithBinding()
  // - OrtBindInput()
  const jsepWrapAsync = (func) => {
    return (...args) => {
      const previousAsync = Asyncify.currData;
      const ret = func(...args);
      if (Asyncify.currData != previousAsync) {
        return Asyncify.whenDone();
      }
      return ret;
    };
  };

  // This is a wrapper for OrtRun() and OrtRunWithBinding() to ensure that Promises are handled correctly.
  const runAsync = (runAsyncFunc) => {
    return async (...args) => {
      try {
        // Module.jsepSessionState should be null, unless we are in the middle of a session.
        // If it is not null, it means that the previous session has not finished yet.
        if (Module.jsepSessionState) {
          throw new Error('Session already started');
        }
        const state = Module.jsepSessionState = {sessionHandle: args[0], errors: []};

        // Run the acyncified function: OrtRun() or OrtRunWithBinding()
        const ret = await runAsyncFunc(...args);

        // Check if the session is still valid. this object should be the same as the one we set above.
        if (Module.jsepSessionState !== state) {
          throw new Error('Session mismatch');
        }

        // Flush the backend. This will submit all pending commands to the GPU.
        backend['flush']();

        // Await all pending promises. This includes GPU validation promises for diagnostic purposes.
        const errorPromises = state.errors;
        if (errorPromises.length > 0) {
          let errors = await Promise.all(errorPromises);
          errors = errors.filter(e => e);
          if (errors.length > 0) {
            throw new Error(errors.join('\n'));
          }
        }

        return ret;
      } finally {
        Module.jsepSessionState = null;
      }
    };
  };

  // replace the original functions with asyncified versions
  Module['_OrtRun'] = runAsync(jsepWrapAsync(Module['_OrtRun']));
  Module['_OrtRunWithBinding'] = runAsync(jsepWrapAsync(Module['_OrtRunWithBinding']));
  Module['_OrtBindInput'] = jsepWrapAsync(Module['_OrtBindInput']);

  // expose webgpu backend functions
  Module['jsepRegisterBuffer'] = (sessionId, index, buffer, size) => {
    return backend['registerBuffer'](sessionId, index, buffer, size);
  };
  Module['jsepUnregisterBuffers'] = sessionId => {
    backend['unregisterBuffers'](sessionId);
  };
  Module['jsepGetBuffer'] = (dataId) => {
    return backend['getBuffer'](dataId);
  };
  Module['jsepCreateDownloader'] = (gpuBuffer, size, type) => {
    return backend['createDownloader'](gpuBuffer, size, type);
  };
};

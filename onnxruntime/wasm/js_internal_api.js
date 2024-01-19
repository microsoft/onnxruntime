// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

/**
 * Mount external data files of a model to the virtual file system (MEMFS).
 *
 * @param {string} externalDataFilesPath
 * @param {Uint8Array} externalDataFilesData
 */
Module['mountExternalData'] = (externalDataFilePath, externalDataFileData) => {
  const files = Module.MountedFiles || (Module.MountedFiles = new Map());
    files.set(externalDataFilePath, externalDataFileData);
};

/**
 * Unmount external data files of a model from the virtual file system (MEMFS).
 */
Module['unmountExternalData'] = () => {
  delete Module.MountedFiles;
};

/**
 * init JSEP
 */
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
  //
  // Note: about parameters "getFunc" and "setFunc":
  // - Emscripten has different behaviors for Debug and Release builds for generating exported function wrapper.
  //
  //   - In Debug build, it will generate a wrapper function for each exported function. For example, it generates a
  //     wrapper for OrtRun() like this (minified):
  //     ```
  //     var _OrtRun = Module["_OrtRun"] = createExportWrapper("OrtRun");
  //     ```
  //
  //   - In Release build, it will generate a lazy loading wrapper for each exported function. For example, it generates
  //     a wrapper for OrtRun() like this (minified):
  //     ```
  //     d._OrtRun = (a, b, c, e, f, h, l, q) => (d._OrtRun = J.ka)(a, b, c, e, f, h, l, q);
  //     ```
  //
  //   The behavior of these two wrappers are different. The debug build will assign `Module["_OrtRun"]` only once
  //   because `createExportWrapper()` does not reset `Module["_OrtRun"]` inside. The release build, however, will
  //   reset d._OrtRun to J.ka when the first time it is called.
  //
  //   The difference is important because we need to design the async wrapper in a way that it can handle both cases.
  //
  //   Now, let's look at how the async wrapper is designed to work for both cases:
  //
  //   - Debug build:
  //      1. When Web assembly is being loaded, `Module["_OrtRun"]` is assigned to `createExportWrapper("OrtRun")`.
  //      2. When the first time `Module["jsepInit"]` is called, `Module["_OrtRun"]` is re-assigned to a new async
  //         wrapper function.
  //      Value of `Module["_OrtRun"]` will not be changed again.
  //
  //   - Release build:
  //      1. When Web assembly is being loaded, `Module["_OrtRun"]` is assigned to a lazy loading wrapper function.
  //      2. When the first time `Module["jsepInit"]` is called, `Module["_OrtRun"]` is re-assigned to a new async
  //         wrapper function.
  //      3. When the first time `Module["_OrtRun"]` is called, the async wrapper will be called. It will call into this
  //         function:
  //         ```
  //         (a, b, c, e, f, h, l, q) => (d._OrtRun = J.ka)(a, b, c, e, f, h, l, q);
  //         ```
  //         This function will assign d._OrtRun (ie. the minimized `Module["_OrtRun"]`) to the real function (J.ka).
  //      4. Since d._OrtRun is re-assigned, we need to update the async wrapper to re-assign its stored
  //         function to the updated value (J.ka), and re-assign the value of `d._OrtRun` back to the async wrapper.
  //      Value of `Module["_OrtRun"]` will not be changed again.
  //
  //   The value of `Module["_OrtRun"]` will need to be assigned for 2 times for debug build and 4 times for release
  //   build.
  //
  //   This is why we need this `getFunc` and `setFunc` parameters. They are used to get the current value of an
  //   exported function and set the new value of an exported function.
  //
  const jsepWrapAsync = (func, getFunc, setFunc) => {
    return (...args) => {
      // cache the async data before calling the function.
      const previousAsync = Asyncify.currData;

      const previousFunc = getFunc?.();
      const ret = func(...args);
      const newFunc = getFunc?.();
      if (previousFunc !== newFunc) {
        // The exported function has been updated.
        // Set the sync function reference to the new function.
        func = newFunc;
        // Set the exported function back to the async wrapper.
        setFunc(previousFunc);
        // Remove getFunc and setFunc. They are no longer needed.
        setFunc = null;
        getFunc = null;
      }

      // If the async data has been changed, it means that the function started an async operation.
      if (Asyncify.currData != previousAsync) {
        // returns the promise
        return Asyncify.whenDone();
      }
      // the function is synchronous. returns the result.
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
  Module['_OrtRun'] = runAsync(jsepWrapAsync(
      Module['_OrtRun'],
      () => Module['_OrtRun'],
      v => Module['_OrtRun'] = v));
  Module['_OrtRunWithBinding'] = runAsync(jsepWrapAsync(
      Module['_OrtRunWithBinding'],
      () => Module['_OrtRunWithBinding'],
      v => Module['_OrtRunWithBinding'] = v));
  Module['_OrtBindInput'] = jsepWrapAsync(
      Module['_OrtBindInput'],
      () => Module['_OrtBindInput'],
      v => Module['_OrtBindInput'] = v);

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

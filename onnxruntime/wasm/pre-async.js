// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

//
// This file contains the pre-run code for the ORT WebAssembly module. The code in this file will be injected into the
// final module using Emscripten's `--pre-js` option.
//
// This file will only be used in build with flag `-s ASYNCIFY=1`.

/**
 * initialize for asyncify support.
 */
let initAsyncImpl = () => {
  // This is a simplified version of cwrap() with options.async === true (-sASYNCIFY=1)
  // It removes some overhead in cwarp() and ccall() that we don't need.
  //
  // Currently in ASYNCIFY build, we only use this for the following functions:
  // - OrtAppendExecutionProvider()
  // - OrtCreateSession()
  // - OrtRun()
  // - OrtRunWithBinding()
  // - OrtBindInput()
  //
  // We need to wrap these functions with an async wrapper so that they can be called in an async context.
  //
  const wrapAsync = (func) => {
    return (...args) => {
      // cache the async data before calling the function.
      const previousAsync = Asyncify.currData;

      const ret = func(...args);

      // If the async data has been changed, it means that the function started an async operation.
      if (Asyncify.currData != previousAsync) {
        // returns the promise
        return Asyncify.whenDone();
      }
      // the function is synchronous. returns the result.
      return ret;
    };
  };

  // replace the original functions with asyncified versions
  const wrapAsyncAPIs = (funcNames) => {
    for (const funcName of funcNames) {
      Module[funcName] = wrapAsync(Module[funcName]);
    }
  };

  wrapAsyncAPIs([
    "_OrtAppendExecutionProvider",
    "_OrtCreateSession",
    "_OrtRun",
    "_OrtRunWithBinding",
    "_OrtBindInput",
  ]);

  // If JSEP is enabled, wrap OrtRun() and OrtRunWithBinding() with asyncify.
  if (typeof jsepRunAsync !== "undefined") {
    Module["_OrtRun"] = jsepRunAsync(Module["_OrtRun"]);
    Module["_OrtRunWithBinding"] = jsepRunAsync(Module["_OrtRunWithBinding"]);
  }

  // remove this function to make sure it is called only once.
  initAsyncImpl = undefined;
};

Module["asyncInit"] = () => {
  initAsyncImpl?.();
};

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
  // - OrtCreateSession()
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
  //      2. When the first time `Module["initAsync"]` is called, `Module["_OrtRun"]` is re-assigned to a new async
  //         wrapper function.
  //      Value of `Module["_OrtRun"]` will not be changed again.
  //
  //   - Release build:
  //      1. When Web assembly is being loaded, `Module["_OrtRun"]` is assigned to a lazy loading wrapper function.
  //      2. When the first time `Module["initAsync"]` is called, `Module["_OrtRun"]` is re-assigned to a new async
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
  const wrapAsync = (func, getFunc, setFunc) => {
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

  // replace the original functions with asyncified versions
  const wrapAsyncAPIs = (funcNames) => {
    for (const funcName of funcNames) {
      Module[funcName] = wrapAsync(
        Module[funcName],
        () => Module[funcName],
        (v) => (Module[funcName] = v)
      );
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

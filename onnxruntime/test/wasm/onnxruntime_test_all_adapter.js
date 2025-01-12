// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// This file is used to be injected into "onnxruntime_test_all" as specified by flag "--pre-js" by emcc.
// It dumps the test report file from emscripten's MEMFS to real file system

// Module is predefined for scripts injected from "--pre-js"
(function () {
    let args = [];
    let onTestComplete;
    let gtestOutputFilepath = '';
    let gtestOutputFiledata;
    if (typeof process !== 'undefined') {
        // In Node.js
        args = process.argv;
        globalThis.PTHREAD_POOL_SIZE = Math.min(8, require('os').cpus().length);
        onTestComplete = function () {
            if (gtestOutputFilepath) {
                require('fs').writeFileSync(gtestOutputFilepath, gtestOutputFiledata);
            }
        };
    } else if (typeof __karma__ !== 'undefined') {
        // In browser (launched by karma)
        args = __karma__.config.args;
        globalThis.PTHREAD_POOL_SIZE = Math.min(8, navigator.hardwareConcurrency);
        onTestComplete = function (exitCode) {
            __karma__.result({
                id: '',
                description: '',
                suite: [],
                success: exitCode === 0,
                log: []
              });
            __karma__.complete({file: gtestOutputFilepath, data: gtestOutputFiledata});
        };

        Module["arguments"] = args;
        __karma__.info({total: 1});
        __karma__.start = function () {};
    }

    // check for flag "--gtest_output=xml:"
    const argGtestOutputPrefix = '--gtest_output=xml:';

    // check for flag "--wasm-threads="
    const argWasmThreads = '--wasm-threads=';

    for (const arg of args) {
        if (arg.startsWith(argGtestOutputPrefix)) {
            gtestOutputFilepath = arg.substring(argGtestOutputPrefix.length);
        } else if (arg.startsWith(argWasmThreads)) {
            globalThis.PTHREAD_POOL_SIZE = Number.parseInt(arg.substring(argWasmThreads.length));
        }
    }

    Module["onExit"] = function(exitCode) {
        if (gtestOutputFilepath) {
            gtestOutputFiledata = FS.readFile(gtestOutputFilepath);
        }
        onTestComplete(exitCode);
    };

})();

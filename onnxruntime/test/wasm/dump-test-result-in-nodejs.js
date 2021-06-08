// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This file is used to be injected into "onnxruntime_test_all" as specified by flag "--pre-js" by emcc.
// It dumps the test report file from emscripten's MEMFS to real file system

// Module is predefined for scripts injected from "--pre-js"
(function () {
    if (typeof process !== 'undefined') {
        // check for flag "--gtest_output=xml:"
        const argGtestOutputPrefix = '--gtest_output=xml:';

        for (const arg of process.argv) {
            if (arg.startsWith(argGtestOutputPrefix)) {
                const filename = arg.substring(argGtestOutputPrefix.length);
                if (filename) {
                    Module["onExit"] = function () {
                        // read test output from MEMFS and write to real file system.
                        const filedata = Module.FS.readFile(filename);
                        require('fs').writeFileSync(filename, filedata);
                    };
                }
                break;
            }
        }
    }
})();


// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.


const fs = require('node:fs');
const path = require('node:path');

//
// Post-process the mjs file.
//

//
// USAGE: node wasm_post_build.js <mjsFilePath>
//

const mjsFilePath = process.argv[2];
let contents = fs.readFileSync(mjsFilePath).toString();

// STEP.1 - Apply workaround for spawning workers using `import.meta.url`.

// This script is a workaround for enabling:
// - using onnxruntime-web with Multi-threading enabled when import from CDN
// - using onnxruntime-web when consumed in some frameworks like Vite
//
// In the use case mentioned above, the file name of the script may be changed. So we need to replace the line:
// `new Worker(new URL("ort-wasm-*.mjs", import.meta.url),`
// with
// `new Worker(new URL(import.meta.url),`
//
// This behavior is introduced in https://github.com/emscripten-core/emscripten/pull/22165. Since it's unlikely to be
// reverted, and there is no config to disable this behavior, we have to use a post-build script to workaround it.

// This step should only be applied for multithreading builds
if (path.basename(mjsFilePath).includes('-threaded')) {
    const regex = 'new Worker\\(new URL\\(".+?", ?import\\.meta\\.url\\),';
    const matches = [...contents.matchAll(new RegExp(regex, 'g'))];
    if (matches.length !== 1) {
        throw new Error(
            `Unexpected number of matches for "${regex}" in "${mjsFilePath}": ${matches.length}.`,
        );
    }

    // Replace the only occurrence.
    contents = contents.replace(
        new RegExp(regex),
        `new Worker(new URL(import.meta.url),`,
    );
}

// STEP.2 - Workaround the issue referred in https://issues.chromium.org/issues/435879324

// Closure compiler will minimize the key of object `FeatureNameString2Enum`, turning `subgroup` into something else.

// This workaround is to replace the generated code as following:
//
// (for debug build)
//
// > subgroups: "17",
// --- change to -->
// > "subgroups": "17",
//
// (for release build)
//
// > Pe:"17",
// --- change to -->
// > "subgroups":"17",
//

// This step should only be applied for WebGPU EP builds
if (path.basename(mjsFilePath).includes('.async')) {
    const regexDebug = 'subgroups: "17"';
    const regexRelease = '[a-zA-Z_$][a-zA-Z0-9_$]*:"17"';

    const matchesDebug = [...contents.matchAll(new RegExp(regexDebug, 'g'))];
    const matchesRelease = [...contents.matchAll(new RegExp(regexRelease, 'g'))];

    if (matchesDebug.length === 1 && matchesRelease.length === 0) {
        contents = contents.replace(
            new RegExp(regexDebug),
            '"subgroups": "17"',
        );
    } else if (matchesDebug.length === 0 && matchesRelease.length === 1) {
        contents = contents.replace(
            new RegExp(regexRelease),
            '"subgroups":"17"',
        );
    } else {
        throw new Error(
            `Unexpected number of matches for "${regexDebug}" and "${regexRelease}" in "${mjsFilePath}": Debug=${matchesDebug.length}, Release=${matchesRelease.length}.`,
        );
    }
}


fs.writeFileSync(mjsFilePath, contents);

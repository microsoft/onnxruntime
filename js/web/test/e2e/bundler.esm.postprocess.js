// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

// This is a simple script to postprocess the output of some bundlers (webpack, parcel).
//
// Some bundlers may rewrite `import.meta.url` into a local path (file://). This behavior is not
// what we want because onnxruntime-web need `import.meta.url` to be evaluated at runtime.
//
// This script will replace the local path back to `import.meta.url` in the output files.
//

// Usage:
//
// node bundler.esm.postprocess.js <output-file>
//

const fs = require('node:fs');

const inputFilePath = process.argv[2];

if (!inputFilePath || !fs.existsSync(inputFilePath)) {
  console.error('Please specify a valid input file.');
  process.exit(1);
}

const content = fs.readFileSync(inputFilePath, 'utf8');

// replace all `"file://*/ort.*.mjs"` paths back to `import.meta.url`. Try to keep the same length to make source map
// work.
const updatedContent = content.replace(/['"]file:\/\/.+?\/ort\..+?\.mjs['"]/g, (match) => {
  return 'import.meta.url'.padEnd(match.length, ' ');
});

fs.writeFileSync(inputFilePath, updatedContent, 'utf8');

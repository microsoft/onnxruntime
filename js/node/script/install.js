// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// This script is written in JavaScript. This is because it is used in "install" script in package.json, which is called
// when the package is installed either as a dependency or from "npm ci"/"npm install" without parameters. TypeScript is
// not always available.

// The purpose of this script is to download the required binaries for the platform and architecture.
// Currently, most of the binaries are already bundled in the package, except for the following:
// - Linux/x64/CUDA 11
// - Linux/x64/CUDA 12
//
// The CUDA binaries are not bundled because they are too large to be allowed in the npm registry. Instead, they are
// downloaded from the GitHub release page of ONNX Runtime. The script will download the binaries if they are not
// already present in the package.

// Step.1: Check if we should exit early
const os = require('os');
const fs = require('fs');
const path = require('path');
const tar = require('tar');
const {Readable} = require('stream');

const FORCE_INSTALL = process.argv.includes('--onnxruntime-node-force-install-cuda') ||
    process.env.npm_config_onnxruntime_node_force_install_cuda;
const NO_INSTALL = process.argv.includes('--onnxruntime-node-no-install-cuda') ||
    process.env.npm_config_onnxruntime_node_no_install_cuda;

const IS_LINUX_X64 = os.platform() === 'linux' && os.arch() === 'x64';
const BIN_FOLDER = path.join(__dirname, '..', 'bin/napi-v3/linux/x64');
const BIN_FOLDER_EXISTS = fs.existsSync(BIN_FOLDER);
const CUDA_DLL_EXISTS = fs.existsSync(path.join(BIN_FOLDER, 'libonnxruntime_providers_cuda.so'));
const ORT_VERSION = require('../package.json').version;

const shouldInstall = FORCE_INSTALL || (IS_LINUX_X64 && BIN_FOLDER_EXISTS && !CUDA_DLL_EXISTS);
if (NO_INSTALL || !shouldInstall) {
  process.exit(0);
}

// Step.2: Download the required binaries
const artifactUrl = {
  11: `https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${
      ORT_VERSION}.tgz`,
  12: `https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-cuda12-${
      ORT_VERSION}.tgz`
}[tryGetCudaVersion()];
console.log(`Downloading "${artifactUrl}"...`);
fetch(artifactUrl).then(res => {
  if (!res.ok) {
    throw new Error(`Failed to download the binaries: ${res.status} ${res.statusText}.

Use "--onnxruntime-node-no-install-cuda" to skip the installation. You will still be able to use ONNX Runtime, but the CUDA EP will not be available.`);
  }

  // Extract the binaries

  const FILES = new Set([
    'libonnxruntime_providers_tensorrt.so',
    'libonnxruntime_providers_shared.so',
    `libonnxruntime.so.${ORT_VERSION}`,
    'libonnxruntime_providers_cuda.so',
  ]);

  Readable.fromWeb(res.body)
      .pipe(tar.t())
      .on('entry',
          (entry) => {
            const filename = path.basename(entry.path);
            if (entry.type === 'File' && FILES.has(filename)) {
              console.log(`Extracting "${filename}" to "${BIN_FOLDER}"...`);
              entry.pipe(fs.createWriteStream(path.join(BIN_FOLDER, filename)));
            }
          })
      .on('error', (err) => {
        throw new Error(`Failed to extract the binaries: ${err.message}.

Use "--onnxruntime-node-no-install-cuda" to skip the installation. You will still be able to use ONNX Runtime, but the CUDA EP will not be available.`);
      });
});


function tryGetCudaVersion() {
  // Should only return 11 or 12.

  // TODO: try to get the CUDA version from the system ( `nvcc --version` )

  return 11;
}

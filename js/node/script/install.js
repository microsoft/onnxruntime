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
const { Readable } = require('stream');

// commandline flag:
// --onnxruntime-node-install-cuda         Force install the CUDA EP binaries. Try to detect the CUDA version.
// --onnxruntime-node-install-cuda=v11     Force install the CUDA EP binaries for CUDA 11.
// --onnxruntime-node-install-cuda=v12     Force install the CUDA EP binaries for CUDA 12.
// --onnxruntime-node-install-cuda=skip    Skip the installation of the CUDA EP binaries.
//
// Alternatively, use environment variable "ONNXRUNTIME_NODE_INSTALL_CUDA"
//
// If the flag is not provided, the script will only install the CUDA EP binaries when:
// - The platform is Linux/x64.
// - The binaries are not already present in the package.
// - The installation is not a local install (when used inside ONNX Runtime repo).
//
const INSTALL_CUDA_FLAG = parseInstallCudaFlag();
const NO_INSTALL = INSTALL_CUDA_FLAG === 'skip';
const FORCE_INSTALL = !NO_INSTALL && INSTALL_CUDA_FLAG;

const IS_LINUX_X64 = os.platform() === 'linux' && os.arch() === 'x64';
const BIN_FOLDER = path.join(__dirname, '..', 'bin/napi-v3/linux/x64');
const BIN_FOLDER_EXISTS = fs.existsSync(BIN_FOLDER);
const CUDA_DLL_EXISTS = fs.existsSync(path.join(BIN_FOLDER, 'libonnxruntime_providers_cuda.so'));
const ORT_VERSION = require('../package.json').version;

const npm_config_local_prefix = process.env.npm_config_local_prefix;
const npm_package_json = process.env.npm_package_json;
const SKIP_LOCAL_INSTALL =
  npm_config_local_prefix && npm_package_json && path.dirname(npm_package_json) === npm_config_local_prefix;

const shouldInstall = FORCE_INSTALL || (!SKIP_LOCAL_INSTALL && IS_LINUX_X64 && BIN_FOLDER_EXISTS && !CUDA_DLL_EXISTS);
if (NO_INSTALL || !shouldInstall) {
  process.exit(0);
}

// Step.2: Download the required binaries
const artifactUrl = {
  11: `https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-${
    ORT_VERSION
  }.tgz`,
  12: `https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/onnxruntime-linux-x64-gpu-cuda12-${
    ORT_VERSION
  }.tgz`,
}[INSTALL_CUDA_FLAG || tryGetCudaVersion()];
console.log(`Downloading "${artifactUrl}"...`);
fetch(artifactUrl).then((res) => {
  if (!res.ok) {
    throw new Error(`Failed to download the binaries: ${res.status} ${res.statusText}.

Use "--onnxruntime-node-install-cuda=skip" to skip the installation. You will still be able to use ONNX Runtime, but the CUDA EP will not be available.`);
  }

  // Extract the binaries

  const FILES = new Set([
    'libonnxruntime_providers_tensorrt.so',
    'libonnxruntime_providers_shared.so',
    `libonnxruntime.so.${ORT_VERSION}`,
    'libonnxruntime_providers_cuda.so',
  ]);

  Readable.fromWeb(res.body)
    .pipe(
      tar.t({
        strict: true,
        onentry: (entry) => {
          const filename = path.basename(entry.path);
          if (entry.type === 'File' && FILES.has(filename)) {
            console.log(`Extracting "${filename}" to "${BIN_FOLDER}"...`);
            entry.pipe(fs.createWriteStream(path.join(BIN_FOLDER, filename)));
            entry.on('finish', () => {
              console.log(`Finished extracting "${filename}".`);
            });
          }
        },
      }),
    )
    .on('error', (err) => {
      throw new Error(`Failed to extract the binaries: ${err.message}.

Use "--onnxruntime-node-install-cuda=skip" to skip the installation. You will still be able to use ONNX Runtime, but the CUDA EP will not be available.`);
    });
});

function tryGetCudaVersion() {
  // Should only return 11 or 12.

  // TODO: try to get the CUDA version from the system ( `nvcc --version` )

  return 11;
}

function parseInstallCudaFlag() {
  let flag = process.env.ONNXRUNTIME_NODE_INSTALL_CUDA || process.env.npm_config_onnxruntime_node_install_cuda;
  if (!flag) {
    for (let i = 0; i < process.argv.length; i++) {
      if (process.argv[i].startsWith('--onnxruntime-node-install-cuda=')) {
        flag = process.argv[i].split('=')[1];
        break;
      } else if (process.argv[i] === '--onnxruntime-node-install-cuda') {
        flag = 'true';
      }
    }
  }
  switch (flag) {
    case 'true':
    case '1':
    case 'ON':
      return tryGetCudaVersion();
    case 'v11':
      return 11;
    case 'v12':
      return 12;
    case 'skip':
    case undefined:
      return flag;
    default:
      throw new Error(`Invalid value for --onnxruntime-node-install-cuda: ${flag}`);
  }
}

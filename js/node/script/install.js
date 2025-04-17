// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// This script is written in JavaScript. This is because it is used in "install" script in package.json, which is called
// when the package is installed either as a dependency or from "npm ci"/"npm install" without parameters. TypeScript is
// not always available.

// The purpose of this script is to download the required binaries for the platform and architecture.
// Currently, most of the binaries are already bundled in the package, except for the files that described in the file
// install-metadata.js.
//
// Some files (eg. the CUDA EP binaries) are not bundled because they are too large to be allowed in the npm registry.
// Instead, they are downloaded from the Nuget feed. The script will download the binaries if they are not already
// present in the NPM package.

// Step.1: Check if we should exit early
const os = require('os');
const path = require('path');
const { bootstrap: globalAgentBootstrap } = require('global-agent');
const { installPackages, parseInstallFlag } = require('./install-utils.js');

const INSTALL_METADATA = require('./install-metadata.js');

// Bootstrap global-agent to honor the proxy settings in
// environment variables, e.g. GLOBAL_AGENT_HTTPS_PROXY.
// See https://github.com/gajus/global-agent/blob/v3.0.0/README.md#environment-variables for details.
globalAgentBootstrap();

// commandline flag:
//
// --onnxruntime-node-install              Force install the files that are not bundled in the package.
//
// --onnxruntime-node-install=skip         Skip the installation of the files that are not bundled in the package.
//
// --onnxruntime-node-install=cuda12       Force install the CUDA EP binaries for CUDA 12.
//
// --onnxruntime-node-install-cuda         Force install the CUDA EP binaries.
//                                         (deprecated, use --onnxruntime-node-install=cuda12)
//
// --onnxruntime-node-install-cuda=skip    Skip the installation of the CUDA EP binaries.
//                                         (deprecated, use --onnxruntime-node-install=skip)
//
//
// Alternatively, use environment variable "ONNXRUNTIME_NODE_INSTALL" or "ONNXRUNTIME_NODE_INSTALL_CUDA" (deprecated).
//
// If the flag is not provided, the script will look up the metadata file to determine the manifest.
//

/**
 * Possible values:
 * - undefined: the default behavior. This is the value when no installation flag is specified.
 *
 * - false: skip installation. This is the value when the installation flag is set to "skip":
 *   --onnxruntime-node-install=skip
 *
 * - true: force installation. This is the value when the installation flag is set with no value:
 *   --onnxruntime-node-install
 *
 * - string: the installation flag is set to a specific value:
 *   --onnxruntime-node-install=cuda12
 */
const INSTALL_FLAG = parseInstallFlag();

// if installation is skipped, exit early
if (INSTALL_FLAG === false) {
  process.exit(0);
}
// if installation is not specified, exit early when the installation is local (e.g. `npm ci` in <ORT_ROOT>/js/node/)
if (INSTALL_FLAG === undefined) {
  const npm_config_local_prefix = process.env.npm_config_local_prefix;
  const npm_package_json = process.env.npm_package_json;
  const IS_LOCAL_INSTALL =
    npm_config_local_prefix && npm_package_json && path.dirname(npm_package_json) === npm_config_local_prefix;
  if (IS_LOCAL_INSTALL) {
    process.exit(0);
  }
}

const PLATFORM = `${os.platform()}/${os.arch()}`;
let INSTALL_MANIFEST_NAMES = INSTALL_METADATA.requirements[PLATFORM] ?? [];

// if installation is specified explicitly, validate the manifest
if (typeof INSTALL_FLAG === 'string') {
  const installations = INSTALL_FLAG.split(',').map((x) => x.trim());
  for (const installation of installations) {
    if (INSTALL_MANIFEST_NAMES.indexOf(installation) === -1) {
      throw new Error(`Invalid installation: ${installation} for platform: ${PLATFORM}`);
    }
  }
  INSTALL_MANIFEST_NAMES = installations;
}

const BIN_FOLDER = path.join(__dirname, '..', 'bin/napi-v6', PLATFORM);
const INSTALL_MANIFESTS = [];

const PACKAGES = new Set();
for (const name of INSTALL_MANIFEST_NAMES) {
  const manifest = INSTALL_METADATA.manifests[`${PLATFORM}:${name}`];
  if (!manifest) {
    throw new Error(`Manifest not found: ${name} for platform: ${PLATFORM}`);
  }

  for (const [filename, { package: pkg, path: pathInPackage }] of Object.entries(manifest)) {
    const packageCandidates = INSTALL_METADATA.packages[pkg];
    if (!packageCandidates) {
      throw new Error(`Package information not found: ${pkg}`);
    }
    PACKAGES.add(packageCandidates);

    INSTALL_MANIFESTS.push({
      filepath: path.normalize(path.join(BIN_FOLDER, filename)),
      packagesInfo: packageCandidates,
      pathInPackage,
    });
  }
}

// If the installation flag is not specified, we do a check to see if the files are already installed.
if (INSTALL_FLAG === undefined) {
  let hasMissingFiles = false;
  for (const { filepath } of INSTALL_MANIFESTS) {
    if (!require('fs').existsSync(filepath)) {
      hasMissingFiles = true;
      break;
    }
  }
  if (!hasMissingFiles) {
    process.exit(0);
  }
}

void installPackages(PACKAGES, INSTALL_MANIFESTS, INSTALL_METADATA.feeds);

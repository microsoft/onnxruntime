// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const fs = require('node:fs');
const path = require('node:path');

const getNativeBindingPath = (
  packageRoot = path.join(__dirname, '..'),
  platform = process.platform,
  arch = process.arch,
) => {
  const relativeBinding = `bin/napi-v6/${platform}/${arch}/onnxruntime_binding.node`;
  const bundledBinding = path.join(packageRoot, relativeBinding);
  // Source builds and the existing all-platform package keep their current layout.
  if (fs.existsSync(bundledBinding)) {
    return bundledBinding;
  }

  const packageName = `onnxruntime-node-${platform}-${arch}`;
  let bindingPath;
  try {
    bindingPath = require.resolve(`${packageName}/${relativeBinding}`, { paths: [packageRoot] });
  } catch (error) {
    if (error.code !== 'MODULE_NOT_FOUND') {
      throw error;
    }
    throw new Error(`Missing ${packageName}. Install onnxruntime-node with optional dependencies enabled.`, {
      cause: error,
    });
  }

  const nativeManifest = JSON.parse(
    fs.readFileSync(require.resolve(`${packageName}/package.json`, { paths: [packageRoot] }), 'utf8'),
  );
  const parentManifest = JSON.parse(fs.readFileSync(path.join(packageRoot, 'package.json'), 'utf8'));
  if (nativeManifest.version !== parentManifest.version) {
    throw new Error(
      `Version mismatch: ${packageName}@${nativeManifest.version}, onnxruntime-node@${parentManifest.version}`,
    );
  }
  return bindingPath;
};

const getNativeDirectory = (
  packageRoot = path.join(__dirname, '..'),
  platform = process.platform,
  arch = process.arch,
) => {
  const manifest = JSON.parse(fs.readFileSync(path.join(packageRoot, 'package.json'), 'utf8'));
  if (!manifest.optionalDependencies?.[`onnxruntime-node-${platform}-${arch}`]) {
    // Explicit source-build downloads may run before the native binding has been compiled.
    return path.join(packageRoot, 'bin/napi-v6', platform, arch);
  }
  return path.dirname(getNativeBindingPath(packageRoot, platform, arch));
};

module.exports = { getNativeBindingPath, getNativeDirectory };

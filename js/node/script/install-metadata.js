// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

const metadataVersions = require('./install-metadata-versions.js');

const metadata = {
  // Requirements defines a list of manifest to install for a specific platform/architecture combination.
  requirements: {
    'win32/x64': [],
    'win32/arm64': [],
    'linux/x64': ['cuda12'],
    'linux/arm64': [],
    'darwin/x64': [],
    'darwin/arm64': [],
  },
  // Each manifest defines a list of files to install
  manifests: {
    'linux/x64:cuda12': {
      './libonnxruntime_providers_cuda.so': {
        package: 'nuget:linux/x64:cuda12',
        path: 'runtimes/win-x64/native/libonnxruntime_providers_cuda.so',
      },
      './libonnxruntime_providers_shared.so': {
        package: 'nuget:linux/x64:cuda12',
        path: 'runtimes/win-x64/native/libonnxruntime_providers_shared.so',
      },
      './libonnxruntime_providers_tensorrt.so': {
        package: 'nuget:linux/x64:cuda12',
        path: 'runtimes/win-x64/native/libonnxruntime_providers_tensorrt.so',
      },
    },
  },
  // Each package defines a list of package metadata. The first available package will be used.
  packages: {
    'nuget:win32/x64:cuda12': {
      name: 'Microsoft.ML.OnnxRuntime.Gpu.Windows',
      versions: metadataVersions.nuget,
    },
    'nuget:linux/x64:cuda12': {
      name: 'Microsoft.ML.OnnxRuntime.Gpu.Linux',
      versions: metadataVersions.nuget,
    },
  },
  feeds: {
    nuget: {
      type: 'nuget',
      index: 'https://api.nuget.org/v3/index.json',
    },
    nuget_nightly: {
      type: 'nuget',
      index: 'https://aiinfra.pkgs.visualstudio.com/PublicPackages/_packaging/ORT-Nightly/nuget/v3/index.json',
    },
  },
};

module.exports = metadata;

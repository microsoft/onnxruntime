// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as fs from 'fs-extra';
import * as path from 'path';
import * as process from 'process';

function updatePackageJson() {
  const commonPackageJsonPath = path.join(__dirname, '..', '..', 'common', 'package.json');
  const selfPackageJsonPath = path.join(__dirname, '..', 'package.json');
  console.log(`=== start to update package.json: ${selfPackageJsonPath}`);
  const packageCommon = fs.readJSONSync(commonPackageJsonPath);
  const packageSelf = fs.readJSONSync(selfPackageJsonPath);

  if (process.env.npm_config_ort_js_pack_mode === 'e2e' || process.env.ORT_JS_PACK_MODE === 'e2e') {
    // for E2E testing mode, we remove "onnxruntime-common" as a dependency.
    // we do this because yarn cannot resolve it with an unpublished version.
    delete packageSelf.dependencies['onnxruntime-common'];
  } else {
    const version = packageCommon.version;
    packageSelf.dependencies['onnxruntime-common'] = `~${version}`;
  }
  fs.writeJSONSync(selfPackageJsonPath, packageSelf, {spaces: 2});
  console.log('=== finished updating package.json.');
}

// update version of dependency "onnxruntime-common" before packing
updatePackageJson();

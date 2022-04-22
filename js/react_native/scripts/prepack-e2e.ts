// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import * as fs from 'fs-extra';
import * as path from 'path';

function updatePackageJson() {
  const selfPackageJsonPath = path.join(__dirname, '..', 'package.json');
  console.log(`=== start to update package.json: ${selfPackageJsonPath}`);
  const packageSelf = fs.readJSONSync(selfPackageJsonPath);
  delete packageSelf.dependencies['onnxruntime-common'];
  fs.writeJSONSync(selfPackageJsonPath, packageSelf, {spaces: 2});
  console.log('=== finished updating package.json.');
}

// update version of dependency "onnxruntime-common" before packing
updatePackageJson();

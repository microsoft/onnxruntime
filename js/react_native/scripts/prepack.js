// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

"use strict";

var fs = require("fs-extra");
var path = require("path");

function updatePackageJson() {
  var commonPackageJsonPath = path.join(__dirname, '..', '..', 'common', 'package.json');
  var selfPackageJsonPath = path.join(__dirname, '..', 'package.json');
  console.log("=== start to update package.json: " + selfPackageJsonPath);
  var packageCommon = fs.readJSONSync(commonPackageJsonPath);
  var packageSelf = fs.readJSONSync(selfPackageJsonPath);
  var version = packageCommon.version;
  packageSelf.dependencies['onnxruntime-common'] = "~" + version;
  fs.writeJSONSync(selfPackageJsonPath, packageSelf, { spaces: 2 });
  console.log('=== finished updating package.json.');
}

// update version of dependency "onnxruntime-common" before packing
updatePackageJson();

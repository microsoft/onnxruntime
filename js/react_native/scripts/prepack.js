"use strict";
// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
Object.defineProperty(exports, "__esModule", { value: true });
const fs = __importStar(require("fs-extra"));
const path = __importStar(require("path"));
const process = __importStar(require("process"));
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
    }
    else {
        const version = packageCommon.version;
        packageSelf.dependencies['onnxruntime-common'] = `${version}`;
    }
    fs.writeJSONSync(selfPackageJsonPath, packageSelf, { spaces: 2 });
    console.log('=== finished updating package.json.');
}
// update version of dependency "onnxruntime-common" before packing
updatePackageJson();

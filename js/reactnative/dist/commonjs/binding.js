"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});
exports.binding = void 0;

var _reactNative = require("react-native");

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
// export native binding
const {
  Onnxruntime
} = _reactNative.NativeModules;
const binding = Onnxruntime;
exports.binding = binding;
//# sourceMappingURL=binding.js.map
"use strict";

Object.defineProperty(exports, "__esModule", {
  value: true
});

var _onnxruntimeCommon = require("onnxruntime-common");

Object.keys(_onnxruntimeCommon).forEach(function (key) {
  if (key === "default" || key === "__esModule") return;
  if (key in exports && exports[key] === _onnxruntimeCommon[key]) return;
  Object.defineProperty(exports, key, {
    enumerable: true,
    get: function () {
      return _onnxruntimeCommon[key];
    }
  });
});

var _backend = require("./backend");

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
(0, _onnxruntimeCommon.registerBackend)('react-native', _backend.onnxruntimeBackend, 1);
//# sourceMappingURL=index.js.map
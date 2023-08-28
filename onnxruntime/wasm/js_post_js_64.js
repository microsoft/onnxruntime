// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

Module["PTR_SIZE"] = 8;
Module["createFileFromArrayBuffer"] = (path, buffer) => {
  const weightsFile = FS.create(path);
  weightsFile.contents = buffer;
  weightsFile.usedBytes = buffer.byteLength;
}

// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// init JSEP
Module["jsepInit"] = function (backend, alloc, free, upload, download, run) {
    Module.jsepBackend = backend;
    Module.jsepAlloc = alloc;
    Module.jsepFree = free;
    Module.jsepUpload = upload;
    Module.jsepDownload = download;
    Module.jsepRun = run;
};

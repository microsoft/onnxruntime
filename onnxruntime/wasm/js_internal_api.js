// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

'use strict';

// init JSEP
Module['jsepInit'] = (backend, alloc, free, copy, copyAsync, createKernel, releaseKernel, runKernel) => {
  Module.jsepBackend = backend;
  Module.jsepAlloc = alloc;
  Module.jsepFree = free;
  Module.jsepCopy = copy;
  Module.jsepCopyAsync = copyAsync;
  Module.jsepCreateKernel = createKernel;
  Module.jsepReleaseKernel = releaseKernel;
  Module.jsepRunKernel = runKernel;

  Module['jsepOnRunStart'] = sessionId => {
    Module['jsepRunPromise'] = new Promise(r => {
      Module.jsepRunPromiseResolve = r;
    });

    if (Module.jsepSessionState) {
      throw new Error('Session already started');
    }

    Module.jsepSessionState = {
      sessionId,
      errors: []
    };
  };

  Module['jsepOnRunEnd'] = sessionId => {
    if (Module.jsepSessionState.sessionId !== sessionId) {
      throw new Error('Session ID mismatch');
    }

    const errorPromises = Module.jsepSessionState.errors;
    Module.jsepSessionState = null;

    return errorPromises.length === 0 ? Promise.resolve() : new Promise((resolve, reject) => {
      Promise.all(errorPromises).then(errors => {
        errors = errors.filter(e => e);
        if (errors.length > 0) {
          reject(new Error(errors.join('\n')));
        } else {
          resolve();
        }
      }, reason => {
        reject(reason);
      });
    });
  };
};

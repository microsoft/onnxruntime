// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Backend} from './backend';

interface BackendInfo {
  backend: Backend;
  priority: number;

  initializing?: boolean;
  initialized?: boolean;
  aborted?: boolean;
}

const backends: {[name: string]: BackendInfo} = {};
const backendsSortedByPriority: string[] = [];

/**
 * Register a backend.
 *
 * @param name - the name as a key to lookup as an execution provider.
 * @param backend - the backend object.
 * @param priority - an integer indicating the priority of the backend. Higher number means higher priority.
 */
export const registerBackend = (name: string, backend: Backend, priority: number): void => {
  if (backend && typeof backend.init === 'function' && typeof backend.createSessionHandler === 'function') {
    const currentBackend = backends[name];
    if (currentBackend === undefined) {
      backends[name] = {backend, priority};
    } else if (currentBackend.backend === backend) {
      return;
    } else {
      throw new Error(`backend "${name}" is already registered`);
    }

    for (let i = 0; i < backendsSortedByPriority.length; i++) {
      if (backends[backendsSortedByPriority[i]].priority <= priority) {
        backendsSortedByPriority.splice(i, 0, name);
        return;
      }
    }
    backendsSortedByPriority.push(name);
    return;
  }

  throw new TypeError('not a valid backend');
};

/**
 * Resolve backend by specified hints.
 *
 * @param backendHints - a list of execution provider names to lookup. If omitted use registered backends as list.
 * @returns a promise that resolves to the backend.
 */
export const resolveBackend = async(backendHints: readonly string[]): Promise<Backend> => {
  const backendNames = backendHints.length === 0 ? backendsSortedByPriority : backendHints;
  const errors = [];
  for (const backendName of backendNames) {
    const backendInfo = backends[backendName];
    if (backendInfo) {
      if (backendInfo.initialized) {
        return backendInfo.backend;
      } else if (backendInfo.initializing) {
        throw new Error(`backend "${backendName}" is being initialized; cannot initialize multiple times.`);
      } else if (backendInfo.aborted) {
        continue;  // current backend is unavailable; try next
      }

      try {
        backendInfo.initializing = true;
        await backendInfo.backend.init();
        backendInfo.initialized = true;
        return backendInfo.backend;
      } catch (e) {
        errors.push({name: backendName, err: e});
        backendInfo.aborted = true;
      } finally {
        backendInfo.initializing = false;
      }
    }
  }

  throw new Error(`no available backend found. ERR: ${errors.join(', ')}`);
};

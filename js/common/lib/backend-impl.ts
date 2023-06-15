// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {Backend} from './backend.js';

interface BackendInfo {
  backend: Backend;
  priority: number;

  initPromise?: Promise<void>;
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
 * @param priority - an integer indicating the priority of the backend. Higher number means higher priority. if priority
 * < 0, it will be considered as a 'beta' version and will not be used as a fallback backend by default.
 *
 * @internal
 */
export const registerBackend = (name: string, backend: Backend, priority: number): void => {
  if (backend && typeof backend.init === 'function' && typeof backend.createSessionHandler === 'function') {
    const currentBackend = backends[name];
    if (currentBackend === undefined) {
      backends[name] = {backend, priority};
    } else if (currentBackend.priority > priority) {
      // same name is already registered with a higher priority. skip registeration.
      return;
    } else if (currentBackend.priority === priority) {
      if (currentBackend.backend !== backend) {
        throw new Error(`cannot register backend "${name}" using priority ${priority}`);
      }
    }

    if (priority >= 0) {
      const i = backendsSortedByPriority.indexOf(name);
      if (i !== -1) {
        backendsSortedByPriority.splice(i, 1);
      }

      for (let i = 0; i < backendsSortedByPriority.length; i++) {
        if (backends[backendsSortedByPriority[i]].priority <= priority) {
          backendsSortedByPriority.splice(i, 0, name);
          return;
        }
      }
      backendsSortedByPriority.push(name);
    }
    return;
  }

  throw new TypeError('not a valid backend');
};

/**
 * Resolve backend by specified hints.
 *
 * @param backendHints - a list of execution provider names to lookup. If omitted use registered backends as list.
 * @returns a promise that resolves to the backend.
 *
 * @internal
 */
export const resolveBackend = async(backendHints: readonly string[]): Promise<Backend> => {
  const backendNames = backendHints.length === 0 ? backendsSortedByPriority : backendHints;
  const errors = [];
  for (const backendName of backendNames) {
    const backendInfo = backends[backendName];
    if (backendInfo) {
      if (backendInfo.initialized) {
        return backendInfo.backend;
      } else if (backendInfo.aborted) {
        continue;  // current backend is unavailable; try next
      }

      const isInitializing = !!backendInfo.initPromise;
      try {
        if (!isInitializing) {
          backendInfo.initPromise = backendInfo.backend.init();
        }
        await backendInfo.initPromise;
        backendInfo.initialized = true;
        return backendInfo.backend;
      } catch (e) {
        if (!isInitializing) {
          errors.push({name: backendName, err: e});
        }
        backendInfo.aborted = true;
      } finally {
        delete backendInfo.initPromise;
      }
    }
  }

  throw new Error(`no available backend found. ERR: ${errors.map(e => `[${e.name}] ${e.err}`).join(', ')}`);
};

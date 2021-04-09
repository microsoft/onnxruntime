import {Backend} from './backend';

interface BackendInfo {
  backend: Backend;
  priority: number;
}

const backends: {[name: string]: BackendInfo} = {};
const backendsSortedByPriority: string[] = [];

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

export const resolveBackend = async(backendHints: readonly string[]): Promise<Backend> => {
  if (backendHints.length === 0) {
    backendHints = backendsSortedByPriority;
  }

  for (const backendName of backendHints) {
    const backendInfo = backends[backendName];
    if (backendInfo) {
      await backendInfo.backend.init();
      return backendInfo.backend;
    }
  }

  throw new Error('no available backend found.');
};

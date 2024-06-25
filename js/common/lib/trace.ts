// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env} from './env-impl.js';

/**
 * @ignore
 */
export const TRACE = (deviceType: string, label: string) => {
  if (typeof env.trace === 'undefined' ? !env.wasm.trace : !env.trace) {
    return;
  }
  // eslint-disable-next-line no-console
  console.timeStamp(`${deviceType}::ORT::${label}`);
};

/**
 * @ignore
 */
export const traceFunc = (originalMethod: any, context: ClassMethodDecoratorContext) => {
  if (typeof env.trace === 'undefined' ? !env.wasm.trace : !env.trace) {
    return;
  }

  const methodName = String(context.name);
  return function(this: any, ...args: any[]) {
    TRACE('CPU', `FUNC_BEGIN::${methodName}`);
    const result = originalMethod.apply(this, ...args);
    TRACE('CPU', `FUNC_END::${methodName}`);
    return result;
  };
}

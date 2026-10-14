// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { expect } from 'chai';
import { env } from 'onnxruntime-common';

import { onnxjsBackend } from '../../lib/backend-onnxjs';
import { createDeprecationWarning } from '../../lib/deprecation-warning';

describe('#UnitTest# - deprecation warning', () => {
  const originalLogLevel = env.logLevel;
  const originalConsoleWarn = console.warn;
  let warnings: unknown[][];

  beforeEach(() => {
    warnings = [];
    console.warn = (...data: unknown[]) => {
      warnings.push(data);
    };
  });

  afterEach(() => {
    env.logLevel = originalLogLevel;
    console.warn = originalConsoleWarn;
  });

  for (const logLevel of ['verbose', 'info', 'warning'] as const) {
    it(`warns once when log level is ${logLevel}`, () => {
      env.logLevel = logLevel;
      const warn = createDeprecationWarning('deprecated');

      warn();
      warn();

      expect(warnings).to.deep.equal([['deprecated']]);
    });
  }

  for (const logLevel of ['error', 'fatal'] as const) {
    it(`does not warn when log level is ${logLevel}`, () => {
      env.logLevel = logLevel;

      createDeprecationWarning('deprecated')();

      expect(warnings).to.have.lengthOf(0);
    });
  }

  it('warns once when the WebGL backend is initialized', async () => {
    env.logLevel = 'warning';

    await onnxjsBackend.init();
    await onnxjsBackend.init();

    expect(warnings).to.have.lengthOf(1);
    expect(warnings[0][0]).to.contain('https://github.com/microsoft/onnxruntime/issues/32241');
  });
});

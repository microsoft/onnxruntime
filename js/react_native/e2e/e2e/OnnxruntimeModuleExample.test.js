// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* Initial Simple Detox Test Setup. Can potentially add more unit tests. */

describe('OnnxruntimeModuleExample', () => {
  beforeAll(async () => {
    await device.launchApp();
  });

  beforeEach(async () => {
    await device.launchApp({ newInstance: true });
  });

  it('OnnxruntimeModuleExampleE2ETest OutputComponentExists', async () => {
    await element(by.id('output'));
  });

  it('OnnxruntimeModuleExampleE2ETest InferenceResultValueIsCorrect', async () => {
    await expect(element(by.id('output'))).toHaveText('Result: 3');
  });
});
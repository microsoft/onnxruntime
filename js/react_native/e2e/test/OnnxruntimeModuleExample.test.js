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

  it('OnnxruntimeModuleExampleE2ETest CheckOutputComponentExists', async () => {
    await element(by.label('output'));
  });

  it('OnnxruntimeModuleExampleE2ETest CheckInferenceResultValueIsCorrect', async () => {
    if (device.getPlatform() === 'ios') {
      await expect(element(by.label('output')).atIndex(1)).toHaveText('Result: 3');
    }
    if (device.getPlatform() === 'android') {
      await expect(element(by.label('output'))).toHaveText('Result: 3');
    }
  });
});
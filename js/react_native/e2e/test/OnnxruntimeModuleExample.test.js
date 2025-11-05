// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* Initial Simple Detox Test Setup. Can potentially add more unit tests. */

const delay = (ms) => new Promise(resolve => setTimeout(resolve, ms));

describe('OnnxruntimeModuleExample', () => {
  beforeAll(async () => {
    await device.launchApp();
  });

  beforeEach(async () => {
    await device.reloadReactNative();
  });

  it('MNIST test inference result should be correct', async () => {
    // Tap MNIST test button
    await element(by.text('MNIST Test')).tap();

    await delay(500);

    // Check the inference result
    if (device.getPlatform() === 'ios') {
      await expect(element(by.label('output')).atIndex(1)).toHaveText('Result: 3');
    }
    if (device.getPlatform() === 'android') {
      await expect(element(by.label('output'))).toHaveText('Result: 3');
    }
  });

  it('Basic Types test should run successfully', async () => {
    // Tap Basic Types test button
    await element(by.text('Basic Types Test')).tap();

    await delay(500);

    // Run the tests
    await element(by.text('Run Tests')).tap();
    await delay(500);

    // Check have no error
    await expect(element(by.label('statusError'))).not.toBeVisible();
  });
});

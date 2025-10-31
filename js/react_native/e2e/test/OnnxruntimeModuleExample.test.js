// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/* Initial Simple Detox Test Setup. Can potentially add more unit tests. */

describe('OnnxruntimeModuleExample', () => {
  beforeAll(async () => {
    await device.launchApp();
  });

  beforeEach(async () => {
    await device.reloadReactNative();
  });

  it('MNIST test inference result should be correct', async () => {
    // Tap MNIST test button
    await element(by.label('mnist-test-button')).tap();

    // Wait for the back button to appear (indicating navigation completed)
    await waitFor(element(by.label('back-button')))
      .toBeVisible()
      .withTimeout(2000);

    // Wait for inference to complete and output to appear
    await waitFor(element(by.label('output')))
      .toBeVisible()
      .withTimeout(10000);

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
    await element(by.label('basic-types-test-button')).tap();

    // Wait for the back button to appear (indicating navigation completed)
    await waitFor(element(by.label('back-button')))
      .toBeVisible()
      .withTimeout(2000);

    // Check that Basic Types Test page is visible
    await expect(element(by.text('Basic Types Test'))).toBeVisible();
    await expect(element(by.label('run-tests-button'))).toBeVisible();

    // Run the tests
    await element(by.label('run-tests-button')).tap();

    // Wait for tests to complete
    await waitFor(element(by.label('statusPending')))
      .not.toBeVisible()
      .withTimeout(15000);

    // Check have no error
    await expect(element(by.label('statusError'))).not.toBeVisible();
  });
});

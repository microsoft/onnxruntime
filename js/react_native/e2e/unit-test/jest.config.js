// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

module.exports = {
  rootDir: '..',
  testMatch: ['<rootDir>/unit-test/**/*.test.js'],
  preset: 'react-native',
  transform: {
    '^.+\\.[jt]sx?$': ['babel-jest', { presets: ['module:@react-native/babel-preset'] }],
  },
};

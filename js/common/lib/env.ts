// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

interface Env {
  /**
   * Indicate whether run in debug mode.
   */
  debug?: boolean;

  [name: string]: unknown;
}

/**
 * Represent a set of flags as a global singleton.
 */
export const env: Env = {};

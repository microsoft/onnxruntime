// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import {env} from 'onnxruntime-common';

import {logLevelStringToEnum} from '../wasm-common';

type LogLevel = NonNullable<typeof env.logLevel>;
type MessageString = string;
type MessageFunction = () => string;
type Message = MessageString|MessageFunction;

const logLevelPrefix = ['V', 'I', 'W', 'E', 'F'];

const doLog = (level: number, message: string): void => {
  // eslint-disable-next-line no-console
  console.log(`[${logLevelPrefix[level]},${new Date().toISOString()}]${message}`);
};

/**
 * A simple logging utility to log messages to the console.
 */
export const LOG = (logLevel: LogLevel, msg: Message): void => {
  const messageLevel = logLevelStringToEnum(logLevel);
  const configLevel = logLevelStringToEnum(env.logLevel!);
  if (messageLevel >= configLevel) {
    doLog(messageLevel, typeof msg === 'function' ? msg() : msg);
  }
};

/**
 * A simple logging utility to log messages to the console. Only logs when debug is enabled.
 */
export const LOG_DEBUG: typeof LOG = (...args: Parameters<typeof LOG>) => {
  if (env.debug) {
    LOG(...args);
  }
};

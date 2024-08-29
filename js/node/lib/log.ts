// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

import { Env } from 'onnxruntime-common';

/**
 * Map string log level to integer value
 */
export const logLevelStringToEnum = (logLevel?: 'verbose' | 'info' | 'warning' | 'error' | 'fatal'): number => {
  switch (logLevel) {
    case 'verbose':
      return 0;
    case 'info':
      return 1;
    case 'warning':
      return 2;
    case 'error':
      return 3;
    case 'fatal':
      return 4;
    default:
      throw new Error(`unsupported logging level: ${logLevel}`);
  }
};

type LogLevel = NonNullable<Env['logLevel']>;
type MessageString = string;
type MessageFunction = () => string;
type Message = MessageString | MessageFunction;

const logLevelPrefix = ['V', 'I', 'W', 'E', 'F'];

const doLog = (level: number, message: string): void => {
  // eslint-disable-next-line no-console
  console.log(`[${logLevelPrefix[level]},${new Date().toISOString()}]${message}`);
};

let configLogLevel: LogLevel | undefined;
let debug: boolean | undefined;

export const configureLogger = ($configLogLevel: LogLevel, $debug: boolean): void => {
  configLogLevel = $configLogLevel;
  debug = $debug;
};

/**
 * A simple logging utility to log messages to the console.
 */
export const LOG = (logLevel: LogLevel, msg: Message): void => {
  const messageLevel = logLevelStringToEnum(logLevel);
  const configLevel = logLevelStringToEnum(configLogLevel);
  if (messageLevel >= configLevel) {
    doLog(messageLevel, typeof msg === 'function' ? msg() : msg);
  }
};

/**
 * A simple logging utility to log messages to the console. Only logs when debug is enabled.
 */
export const LOG_DEBUG: typeof LOG = (...args: Parameters<typeof LOG>) => {
  if (debug) {
    LOG(...args);
  }
};

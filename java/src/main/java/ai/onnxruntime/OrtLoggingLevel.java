/*
 * Copyright (c) 2019, 2020, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.util.logging.Logger;

/** The logging level for messages from the environment and session. */
public enum OrtLoggingLevel {
  ORT_LOGGING_LEVEL_VERBOSE(0),
  ORT_LOGGING_LEVEL_INFO(1),
  ORT_LOGGING_LEVEL_WARNING(2),
  ORT_LOGGING_LEVEL_ERROR(3),
  ORT_LOGGING_LEVEL_FATAL(4);
  private final int value;

  private static final Logger logger = Logger.getLogger(OrtLoggingLevel.class.getName());
  private static final OrtLoggingLevel[] values = new OrtLoggingLevel[5];

  static {
    for (OrtLoggingLevel ot : OrtLoggingLevel.values()) {
      values[ot.value] = ot;
    }
  }

  OrtLoggingLevel(int value) {
    this.value = value;
  }

  /**
   * Gets the native value associated with this logging level.
   *
   * @return The native value.
   */
  public int getValue() {
    return value;
  }

  /**
   * Maps from the C API's int enum to the Java enum.
   *
   * @param logLevel The index of the Java enum.
   * @return The Java enum.
   */
  public static OrtLoggingLevel mapFromInt(int logLevel) {
    if ((logLevel > 0) && (logLevel < values.length)) {
      return values[logLevel];
    } else {
      logger.warning("Unknown logging level " + logLevel + " setting to ORT_LOGGING_LEVEL_VERBOSE");
      return ORT_LOGGING_LEVEL_VERBOSE;
    }
  }
}

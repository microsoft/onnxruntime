/*
 * Copyright (c) 2021, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers;

import java.util.EnumSet;

/** An interface for bitset enums that should be aggregated into a single integer. */
public interface OrtFlags {

  /**
   * Gets the underlying flag value.
   *
   * @return The flag value.
   */
  public int getValue();

  /**
   * Converts an EnumSet of flags into the value expected by the C API.
   *
   * @param set The enum set to aggregate the values from.
   * @param <E> The enum type to aggregate.
   * @return The aggregated values
   */
  public static <E extends Enum<E> & OrtFlags> int aggregateToInt(EnumSet<E> set) {
    int value = 0;

    for (OrtFlags flag : set) {
      value |= flag.getValue();
    }

    return value;
  }
}

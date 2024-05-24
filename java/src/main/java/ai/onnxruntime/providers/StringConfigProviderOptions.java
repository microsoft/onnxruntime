/*
 * Copyright (c) 2022, 2024, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime.providers;

import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtProviderOptions;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.Objects;
import java.util.stream.Collectors;

/**
 * Abstract base class for provider options which are configured solely by key value string pairs.
 */
abstract class StringConfigProviderOptions extends OrtProviderOptions {
  /** A Java side copy of the options. */
  protected final Map<String, String> options;

  protected StringConfigProviderOptions(long nativeHandle) {
    super(nativeHandle);
    // LinkedHashMap to ensure iteration order matches insertion order for the native object.
    this.options = new LinkedHashMap<>();
  }

  /**
   * Adds a configuration option to this options.
   *
   * @param key The key.
   * @param value The value.
   * @throws OrtException If the addition failed.
   */
  public void add(String key, String value) throws OrtException {
    checkClosed();
    Objects.requireNonNull(key, "Key must not be null");
    Objects.requireNonNull(value, "Value must not be null");
    options.put(key, value);
  }

  /**
   * Parses the output of {@code getOptionsString()} and adds those options to this options
   * instance.
   *
   * @param serializedForm The serialized form to parse.
   * @throws OrtException If the option could not be added.
   */
  public void parseOptionsString(String serializedForm) throws OrtException {
    String[] options = serializedForm.split(";");
    for (String o : options) {
      if (o.contains("=")) {
        String[] curOption = o.split("=");
        if ((curOption.length == 2) && !curOption[0].isEmpty() && !curOption[1].isEmpty()) {
          add(curOption[0], curOption[1]);
        } else {
          throw new IllegalArgumentException("Failed to parse option from string '" + o + "'");
        }
      }
    }
  }

  @Override
  public String toString() {
    return this.getClass().getSimpleName() + "(" + getOptionsString() + ")";
  }

  /**
   * Returns the serialized options string
   *
   * @return The serialized options string.
   */
  public String getOptionsString() {
    return options.entrySet().stream()
        .map(e -> e.getKey() + "=" + e.getValue())
        .collect(Collectors.joining(";", "", ";"));
  }

  @Override
  protected void applyToNative() throws OrtException {
    if (!options.isEmpty()) {
      String[] keys = new String[options.size()];
      String[] values = new String[options.size()];
      int i = 0;
      for (Map.Entry<String, String> e : options.entrySet()) {
        keys[i] = e.getKey();
        values[i] = e.getValue();
        i++;
      }

      applyToNative(getApiHandle(), this.nativeHandle, keys, values);
    }
  }

  /**
   * Add all the options to this options instance.
   *
   * @param apiHandle The api pointer.
   * @param nativeHandle The native options pointer.
   * @param key The option keys.
   * @param value The option values.
   * @throws OrtException If the addition failed.
   */
  protected abstract void applyToNative(
      long apiHandle, long nativeHandle, String[] key, String[] value) throws OrtException;
}

/*
 * Copyright (c) 2019, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.util.Map;

/**
 * Top interface for input and output values from ONNX models. Currently implemented by {@link
 * OnnxTensor}, {@link OnnxSequence} and {@link OnnxMap}. Will be sealed to these types one day.
 *
 * <p>Does not support sparse tensors.
 */
public interface OnnxValue extends AutoCloseable {

  /** The type of the {@link OnnxValue}, mirroring the id in the C API. */
  public enum OnnxValueType {
    ONNX_TYPE_UNKNOWN(0),
    ONNX_TYPE_TENSOR(1),
    ONNX_TYPE_SEQUENCE(2),
    ONNX_TYPE_MAP(3),
    ONNX_TYPE_OPAQUE(4),
    ONNX_TYPE_SPARSETENSOR(5);

    /** The id number of this type in the C API. */
    public final int value;

    OnnxValueType(int value) {
      this.value = value;
    }
  }

  /**
   * Gets the type of this OnnxValue.
   *
   * @return The value type.
   */
  public OnnxValueType getType();

  /**
   * Returns the value as a Java object copying it out of the native heap. This operation can be
   * quite slow for high dimensional tensors, where you should prefer {@link
   * OnnxTensor#getByteBuffer()} etc.
   *
   * <p>Overridden by the subclasses with a more specific type if available.
   *
   * @return The value.
   * @throws OrtException If an error occurred reading the value.
   */
  public Object getValue() throws OrtException;

  /**
   * Gets the type info object associated with this OnnxValue.
   *
   * @return The type information.
   */
  public ValueInfo getInfo();

  /** Closes the OnnxValue, freeing it's native memory. */
  @Override
  public void close();

  /**
   * Calls close on each element of the iterable.
   *
   * @param itr An iterable of closeable OnnxValues.
   */
  public static void close(Iterable<? extends OnnxValue> itr) {
    for (OnnxValue t : itr) {
      t.close();
    }
  }

  /**
   * Calls close on each {@link OnnxValue} in the map.
   *
   * @param map A map of {@link OnnxValue}s.
   */
  public static void close(Map<String, ? extends OnnxValue> map) {
    for (OnnxValue t : map.values()) {
      t.close();
    }
  }
}

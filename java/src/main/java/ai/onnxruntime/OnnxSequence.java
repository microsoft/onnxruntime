/*
 * Copyright (c) 2019, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import java.io.IOException;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.logging.Logger;

/**
 * A sequence of {@link OnnxValue}s all of the same type.
 *
 * <p>Supports the types mentioned in "onnxruntime_c_api.h", currently
 *
 * <ul>
 *   <li>OnnxTensor&lt;String&gt;
 *   <li>OnnxTensor&lt;Long&gt;
 *   <li>OnnxTensor&lt;Float&gt;
 *   <li>OnnxTensor&lt;Double&gt;
 *   <li>OnnxMap&lt;String,Float&gt;
 *   <li>OnnxMap&lt;Long,Float&gt;
 * </ul>
 */
public class OnnxSequence implements OnnxValue {
  private static final Logger logger = Logger.getLogger(OnnxSequence.class.getName());

  static {
    try {
      OnnxRuntime.init();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load onnx-runtime library", e);
    }
  }

  /** The native pointer. */
  final long nativeHandle;

  private final long allocatorHandle;

  private final SequenceInfo info;

  private boolean closed;

  /**
   * Creates the wrapper object for a native sequence.
   *
   * <p>Called from native code.
   *
   * @param nativeHandle The reference to the native sequence object.
   * @param allocatorHandle The reference to the allocator.
   * @param info The sequence type information.
   */
  OnnxSequence(long nativeHandle, long allocatorHandle, SequenceInfo info) {
    this.nativeHandle = nativeHandle;
    this.allocatorHandle = allocatorHandle;
    this.info = info;
    this.closed = false;
  }

  @Override
  public OnnxValueType getType() {
    return OnnxValueType.ONNX_TYPE_SEQUENCE;
  }

  /**
   * Extracts a Java list of the {@link OnnxValue}s which can then be further unwrapped.
   *
   * <p>Returns either a {@link List} of either {@link OnnxTensor} or {@link OnnxMap}.
   *
   * <p>Note unlike the other {@link OnnxValue#getValue()} methods, this does not copy the values
   * themselves into the Java heap, it merely exposes them as {@link OnnxValue} instances, allowing
   * users to use the faster copy methods available for {@link OnnxTensor}. This also means that
   * those values need to be closed separately from this instance, and are not closed by {@link
   * #close} on this object.
   *
   * @return A Java list containing the values.
   * @throws OrtException If the runtime failed to read an element.
   */
  @Override
  public List<? extends OnnxValue> getValue() throws OrtException {
    checkClosed();
    if (info.sequenceOfMaps) {
      OnnxMap[] maps = getMaps(OnnxRuntime.ortApiHandle, nativeHandle, allocatorHandle);
      return Collections.unmodifiableList(Arrays.asList(maps));
    } else {
      switch (info.sequenceType) {
        case STRING:
        case INT64:
        case FLOAT:
        case DOUBLE:
          OnnxTensor[] tensors =
              getTensors(OnnxRuntime.ortApiHandle, nativeHandle, allocatorHandle);
          return Collections.unmodifiableList(Arrays.asList(tensors));
        case BOOL:
        case UINT8:
        case INT8:
        case INT16:
        case INT32:
        case UNKNOWN:
        default:
          throw new OrtException("Unsupported type in a sequence, found " + info.sequenceType);
      }
    }
  }

  @Override
  public SequenceInfo getInfo() {
    return info;
  }

  @Override
  public String toString() {
    return "OnnxSequence(info=" + info.toString() + ")";
  }

  @Override
  public synchronized boolean isClosed() {
    return closed;
  }

  /** Closes this sequence, releasing the native memory backing it and it's elements. */
  @Override
  public synchronized void close() {
    if (!closed) {
      close(OnnxRuntime.ortApiHandle, nativeHandle);
      closed = true;
    } else {
      logger.warning("Closing an already closed sequence.");
    }
  }

  /** Checks if the OnnxValue is closed, if so throws {@link IllegalStateException}. */
  protected void checkClosed() {
    if (closed) {
      throw new IllegalStateException("Trying to use a closed OnnxValue");
    }
  }

  private native OnnxMap[] getMaps(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native OnnxTensor[] getTensors(long apiHandle, long nativeHandle, long allocatorHandle)
      throws OrtException;

  private native void close(long apiHandle, long nativeHandle);
}

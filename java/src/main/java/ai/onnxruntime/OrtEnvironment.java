/*
 * Copyright (c) 2019, 2020 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtSession.SessionOptions;
import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

/**
 * The host object for the onnx-runtime system. Can create {@link OrtSession}s which encapsulate
 * specific models.
 */
public class OrtEnvironment implements AutoCloseable {

  private static final Logger logger = Logger.getLogger(OrtEnvironment.class.getName());

  public static final String DEFAULT_NAME = "ort-java";

  static {
    try {
      OnnxRuntime.init();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load onnx-runtime library", e);
    }
  }

  private static volatile OrtEnvironment INSTANCE;

  private static final AtomicInteger refCount = new AtomicInteger();

  private static volatile OrtLoggingLevel curLogLevel;

  private static volatile String curLoggingName;

  /**
   * Gets the OrtEnvironment. If there is not an environment currently created, it creates one using
   * {@link OrtEnvironment#DEFAULT_NAME} and {@link OrtLoggingLevel#ORT_LOGGING_LEVEL_WARNING}.
   *
   * @return An onnxruntime environment.
   */
  public static OrtEnvironment getEnvironment() {
    return getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, DEFAULT_NAME);
  }

  /**
   * Gets the OrtEnvironment. If there is not an environment currently created, it creates one using
   * the supplied name and {@link OrtLoggingLevel#ORT_LOGGING_LEVEL_WARNING}.
   *
   * @param name The logging id of the environment.
   * @return An onnxruntime environment.
   */
  public static OrtEnvironment getEnvironment(String name) {
    return getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, name);
  }

  /**
   * Gets the OrtEnvironment. If there is not an environment currently created, it creates one using
   * the {@link OrtEnvironment#DEFAULT_NAME} and the supplied logging level.
   *
   * @param logLevel The logging level to use.
   * @return An onnxruntime environment.
   */
  public static OrtEnvironment getEnvironment(OrtLoggingLevel logLevel) {
    return getEnvironment(logLevel, DEFAULT_NAME);
  }

  /**
   * Gets the OrtEnvironment. If there is not an environment currently created, it creates one using
   * the supplied name and logging level. If an environment already exists with a different name,
   * that environment is returned and a warning is logged.
   *
   * @param loggingLevel The logging level to use.
   * @param name The log id.
   * @return The OrtEnvironment singleton.
   */
  public static synchronized OrtEnvironment getEnvironment(
      OrtLoggingLevel loggingLevel, String name) {
    if (INSTANCE == null) {
      try {
        INSTANCE = new OrtEnvironment(loggingLevel, name);
        curLogLevel = loggingLevel;
        curLoggingName = name;
      } catch (OrtException e) {
        throw new IllegalStateException("Failed to create OrtEnvironment", e);
      }
    } else {
      if ((loggingLevel.getValue() != curLogLevel.getValue()) || (!name.equals(curLoggingName))) {
        logger.warning(
            "Tried to change OrtEnvironment's logging level or name while a reference exists.");
      }
    }
    refCount.incrementAndGet();
    return INSTANCE;
  }

  final long nativeHandle;

  final OrtAllocator defaultAllocator;

  private volatile boolean closed = false;

  /**
   * Create an OrtEnvironment using a default name.
   *
   * @throws OrtException If the environment couldn't be created.
   */
  private OrtEnvironment() throws OrtException {
    this(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, "java-default");
  }

  /**
   * Create an OrtEnvironment using the specified name and log level.
   *
   * @param loggingLevel The logging level to use.
   * @param name The logging id of the environment.
   * @throws OrtException If the environment couldn't be created.
   */
  private OrtEnvironment(OrtLoggingLevel loggingLevel, String name) throws OrtException {
    nativeHandle = createHandle(OnnxRuntime.ortApiHandle, loggingLevel.getValue(), name);
    defaultAllocator = new OrtAllocator(getDefaultAllocator(OnnxRuntime.ortApiHandle), true);
  }

  /**
   * Create a session using the default {@link SessionOptions}, model and the default memory
   * allocator.
   *
   * @param modelPath Path on disk to load the model from.
   * @return An {@link OrtSession} with the specified model.
   * @throws OrtException If the model failed to load, wasn't compatible or caused an error.
   */
  public OrtSession createSession(String modelPath) throws OrtException {
    return createSession(modelPath, new OrtSession.SessionOptions());
  }

  /**
   * Create a session using the specified {@link SessionOptions}, model and the default memory
   * allocator.
   *
   * @param modelPath Path on disk to load the model from.
   * @param options The session options.
   * @return An {@link OrtSession} with the specified model.
   * @throws OrtException If the model failed to load, wasn't compatible or caused an error.
   */
  public OrtSession createSession(String modelPath, SessionOptions options) throws OrtException {
    return createSession(modelPath, defaultAllocator, options);
  }

  /**
   * Create a session using the specified {@link SessionOptions} and model.
   *
   * @param modelPath Path on disk to load the model from.
   * @param allocator The memory allocator to use.
   * @param options The session options.
   * @return An {@link OrtSession} with the specified model.
   * @throws OrtException If the model failed to load, wasn't compatible or caused an error.
   */
  OrtSession createSession(String modelPath, OrtAllocator allocator, SessionOptions options)
      throws OrtException {
    if (!closed) {
      return new OrtSession(this, modelPath, allocator, options);
    } else {
      throw new IllegalStateException("Trying to create an OrtSession on a closed OrtEnvironment.");
    }
  }

  /**
   * Create a session using the specified {@link SessionOptions}, model and the default memory
   * allocator.
   *
   * @param modelArray Byte array representing an ONNX model.
   * @param options The session options.
   * @return An {@link OrtSession} with the specified model.
   * @throws OrtException If the model failed to parse, wasn't compatible or caused an error.
   */
  public OrtSession createSession(byte[] modelArray, SessionOptions options) throws OrtException {
    return createSession(modelArray, defaultAllocator, options);
  }

  /**
   * Create a session using the default {@link SessionOptions}, model and the default memory
   * allocator.
   *
   * @param modelArray Byte array representing an ONNX model.
   * @return An {@link OrtSession} with the specified model.
   * @throws OrtException If the model failed to parse, wasn't compatible or caused an error.
   */
  public OrtSession createSession(byte[] modelArray) throws OrtException {
    return createSession(modelArray, new OrtSession.SessionOptions());
  }

  /**
   * Create a session using the specified {@link SessionOptions} and model.
   *
   * @param modelArray Byte array representing an ONNX model.
   * @param allocator The memory allocator to use.
   * @param options The session options.
   * @return An {@link OrtSession} with the specified model.
   * @throws OrtException If the model failed to parse, wasn't compatible or caused an error.
   */
  OrtSession createSession(byte[] modelArray, OrtAllocator allocator, SessionOptions options)
      throws OrtException {
    if (!closed) {
      return new OrtSession(this, modelArray, allocator, options);
    } else {
      throw new IllegalStateException("Trying to create an OrtSession on a closed OrtEnvironment.");
    }
  }

  /**
   * Turns on or off the telemetry.
   *
   * @param sendTelemetry If true then send telemetry on onnxruntime usage.
   * @throws OrtException If the call failed.
   */
  public void setTelemetry(boolean sendTelemetry) throws OrtException {
    setTelemetry(OnnxRuntime.ortApiHandle, nativeHandle, sendTelemetry);
  }

  /**
   * Is this environment closed?
   *
   * @return True if the environment is closed.
   */
  public boolean isClosed() {
    return closed;
  }

  @Override
  public String toString() {
    return "OrtEnvironment(name=" + curLoggingName + ",logLevel=" + curLogLevel + ")";
  }

  /**
   * Closes the OrtEnvironment. If this is the last reference to the environment then it closes the
   * native handle.
   *
   * @throws OrtException If the close failed.
   */
  @Override
  public synchronized void close() throws OrtException {
    synchronized (refCount) {
      int curCount = refCount.get();
      if (curCount != 0) {
        refCount.decrementAndGet();
      }
      if (curCount == 1) {
        close(OnnxRuntime.ortApiHandle, nativeHandle);
        closed = true;
        INSTANCE = null;
      }
    }
  }

  /**
   * Creates the native object.
   *
   * @param apiHandle The API pointer.
   * @param loggingLevel The logging level.
   * @param name The name of the environment.
   * @return The pointer to the native object.
   * @throws OrtException If the creation failed.
   */
  private static native long createHandle(long apiHandle, int loggingLevel, String name)
      throws OrtException;

  /**
   * Gets a reference to the default allocator.
   *
   * @param apiHandle The API handle to use.
   * @return A pointer to the default allocator.
   * @throws OrtException If it failed to get the allocator.
   */
  private static native long getDefaultAllocator(long apiHandle) throws OrtException;

  /**
   * Closes the OrtEnvironment, frees the handle.
   *
   * @param apiHandle The API pointer.
   * @param nativeHandle The handle to free.
   * @throws OrtException If an error was caused by freeing the handle.
   */
  private static native void close(long apiHandle, long nativeHandle) throws OrtException;

  /**
   * Enables or disables the telemetry.
   *
   * @param apiHandle The API pointer.
   * @param nativeHandle The native handle for the environment.
   * @param sendTelemetry Turn on or off the telemetry.
   * @throws OrtException If an error was caused when setting the telemetry status.
   */
  private static native void setTelemetry(long apiHandle, long nativeHandle, boolean sendTelemetry)
      throws OrtException;
}

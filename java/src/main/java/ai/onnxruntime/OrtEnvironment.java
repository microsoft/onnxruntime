/*
 * Copyright (c) 2019-2021 Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

import ai.onnxruntime.OrtSession.SessionOptions;
import java.io.IOException;
import java.util.EnumSet;
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
   * @return The OrtEnvironment singleton.
   */
  public static synchronized OrtEnvironment getEnvironment() {
    if (INSTANCE == null) {
      // If there's no instance, create one.
      return getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, DEFAULT_NAME);
    } else {
      // else return the current one.
      refCount.incrementAndGet();
      return INSTANCE;
    }
  }

  /**
   * Gets the OrtEnvironment. If there is not an environment currently created, it creates one using
   * the supplied name and {@link OrtLoggingLevel#ORT_LOGGING_LEVEL_WARNING}.
   *
   * <p>If the environment already exists then it returns the existing one and logs a warning if the
   * name or log level is different from the requested one.
   *
   * @param name The logging id of the environment.
   * @return The OrtEnvironment singleton.
   */
  public static OrtEnvironment getEnvironment(String name) {
    return getEnvironment(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, name);
  }

  /**
   * Gets the OrtEnvironment. If there is not an environment currently created, it creates one using
   * the {@link OrtEnvironment#DEFAULT_NAME} and the supplied logging level.
   *
   * <p>If the environment already exists then it returns the existing one and logs a warning if the
   * name or log level is different from the requested one.
   *
   * @param logLevel The logging level to use.
   * @return The OrtEnvironment singleton.
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

  /**
   * Creates an OrtEnvironment using the specified global thread pool options. Note unlike the other
   * {@code getEnvironment} methods if there already is an existing OrtEnvironment this call throws
   * {@link IllegalStateException} as we cannot guarantee that the environment has the appropriate
   * thread pool configuration.
   *
   * @param loggingLevel The logging level to use.
   * @param name The log id.
   * @param threadOptions The global thread pool options.
   * @return The OrtEnvironment singleton.
   */
  public static synchronized OrtEnvironment getEnvironment(
      OrtLoggingLevel loggingLevel, String name, ThreadingOptions threadOptions) {
    if (INSTANCE == null) {
      try {
        INSTANCE = new OrtEnvironment(loggingLevel, name, threadOptions);
        curLogLevel = loggingLevel;
        curLoggingName = name;
      } catch (OrtException e) {
        throw new IllegalStateException("Failed to create OrtEnvironment", e);
      }
      refCount.incrementAndGet();
      return INSTANCE;
    } else {
      // As the thread pool state is unknown, and that's probably not what the user wanted.
      throw new IllegalStateException(
          "Tried to specify the thread pool when creating an OrtEnvironment, but one already exists.");
    }
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
   * Create an OrtEnvironment using the specified name, log level and threading options.
   *
   * @param loggingLevel The logging level to use.
   * @param name The logging id of the environment.
   * @param threadOptions The global thread pool configuration.
   * @throws OrtException If the environment couldn't be created.
   */
  private OrtEnvironment(OrtLoggingLevel loggingLevel, String name, ThreadingOptions threadOptions)
      throws OrtException {
    nativeHandle =
        createHandle(
            OnnxRuntime.ortApiHandle, loggingLevel.getValue(), name, threadOptions.nativeHandle);
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
   * Gets the providers available in this environment.
   *
   * @return An enum set of the available execution providers.
   */
  public static EnumSet<OrtProvider> getAvailableProviders() {
    return OnnxRuntime.providers.clone();
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
   * Creates the native object with a global thread pool.
   *
   * @param apiHandle The API pointer.
   * @param loggingLevel The logging level.
   * @param name The name of the environment.
   * @param threadOptionsHandle The threading options handle.
   * @return The pointer to the native object.
   * @throws OrtException If the creation failed.
   */
  private static native long createHandle(
      long apiHandle, int loggingLevel, String name, long threadOptionsHandle) throws OrtException;

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

  /**
   * Controls the global thread pools in the environment. Only used if the session is constructed
   * using an options with {@link OrtSession.SessionOptions#disablePerSessionThreads()} set.
   */
  public static final class ThreadingOptions implements AutoCloseable {

    private final long nativeHandle;

    private boolean closed = false;

    /** Create an empty threading options. */
    public ThreadingOptions() {
      nativeHandle = createThreadingOptions(OnnxRuntime.ortApiHandle);
    }

    /** Checks if the ThreadingOptions is closed, if so throws {@link IllegalStateException}. */
    private void checkClosed() {
      if (closed) {
        throw new IllegalStateException("Trying to use a closed ThreadingOptions");
      }
    }

    /** Closes the threading options. */
    @Override
    public void close() {
      if (!closed) {
        closeThreadingOptions(OnnxRuntime.ortApiHandle, nativeHandle);
        closed = true;
      } else {
        throw new IllegalStateException("Trying to close a closed ThreadingOptions.");
      }
    }

    /**
     * Sets the number of threads available for inter-op parallelism (i.e. running multiple ops in
     * parallel).
     *
     * <p>Setting it to 0 will allow ORT to choose the number of threads, setting it to 1 will cause
     * the main thread to be used (i.e., no thread pools will be used).
     *
     * @param numThreads The number of threads.
     * @throws OrtException If there was an error in native code.
     */
    public void setGlobalInterOpNumThreads(int numThreads) throws OrtException {
      checkClosed();
      if (numThreads < 0) {
        throw new IllegalArgumentException("Number of threads must be non-negative.");
      }
      setGlobalInterOpNumThreads(OnnxRuntime.ortApiHandle, nativeHandle, numThreads);
    }

    /**
     * Sets the number of threads available for intra-op parallelism (i.e. within a single op).
     *
     * <p>Setting it to 0 will allow ORT to choose the number of threads, setting it to 1 will cause
     * the main thread to be used (i.e., no thread pools will be used).
     *
     * @param numThreads The number of threads.
     * @throws OrtException If there was an error in native code.
     */
    public void setGlobalIntraOpNumThreads(int numThreads) throws OrtException {
      checkClosed();
      if (numThreads < 0) {
        throw new IllegalArgumentException("Number of threads must be non-negative.");
      }
      setGlobalIntraOpNumThreads(OnnxRuntime.ortApiHandle, nativeHandle, numThreads);
    }

    /**
     * Allows spinning of thread pools when their queues are empty. This call sets the value for
     * both inter-op and intra-op thread pools.
     *
     * <p>If the CPU usage is very high then do not enable this.
     *
     * @param allowSpinning If true allow the thread pools to spin.
     * @throws OrtException If there was an error in native code.
     */
    public void setGlobalSpinControl(boolean allowSpinning) throws OrtException {
      checkClosed();
      setGlobalSpinControl(OnnxRuntime.ortApiHandle, nativeHandle, allowSpinning ? 1 : 0);
    }

    /**
     * When this is set it causes intra-op and inter-op thread pools to flush denormal values to
     * zero.
     *
     * @throws OrtException If there was an error in native code.
     */
    public void setGlobalDenormalAsZero() throws OrtException {
      checkClosed();
      setGlobalDenormalAsZero(OnnxRuntime.ortApiHandle, nativeHandle);
    }

    private static native long createThreadingOptions(long apiHandle);

    private native void setGlobalIntraOpNumThreads(
        long apiHandle, long nativeHandle, int numThreads) throws OrtException;

    private native void setGlobalInterOpNumThreads(
        long apiHandle, long nativeHandle, int numThreads) throws OrtException;

    private native void setGlobalSpinControl(long apiHandle, long nativeHandle, int allowSpinning)
        throws OrtException;

    private native void setGlobalDenormalAsZero(long apiHandle, long nativeHandle)
        throws OrtException;

    private native void closeThreadingOptions(long apiHandle, long nativeHandle);
  }
}

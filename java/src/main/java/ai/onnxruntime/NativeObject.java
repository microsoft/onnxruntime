package ai.onnxruntime;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.logging.Logger;

/**
 * This is the base class for anything backed by JNI. It manages open versus closed state and usage
 * handling.
 */
abstract class NativeObject implements AutoCloseable {

  static {
    try {
      OnnxRuntime.init();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load onnx-runtime library", e);
    }
  }

  private static final Logger logger = Logger.getLogger(NativeObject.class.getName());

  private final Object handleLock = new Object();

  private final long handle;

  private volatile boolean closed;

  /** This is used to ensure the close operation will only proceed when there are no more usages. */
  private final AtomicInteger activeUsagesCount;

  NativeObject(long handle) {
    this.handle = handle;
    this.closed = false;
    this.activeUsagesCount = new AtomicInteger(1);
  }

  /**
   * Generates a description with the backing native object's handle.
   *
   * @return the description
   */
  @Override
  public String toString() {
    return getClass().getSimpleName() + "@" + Long.toHexString(handle);
  }

  /**
   * Check if the resource is closed.
   *
   * @return true if closed
   */
  public final boolean isClosed() {
    return closed;
  }

  /**
   * This internal method allows implementations to specify the manner in which this object's
   * backing native object(s) are released/freed/closed.
   *
   * @param handle a long representation of the address of the backing native object.
   */
  abstract void doClose(long handle);

  /**
   * Releases any native resources related to this object.
   *
   * <p>This method must be called or else the application will leak off-heap memory. It is best
   * practice to use a try-with-resources which will ensure this method always be called. This
   * method will block until any active usages are complete.
   */
  @Override
  public void close() {
    synchronized (handleLock) {
      if (closed) {
        return;
      }
      /*
       * ACTIVE USAGE COUNT UPDATE:
       */
      if (activeUsagesCount.decrementAndGet() > 0) {
        logger.warning("Waiting to close: " + toString());
        /*
         * In this case, there are still usages. Wait here until the last one informs us it is done.
         */
        try {
          handleLock.wait();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException("close interrupted", e);
        }
      }
      /*
       * In the else case, there are no usages out still.
       */
      doClose(handle);
      closed = true;
    }
  }

  /** A managed usage to the backing native object. */
  interface NativeUsage extends AutoCloseable {
    /**
     * Read the handle of the backing native object.
     *
     * @return a long representation of the address of the backing native object.
     */
    long handle();

    /** Use this method when the usage is complete. */
    @Override
    void close();
  }

  /**
   * Get a usage to the backing native object. This method ensures the object is open. It is
   * recommended this be used with a try-with-resources to ensure the {@link NativeUsage} is closed
   * and does not leak out of scope. The usage should not be shared between threads.
   *
   * @return a usage from which the backing native object's handle can be used.
   */
  final NativeUsage use() {
    return new DefaultNativeUsage();
  }

  /**
   * A nullable implementation of use().
   *
   * @param object possibly null NativeObject
   * @return a {@link NativeUsage} for that object, or when null, {@link NativeUsage} which returns
   *     address 0
   */
  static final NativeUsage useOptionally(NativeObject object) {
    if (object == null) {
      return NullNativeUsage.INSTANCE;
    }
    return object.use();
  }

  /** A {@link NativeUsage} implementation for non-null Java objects. */
  private final class DefaultNativeUsage implements NativeUsage {

    private boolean usageClosed;

    public DefaultNativeUsage() {
      this.usageClosed = false;
      /*
       * ACTIVE USAGE COUNT UPDATE:
       */
      if (activeUsagesCount.getAndIncrement() <= 0) {
        /*
         * The old usage count less than or equal to 0 indicating closed, so an exception is thrown. However, it
         * is necessary to call release() here prior to throwing, since the close() (which usually calls
         * release()) will not be called upon exiting the try-with-resources due to the exception.
         */
        release();
        throw new IllegalStateException(
            NativeObject.this.getClass().getSimpleName() + " has been closed already.");
      }
    }

    private void release() {
      /*
       * ACTIVE USAGE COUNT UPDATE:
       */
      if (activeUsagesCount.decrementAndGet() == 0) {
        /*
         * This is the last usage, so inform the thread waiting in NativeObject.close() that we are done.
         */
        synchronized (handleLock) {
          handleLock.notifyAll();
        }
      }
    }

    @Override
    public long handle() {
      if (usageClosed) {
        throw new IllegalStateException("Native usage closed");
      }
      return handle;
    }

    @Override
    public void close() {
      if (usageClosed) {
        throw new IllegalStateException("Native usage closed");
      }
      release();
      usageClosed = true;
    }
  }

  /** A {@link NativeUsage} implementation for null Java objects. */
  private static final class NullNativeUsage implements NativeUsage {

    private static final NativeUsage INSTANCE = new NullNativeUsage();

    private NullNativeUsage() {}

    @Override
    public long handle() {
      return 0L;
    }

    @Override
    public void close() {
      // pass
    }
  }
}

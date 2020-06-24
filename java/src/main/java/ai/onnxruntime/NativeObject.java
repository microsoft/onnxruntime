package ai.onnxruntime;

import java.io.IOException;
import java.util.concurrent.atomic.AtomicInteger;

/**
 * This is the base class for anything backed by JNI. It manages open versus closed state and
 * reference handling.
 */
abstract class NativeObject implements AutoCloseable {

  static {
    try {
      OnnxRuntime.init();
    } catch (IOException e) {
      throw new RuntimeException("Failed to load onnx-runtime library", e);
    }
  }

  private final Object handleLock = new Object();

  private final long handle;

  private volatile boolean closed;

  private final AtomicInteger referenceCount;

  NativeObject(long handle) {
    this.handle = handle;
    this.closed = false;
    this.referenceCount = new AtomicInteger(1);
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
       * REFERENCE COUNT UPDATE:
       */
      if (referenceCount.decrementAndGet() > 0) {
        /*
         * In this case, there are still references being used. Wait here until the last one informs us it is
         * done.
         */
        try {
          handleLock.wait();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException("close interrupted", e);
        }
      }
      /*
       * In the else case, there are no references out still being used.
       */
      doClose(handle);
      closed = true;
    }
  }

  /** A managed reference to the backing native object. */
  interface NativeReference extends AutoCloseable {
    /**
     * Read the handle of the backing native object.
     *
     * @return a long representation of the address of the backing native object.
     */
    long handle();

    /** Use this method when the native object's reference is done being used. */
    @Override
    void close();
  }

  /**
   * Get a reference to the backing native object. This method ensures the object is open. It is
   * recommended this be used with a try-with-resources to ensure the NativeReference is closed and
   * does not leak out of scope. The reference should not be shared between threads.
   *
   * @return a reference from which the backing native object's handle can be used.
   */
  final NativeReference reference() {
    return new DefaultNativeReference();
  }

  /**
   * A nullable implementation of reference().
   *
   * @param object possibly null NativeObject
   * @return a NativeReference for that object, or when null, NativeReference which returns address
   *     0
   */
  static final NativeReference optionalReference(NativeObject object) {
    if (object == null) {
      return NullNativeReference.INSTANCE;
    }
    return object.reference();
  }

  /** A NativeReference implementation for non-null Java objects. */
  private final class DefaultNativeReference implements NativeReference {

    private boolean referenceClosed;

    public DefaultNativeReference() {
      this.referenceClosed = false;
      /*
       * REFERENCE COUNT UPDATE:
       */
      if (referenceCount.getAndIncrement() == 0) {
        /*
         * The old reference count was 0 indicating closed, so an exception is thrown. However, it is necessary
         * to call release() here prior to throwing, since the close() (which usually calls release()) will not
         * be called upon exiting the try-with-resources due to the exception.
         */
        release();
        throw new IllegalStateException(
            NativeObject.this.getClass().getSimpleName() + " has been closed already.");
      }
    }

    private void release() {
      /*
       * REFERENCE COUNT UPDATE:
       */
      if (referenceCount.decrementAndGet() == 0) {
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
      if (referenceClosed) {
        throw new IllegalStateException("Reference closed");
      }
      return handle;
    }

    @Override
    public void close() {
      if (referenceClosed) {
        throw new IllegalStateException("Reference closed");
      }
      release();
      referenceClosed = true;
    }
  }

  /** A NativeReference implementation for null Java objects. */
  private static final class NullNativeReference implements NativeReference {

    private static final NativeReference INSTANCE = new NullNativeReference();

    private NullNativeReference() {}

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

package ai.onnxruntime;

import java.io.IOException;

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

  private int activeCount;

  private final NativeReference reference;

  NativeObject(long handle) {
    this.handle = handle;
    this.closed = false;
    this.reference = new DefaultNativeReference();
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

  /** Throw an exception if the resource is closed. */
  final void ensureOpen() {
    if (closed) {
      throw new IllegalStateException(
          this.getClass().getSimpleName() + " has been closed already.");
    }
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
      while (activeCount > 0) {
        try {
          handleLock.wait();
        } catch (InterruptedException e) {
          Thread.currentThread().interrupt();
          throw new RuntimeException("close interrupted", e);
        }
      }
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
   * does not leak out of scope.
   *
   * @return a reference from which the backing native object's handle can be used.
   */
  final NativeReference reference() {
    synchronized (handleLock) {
      ensureOpen();
      ++activeCount;
    }
    return reference;
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

    @Override
    public long handle() {
      return handle;
    }

    @Override
    public void close() {
      synchronized (handleLock) {
        if (closed) {
          return;
        }
        if (--activeCount == 0) {
          handleLock.notifyAll();
        }
      }
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

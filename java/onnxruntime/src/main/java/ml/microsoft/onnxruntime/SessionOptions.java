package ml.microsoft.onnxruntime;

public class SessionOptions {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  public SessionOptions() throws OrtException {
    initHandle();
  }
  @Override
  protected void finalize() throws Throwable {
    dispose();
    super.finalize();
  }
  private long nativeHandle;
  private native void initHandle() throws OrtException;
  public native void dispose();
  public native void appendNnapiExecutionProvider() throws OrtException;
}

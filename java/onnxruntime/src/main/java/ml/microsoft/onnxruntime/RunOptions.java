package ml.microsoft.onnxruntime;

public class RunOptions {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  public RunOptions() throws OrtException {
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
}

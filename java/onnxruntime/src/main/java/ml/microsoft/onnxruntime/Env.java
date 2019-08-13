package ml.microsoft.onnxruntime;

public class Env {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  public Env(LoggingLevel loggingLevel, String logId) throws OrtException {
    initHandle(loggingLevel, logId);
  }
  @Override
  protected void finalize() throws Throwable {
    dispose();
    super.finalize();
  }
  private long nativeHandle;
  private native void initHandle(LoggingLevel loggingLevel, String logId) throws OrtException;
  public native void dispose();
}

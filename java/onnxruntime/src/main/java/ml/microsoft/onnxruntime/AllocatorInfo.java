package ml.microsoft.onnxruntime;

public class AllocatorInfo {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  public native static AllocatorInfo createCpu(AllocatorType allocatorType, MemType memType);

  private AllocatorInfo() {
  }

  private long nativeHandle;
  public native void dispose();
}

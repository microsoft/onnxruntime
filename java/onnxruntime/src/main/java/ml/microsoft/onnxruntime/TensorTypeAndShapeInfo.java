package ml.microsoft.onnxruntime;

public class TensorTypeAndShapeInfo {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  private TensorTypeAndShapeInfo() {
  }
  private long nativeHandle;
  public native void dispose();
  public native long[] getShape() throws OrtException;
  public native long getDimensionsCount() throws OrtException;
  public native long getElementCount() throws OrtException;
}

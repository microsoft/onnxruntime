package ml.microsoft.onnxruntime;

public class TypeInfo {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  private TypeInfo() {
  }
  private long nativeHandle;
  public native void dispose();
  public native TensorTypeAndShapeInfo getTensorTypeAndShapeInfo() throws OrtException;
}

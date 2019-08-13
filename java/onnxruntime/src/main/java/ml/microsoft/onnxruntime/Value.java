package ml.microsoft.onnxruntime;

import java.nio.ByteBuffer;

public class Value {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  private Value() {
  }
  @Override
  protected void finalize() throws Throwable {
    dispose();
    super.finalize();
  }
  private long nativeHandle;
  public native void dispose();
  public native static Value createTensor(AllocatorInfo allocatorInfo, ByteBuffer data,
      long[] shape, TensorElementDataType type)
      throws OrtException;
  public native ByteBuffer getTensorMutableData() throws OrtException;
  public native TensorTypeAndShapeInfo getTensorTypeAndShapeInfo() throws OrtException;
}

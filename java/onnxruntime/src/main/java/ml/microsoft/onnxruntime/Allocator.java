package ml.microsoft.onnxruntime;

import java.nio.ByteBuffer;

public class Allocator {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  private Allocator() {
  }

  private long nativeHandle;
  public native void dispose();

  public static native Allocator createDefault() throws OrtException;
  public static native ByteBuffer alloc();
  public static native void free(ByteBuffer buffer);
  public static native AllocatorInfo getInfo();
}

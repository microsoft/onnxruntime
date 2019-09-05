package ml.microsoft.onnxruntime;

public class Session {
  static {
    System.loadLibrary("onnxruntime-jni");
  }
  public Session(Env env, String modelPath, SessionOptions sessionOptions) throws OrtException {
    initHandle(env, modelPath, sessionOptions);
  }
  @Override
  protected void finalize() throws Throwable {
    dispose();
    super.finalize();
  }
  private long nativeHandle;
  private native void initHandle(Env env, String modelPath, SessionOptions sessionOptions)
      throws OrtException;
  public native void dispose();
  public native Value[] run(RunOptions runOptions, String[] input_names, Value[] input_values,
      String[] output_names) throws OrtException;
  public native long getInputCount() throws OrtException;
  public native long getOutputCount() throws OrtException;
  public native String getInputName(long index, Allocator allocator) throws OrtException;
  public native String getOutputName(long index, Allocator allocator) throws OrtException;
  public native TypeInfo getInputTypeInfo(long index) throws OrtException;
  public native TypeInfo getOutputTypeInfo(long index) throws OrtException;
}

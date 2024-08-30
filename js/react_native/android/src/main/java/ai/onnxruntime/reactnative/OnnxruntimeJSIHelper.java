package ai.onnxruntime.reactnative;

import androidx.annotation.NonNull;
import com.facebook.react.bridge.JavaScriptContextHolder;
import com.facebook.react.bridge.ReactApplicationContext;
import com.facebook.react.bridge.ReactContextBaseJavaModule;
import com.facebook.react.bridge.ReactMethod;
import com.facebook.react.module.annotations.ReactModule;
import com.facebook.react.modules.blob.BlobModule;

@ReactModule(name = OnnxruntimeJSIHelper.NAME)
public class OnnxruntimeJSIHelper extends ReactContextBaseJavaModule {
  public static final String NAME = "OnnxruntimeJSIHelper";

  private static ReactApplicationContext reactContext;
  protected BlobModule blobModule;

  public OnnxruntimeJSIHelper(ReactApplicationContext context) {
    super(context);
    reactContext = context;
  }

  @Override
  @NonNull
  public String getName() {
    return NAME;
  }

  public void checkBlobModule() {
    if (blobModule == null) {
      blobModule = getReactApplicationContext().getNativeModule(BlobModule.class);
      if (blobModule == null) {
        throw new RuntimeException("BlobModule is not initialized");
      }
    }
  }

  @ReactMethod(isBlockingSynchronousMethod = true)
  public boolean install() {
    try {
      System.loadLibrary("onnxruntimejsihelper");
      JavaScriptContextHolder jsContext = getReactApplicationContext().getJavaScriptContextHolder();
      nativeInstall(jsContext.get(), this);
      return true;
    } catch (Exception exception) {
      return false;
    }
  }

  public byte[] getBlobBuffer(String blobId, int offset, int size) {
    checkBlobModule();
    byte[] bytes = blobModule.resolve(blobId, offset, size);
    blobModule.remove(blobId);
    if (bytes == null) {
      throw new RuntimeException("Failed to resolve Blob #" + blobId + "! Not found.");
    }
    return bytes;
  }

  public String createBlob(byte[] buffer) {
    checkBlobModule();
    String blobId = blobModule.store(buffer);
    if (blobId == null) {
      throw new RuntimeException("Failed to create Blob!");
    }
    return blobId;
  }

  public static native void nativeInstall(long jsiPointer, OnnxruntimeJSIHelper instance);
}

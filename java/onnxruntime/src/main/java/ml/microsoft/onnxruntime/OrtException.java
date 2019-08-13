package ml.microsoft.onnxruntime;

public class OrtException extends Exception {
  private ErrorCode errorCode;

  public OrtException(String message, int errorCodeInt) {
    super(message);
    this.errorCode = ErrorCode.values()[errorCodeInt];
  }

  public ErrorCode getErrorCode() {
    return errorCode;
  }
}

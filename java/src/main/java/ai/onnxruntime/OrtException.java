/*
 * Copyright (c) 2019, 2023, Oracle and/or its affiliates. All rights reserved.
 * Licensed under the MIT License.
 */
package ai.onnxruntime;

/** An exception which contains the error message and code produced by the native onnxruntime. */
public class OrtException extends Exception {
  private static final long serialVersionUID = 1L;

  /** The OrtErrorCode for this exception. */
  private final OrtErrorCode errorCode;

  /**
   * Creates an OrtException with a default Java error code and the specified message.
   *
   * @param message The message to use.
   */
  public OrtException(String message) {
    super(message);
    this.errorCode = OrtErrorCode.ORT_JAVA_UNKNOWN;
  }

  /**
   * Used to throw an exception from native code as it handles the enum lookup in Java.
   *
   * @param code The error code.
   * @param message The message.
   */
  public OrtException(int code, String message) {
    this(OrtErrorCode.mapFromInt(code), message);
  }

  /**
   * Creates an OrtException using the specified error code and message.
   *
   * @param code The error code from the native runtime.
   * @param message The error message.
   */
  public OrtException(OrtErrorCode code, String message) {
    super("Error code - " + code + " - message: " + message);
    this.errorCode = code;
  }

  /**
   * Return the error code.
   *
   * @return The error code.
   */
  public OrtErrorCode getCode() {
    return errorCode;
  }

  /**
   * Maps the {@code OrtErrorCode} struct in {@code onnxruntime_c_api.h} with an additional entry
   * for Java side errors.
   */
  public enum OrtErrorCode {
    /** An unknown error occurred in the Java API. */
    ORT_JAVA_UNKNOWN(-1),
    /** The operation completed without error. */
    ORT_OK(0),
    /** The operation failed. */
    ORT_FAIL(1),
    /** The operation received an invalid argument. */
    ORT_INVALID_ARGUMENT(2),
    /** The operation could not load the required file. */
    ORT_NO_SUCHFILE(3),
    /** The operation could not use the model. */
    ORT_NO_MODEL(4),
    /** There is an internal error in the ORT engine. */
    ORT_ENGINE_ERROR(5),
    /** The operation threw a runtime exception. */
    ORT_RUNTIME_EXCEPTION(6),
    /** The provided protobuf was invalid. */
    ORT_INVALID_PROTOBUF(7),
    /** The model was loaded. */
    ORT_MODEL_LOADED(8),
    /** The requested operation has not been implemented. */
    ORT_NOT_IMPLEMENTED(9),
    /** The ONNX graph is invalid. */
    ORT_INVALID_GRAPH(10),
    /** The ORT execution provider failed. */
    ORT_EP_FAIL(11);

    private final int value;

    private static final OrtErrorCode[] values = new OrtErrorCode[12];

    static {
      for (OrtErrorCode ot : OrtErrorCode.values()) {
        if (ot != ORT_JAVA_UNKNOWN) {
          values[ot.value] = ot;
        }
      }
    }

    OrtErrorCode(int value) {
      this.value = value;
    }

    /**
     * Maps from an int in native land into an OrtErrorCode instance.
     *
     * @param value The value to lookup.
     * @return The enum instance.
     */
    public static OrtErrorCode mapFromInt(int value) {
      if ((value >= 0) && (value < values.length)) {
        return values[value];
      } else {
        return ORT_JAVA_UNKNOWN;
      }
    }
  }
}

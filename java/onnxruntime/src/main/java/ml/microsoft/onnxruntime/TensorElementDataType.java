package ml.microsoft.onnxruntime;

public enum TensorElementDataType {
  UNDEFINED,
  FLOAT, // maps to c type float
  UINT8, // maps to c type uint8_t
  INT8, // maps to c type int8_t
  UINT16, // maps to c type uint16_t
  INT16, // maps to c type int16_t
  INT32, // maps to c type int32_t
  INT64, // maps to c type int64_t
  STRING, // maps to c++ type std::string
  BOOL,
  FLOAT16,
  DOUBLE, // maps to c type double
  UINT32, // maps to c type uint32_t
  UINT64, // maps to c type uint64_t
  COMPLEX64, // complex with float32 real and imaginary components
  COMPLEX128, // complex with float64 real and imaginary components
  BFLOAT16, // Non-IEEE floating-point format based on IEEE754 single-precision
}

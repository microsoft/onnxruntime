#pragma once

namespace onnx_c_ops {

class Status {
public:
  int code;
  inline Status() : code(1) {}
  inline Status(int code) : code(code) {}
  inline Status &operator=(const Status &other) {
    code = other.code;
    return *this;
  }
  inline bool IsOK() const { return code == 1; }
  inline int Code() const { return code; }
  inline bool operator==(const Status &other) const { return code == other.code; }
  inline bool operator!=(const Status &other) const { return !(*this == other); }
  inline static Status OK() { return Status(1); }
};

} // namespace onnx_c_ops

#_init and _fini should be local
VERS_1.0 {
  global:
    PyInit_onnxruntime_pybind11_state;
    extern "C++" {
       onnx::*;
    };

  # Hide everything else.
  local:
    *;
};

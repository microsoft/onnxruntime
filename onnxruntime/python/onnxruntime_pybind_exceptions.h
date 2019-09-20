#include <pybind11/pybind11.h>
#include <stdexcept>

namespace onnxruntime {
namespace python {

// onnxruntime::python exceptions based on Status codes.
class Fail : public std::runtime_error {
 public:
  explicit Fail(const std::string& what) : std::runtime_error(what) {}
};
class InvalidArgument : public std::runtime_error {
 public:
  explicit InvalidArgument(const std::string& what) : std::runtime_error(what) {}
};
class NoSuchFile : public std::runtime_error {
 public:
  explicit NoSuchFile(const std::string& what) : std::runtime_error(what) {}
};
class NoModel : public std::runtime_error {
 public:
  explicit NoModel(const std::string& what) : std::runtime_error(what) {}
};
class EngineError : public std::runtime_error {
 public:
  explicit EngineError(const std::string& what) : std::runtime_error(what) {}
};

void RegisterExceptions(pybind11::module& m) {
  pybind11::register_exception<Fail>(m, "Fail");
  pybind11::register_exception<InvalidArgument>(m, "InvalidArgument");
  pybind11::register_exception<NoSuchFile>(m, "NoSuchFile");
  pybind11::register_exception<NoModel>(m, "NoModel");
  pybind11::register_exception<EngineError>(m, "EngineError");
}
#define ORT_PYBIND_THROW_IF_ERROR(expr)    \
  do {                                     \
    auto _status = (expr);                 \
    auto _msg = _status.ToString();        \
    if ((!_status.IsOK())) {               \
      switch (_status.Code()) {            \
        case StatusCode::FAIL:             \
          throw Fail(_msg);                \
          break;                           \
        case StatusCode::INVALID_ARGUMENT: \
          throw InvalidArgument(_msg);     \
          break;                           \
        case StatusCode::NO_SUCHFILE:      \
          throw NoSuchFile(_msg);          \
          break;                           \
        case StatusCode::NO_MODEL:         \
          throw NoModel(_msg);             \
          break;                           \
        case StatusCode::ENGINE_ERROR:     \
          throw EngineError(_msg);         \
          break;                           \
        default:                           \
          throw std::runtime_error(_msg);  \
          break;                           \
      }                                    \
    }                                      \
  } while (0)

}  // namespace python
}  // namespace onnxruntime

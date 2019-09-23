#include <pybind11/pybind11.h>
#include <stdexcept>

namespace onnxruntime {
namespace python {

// onnxruntime::python exceptions map 1:1 to onnxruntime:common::StatusCode enum.
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
class RuntimeException : public std::runtime_error {
 public:
  explicit RuntimeException(const std::string& what) : std::runtime_error(what) {}
};
class InvalidProtobuf : public std::runtime_error {
 public:
  explicit InvalidProtobuf(const std::string& what) : std::runtime_error(what) {}
};
class ModelLoaded : public std::runtime_error {
 public:
  explicit ModelLoaded(const std::string& what) : std::runtime_error(what) {}
};
class NotImplemented : public std::runtime_error {
 public:
  explicit NotImplemented(const std::string& what) : std::runtime_error(what) {}
};
class InvalidGraph : public std::runtime_error {
 public:
  explicit InvalidGraph(const std::string& what) : std::runtime_error(what) {}
};
class EPFail : public std::runtime_error {
 public:
  explicit EPFail(const std::string& what) : std::runtime_error(what) {}
};

void RegisterExceptions(pybind11::module& m) {
  pybind11::register_exception<Fail>(m, "Fail");
  pybind11::register_exception<InvalidArgument>(m, "InvalidArgument");
  pybind11::register_exception<NoSuchFile>(m, "NoSuchFile");
  pybind11::register_exception<NoModel>(m, "NoModel");
  pybind11::register_exception<EngineError>(m, "EngineError");
  pybind11::register_exception<RuntimeException>(m, "RuntimeException");
  pybind11::register_exception<InvalidProtobuf>(m, "InvalidProtobuf");
  pybind11::register_exception<ModelLoaded>(m, "ModelLoaded");
  pybind11::register_exception<NotImplemented>(m, "NotImplemented");
  pybind11::register_exception<InvalidGraph>(m, "InvalidGraph");
  pybind11::register_exception<EPFail>(m, "EPFail");
}
#define ORT_PYBIND_THROW_IF_ERROR(expr)     \
  do {                                      \
    auto _status = (expr);                  \
    auto _msg = _status.ToString();         \
    if ((!_status.IsOK())) {                \
      switch (_status.Code()) {             \
        case StatusCode::FAIL:              \
          throw Fail(_msg);                 \
          break;                            \
        case StatusCode::INVALID_ARGUMENT:  \
          throw InvalidArgument(_msg);      \
          break;                            \
        case StatusCode::NO_SUCHFILE:       \
          throw NoSuchFile(_msg);           \
          break;                            \
        case StatusCode::NO_MODEL:          \
          throw NoModel(_msg);              \
          break;                            \
        case StatusCode::ENGINE_ERROR:      \
          throw EngineError(_msg);          \
          break;                            \
        case StatusCode::RUNTIME_EXCEPTION: \
          throw RuntimeException(_msg);     \
          break;                            \
        case StatusCode::INVALID_PROTOBUF:  \
          throw InvalidProtobuf(_msg);      \
          break;                            \
        case StatusCode::NOT_IMPLEMENTED:   \
          throw NotImplemented(_msg);       \
          break;                            \
        case StatusCode::INVALID_GRAPH:     \
          throw InvalidGraph(_msg);         \
          break;                            \
        case StatusCode::EP_FAIL:           \
          throw EPFail(_msg);               \
          break;                            \
        default:                            \
          throw std::runtime_error(_msg);   \
          break;                            \
      }                                     \
    }                                       \
  } while (0)

}  // namespace python
}  // namespace onnxruntime

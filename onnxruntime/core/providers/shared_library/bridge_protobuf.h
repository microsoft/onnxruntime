#include <string>

#define SWIG
namespace google {
namespace protobuf {
namespace internal {

inline int
ToIntSize(size_t size) {
  return static_cast<int>(size);
}

size_t StringSpaceUsedExcludingSelfLong(const std::string& str);
const ::std::string& GetEmptyStringAlreadyInited();

}  // namespace internal
}  // namespace protobuf
}  // namespace google

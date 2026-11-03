#include <bit>
#include <cstdint>
#include <fstream>
#include <limits>

static_assert(std::endian::native == std::endian::little);

class NumericMismatch final : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void TransferOriginalOutput(std::fstream& stream, bool capture, const Ort::Value& output,
                            const std::string& where) {
  const auto info = output.GetTensorTypeAndShapeInfo();
  const auto type = info.GetElementType();
  const auto shape = info.GetShape();
  const auto count = info.GetElementCount();
  const size_t element_bytes = type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16 ? 2 :
                               type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ? 4 : 0;
  if (element_bytes == 0 || count > std::numeric_limits<size_t>::max() / element_bytes)
    throw std::runtime_error("Unsupported reference output: " + where);
  const uint64_t header[] = {0x48434f5249473031ull, static_cast<uint64_t>(type),
                             static_cast<uint64_t>(shape.size()), static_cast<uint64_t>(count)};
  auto transfer_metadata = [&](const void* expected, size_t bytes) {
    if (capture) {
      stream.write(static_cast<const char*>(expected), bytes);
    } else {
      std::vector<unsigned char> recorded(bytes);
      stream.read(reinterpret_cast<char*>(recorded.data()), bytes);
      if (!stream) throw std::runtime_error("Truncated original metadata: " + where);
      if (std::memcmp(recorded.data(), expected, bytes) != 0)
        throw NumericMismatch("Original output shape/type mismatch: " + where);
    }
    if (!stream) throw std::runtime_error("Original metadata I/O failed: " + where);
  };
  transfer_metadata(header, sizeof(header));
  for (int64_t dimension : shape) {
    if (dimension < 0) throw std::runtime_error("Unresolved reference output shape: " + where);
    transfer_metadata(&dimension, sizeof(dimension));
  }
  const size_t bytes = count * element_bytes;
  const auto* actual = static_cast<const char*>(output.GetTensorRawData());
  if (capture) {
    stream.write(actual, bytes);
  } else {
    std::vector<char> original(bytes);
    stream.read(original.data(), bytes);
    if (!stream) throw std::runtime_error("Truncated original payload: " + where);
    if (std::memcmp(original.data(), actual, bytes) != 0)
      throw NumericMismatch("Separate original-provider output mismatch: " + where);
  }
  if (!stream) throw std::runtime_error("Original output I/O failed: " + where);
}

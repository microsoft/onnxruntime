#include "lib/Api.Image/pch.h"
#include "inc/DisjointBufferHelpers.h"

namespace _winml {

static void LoadOrStoreDisjointBuffers(
  bool should_load_buffer,
  size_t num_buffers,
  std::function<gsl::span<byte>(size_t)> get_buffer,
  gsl::span<byte>& buffer_span
) {
  auto size_in_bytes = buffer_span.size_bytes();
  auto buffer = buffer_span.data();
  size_t offset_in_bytes = 0;
  for (size_t i = 0; i < num_buffers && size_in_bytes > offset_in_bytes; i++) {
    auto span = get_buffer(i);
    auto current_size_in_bytes = span.size_bytes();
    auto current_buffer = span.data();

    if (size_in_bytes - offset_in_bytes < current_size_in_bytes) {
      current_size_in_bytes = size_in_bytes - offset_in_bytes;
    }

    auto offset_buffer = buffer + offset_in_bytes;
    if (should_load_buffer) {
      memcpy(offset_buffer, current_buffer, current_size_in_bytes);
    } else {
      memcpy(current_buffer, offset_buffer, current_size_in_bytes);
    }

    offset_in_bytes += current_size_in_bytes;
  }
}

void LoadSpanFromDisjointBuffers(
  size_t num_buffers, std::function<gsl::span<byte>(size_t)> get_buffer, gsl::span<byte>& buffer_span
) {
  LoadOrStoreDisjointBuffers(true /*load into the span*/, num_buffers, get_buffer, buffer_span);
}

void StoreSpanIntoDisjointBuffers(
  size_t num_buffers, std::function<gsl::span<byte>(size_t)> get_buffer, gsl::span<byte>& buffer_span
) {
  LoadOrStoreDisjointBuffers(false /*store into buffers*/, num_buffers, get_buffer, buffer_span);
}

}  // namespace _winml

#include "png_encoder_decoder.hpp"

/****************************************************************************************\
    This part of the file implements PNG codec on base of libpng library,
    in particular, this code is based on example.c from libpng
    (see otherlibs/_graphics/readme.txt for copyright notice)
    and png2bmp sample from libpng distribution (Copyright (C) 1999-2001 MIYASAKA Masaru)
\****************************************************************************************/

#include "png.h"
#include "zlib.h"

#include <cassert>

#if defined _MSC_VER && _MSC_VER >= 1200
// interaction between '_setjmp' and C++ object destruction is non-portable
#pragma warning(disable : 4611)
#endif

namespace ort_extensions {

bool PngDecoder::IsPng(const uint8_t* bytes, uint64_t num_bytes) {
  // '\0x89PGN\r\n<sub>\n'
  constexpr const uint8_t signature[] = {0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a};  // check start of file for this
  constexpr int signature_len = 8;

  return num_bytes > signature_len && memcmp(bytes, signature, signature_len) == 0;
}

PngDecoder::~PngDecoder() {
  if (png_ptr_) {
    png_structp png_ptr = (png_structp)png_ptr_;
    png_infop info_ptr = (png_infop)info_ptr_;
    png_infop end_info = (png_infop)end_info_;
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);
  }
}

// read a chunk from bytes_
void PngDecoder::ReadDataFromBuf(void* _png_ptr, uint8_t* dst, size_t size) {
  png_structp png_ptr = (png_structp)_png_ptr;
  PngDecoder* decoder = (PngDecoder*)(png_get_io_ptr(png_ptr));
  assert(decoder);

  if (decoder->cur_offset_ + size > decoder->NumBytes()) {
    png_error(png_ptr, "PNG input buffer is incomplete");
    return;
  }

  memcpy(dst, decoder->Bytes() + decoder->cur_offset_, size);
  decoder->cur_offset_ += size;
}

bool PngDecoder::ReadHeader() {
  bool result = false;

  png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
  assert(png_ptr);
  png_infop info_ptr = png_create_info_struct(png_ptr);
  png_infop end_info = png_create_info_struct(png_ptr);

  assert(info_ptr);
  assert(end_info);

  png_ptr_ = png_ptr;
  info_ptr_ = info_ptr;
  end_info_ = end_info;

  if (setjmp(png_jmpbuf(png_ptr)) == 0) {
    png_set_read_fn(png_ptr, this, (png_rw_ptr)ReadDataFromBuf);

    png_uint_32 wdth, hght;
    int bit_depth, color_type, num_trans = 0;
    png_bytep trans;
    png_color_16p trans_values;

    png_read_info(png_ptr, info_ptr);
    png_get_IHDR(png_ptr, info_ptr, &wdth, &hght,
                 &bit_depth, &color_type, 0, 0, 0);

    // set the shape based on what Decode will do. doesn't necessarily match the original image though as we're
    // going to throw away any alpha channel and drop 16 bit depth to 8.
    SetShape(static_cast<int>(hght), static_cast<int>(wdth), 3);

    color_type_ = color_type;
    bit_depth_ = bit_depth;

    // TODO: Do we need to handle these combinations?
    //       We want to end up with 3 channel RGB though so they might not be relevant given we (I think) set the png
    //       code to throw away the alpha channel, reduce 16-bit to 8, and convert grayscale to 3 channel.
    //
    //
    // if (bit_depth <= 8 || bit_depth == 16) {
    //  switch (color_type) {
    //    case PNG_COLOR_TYPE_RGB:
    //    case PNG_COLOR_TYPE_PALETTE:
    //      png_get_tRNS(png_ptr, info_ptr, &trans, &num_trans, &trans_values);
    //      if (num_trans > 0)
    //        m_type = CV_8UC4;
    //      else
    //        m_type = CV_8UC3;
    //      break;
    //    case PNG_COLOR_TYPE_GRAY_ALPHA:
    //    case PNG_COLOR_TYPE_RGB_ALPHA:
    //      m_type = CV_8UC4;
    //      break;
    //    default:
    //      m_type = CV_8UC1;
    //  }
    //  if (bit_depth == 16)
    //    m_type = CV_MAKETYPE(CV_16U, CV_MAT_CN(m_type));
    //
    //}
    result = true;
  }

  return result;
}

bool PngDecoder::DecodeImpl(uint8_t* output, uint64_t out_bytes) {
  bool result = false;

  png_structp png_ptr = (png_structp)png_ptr_;
  png_infop info_ptr = (png_infop)info_ptr_;
  png_infop end_info = (png_infop)end_info_;

  if (setjmp(png_jmpbuf(png_ptr)) == 0) {
    if (bit_depth_ == 16) {
      png_set_strip_16(png_ptr);
    }

    png_set_strip_alpha(png_ptr);

    if (color_type_ == PNG_COLOR_TYPE_PALETTE) {
      png_set_palette_to_rgb(png_ptr);
    }

    if ((color_type_ & PNG_COLOR_MASK_COLOR) == 0 && bit_depth_ < 8) {
      png_set_expand_gray_1_2_4_to_8(png_ptr);
    }

    if (!(color_type_ & PNG_COLOR_MASK_COLOR)) {
      png_set_gray_to_rgb(png_ptr);  // Gray->RGB
    }

    png_set_interlace_handling(png_ptr);
    png_read_update_info(png_ptr, info_ptr);

    const auto& shape = Shape();
    auto height = shape[0];
    auto width = shape[1];
    auto channels = shape[2];

    std::vector<uint8_t*> row_pointers(height, nullptr);
    auto row_size = png_get_rowbytes(png_ptr, info_ptr);
    assert(row_size == width * channels);  // check assumption

    for (int row = 0; row < height; ++row) {
      row_pointers[row] = output + (row * row_size);
    }

    png_read_image(png_ptr, row_pointers.data());
    png_read_end(png_ptr, end_info);

    result = true;
  }

  return result;
}

/////////////////////// PngEncoder ///////////////////

void PngEncoder::WriteDataToBuf(void* _png_ptr, uint8_t* src, size_t size) {
  if (size == 0)
    return;

  png_structp png_ptr = (png_structp)_png_ptr;
  PngEncoder* encoder = (PngEncoder*)(png_get_io_ptr(png_ptr));
  assert(encoder);

  auto& buffer = encoder->Buffer();

  if (encoder->cur_offset_ + size > buffer.size()) {
    assert(false);  // unexpected - this means no compression is occurring when writing
  }

  memcpy(buffer.data() + encoder->cur_offset_, src, size);

  encoder->cur_offset_ += size;
}

void PngEncoder::FlushBuf(void* png_ptr) {
  // we're writing to memory so this is a no-op
}

bool PngEncoder::EncodeImpl() {
  bool result = false;

  png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
  assert(png_ptr);

  png_infop info_ptr = png_create_info_struct(png_ptr);
  assert(info_ptr);

  const auto& shape = Shape();
  const int height = static_cast<int>(shape[0]);
  const int width = static_cast<int>(shape[1]);
  const int channels = static_cast<int>(shape[2]);
  const int depth = 8;

  if (setjmp(png_jmpbuf(png_ptr)) == 0) {
    png_set_write_fn(png_ptr, this, (png_rw_ptr)WriteDataToBuf, (png_flush_ptr)FlushBuf);

    // tune parameters for speed
    // (see http://wiki.linuxquestions.org/wiki/Libpng)
    png_set_filter(png_ptr, PNG_FILTER_TYPE_BASE, PNG_FILTER_SUB);
    png_set_compression_level(png_ptr, Z_BEST_SPEED);
    png_set_compression_strategy(png_ptr, Z_DEFAULT_STRATEGY);

    png_set_IHDR(png_ptr, info_ptr, width, height, depth, PNG_COLOR_TYPE_RGB,
                 PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT,
                 PNG_FILTER_TYPE_DEFAULT);

    png_write_info(png_ptr, info_ptr);

    std::vector<uint8_t*> row_pointers(height, nullptr);
    auto row_size = png_get_rowbytes(png_ptr, info_ptr);
    assert(row_size == width * channels);  // check assumption

    // need non-const to make png lib happy
    uint8_t* orig_image_bytes = const_cast<uint8_t*>(Bytes());

    for (int row = 0; row < height; ++row) {
      row_pointers[row] = orig_image_bytes + (row * row_size);
    }

    png_write_image(png_ptr, row_pointers.data());
    png_write_end(png_ptr, info_ptr);

    Buffer().resize(cur_offset_);  // remove unused bytes so size() is correct

    result = true;
  }

  png_destroy_write_struct(&png_ptr, &info_ptr);

  return result;
}

}  // namespace ort_extensions

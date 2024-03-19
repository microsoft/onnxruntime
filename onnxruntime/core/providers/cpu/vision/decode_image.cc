// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "decode_image.hpp"

#include "jpeglib.h"
#include "png.h"

#include "impl/png_encoder_decoder.hpp"

namespace ort_extensions {

namespace {
struct my_error_mgr {
  struct jpeg_error_mgr pub; /* "public" fields */

  jmp_buf setjmp_buffer; /* for return to caller */
};

typedef struct my_error_mgr* my_error_ptr;

void my_error_exit(j_common_ptr cinfo) {
  /* cinfo->err really points to a my_error_mgr struct, so coerce pointer */
  my_error_ptr myerr = (my_error_ptr)cinfo->err;

  /* Always display the message. */
  /* We could postpone this until after returning, if we chose. */
  (*cinfo->err->output_message)(cinfo);

  /* Return control to the setjmp point */
  longjmp(myerr->setjmp_buffer, 1);
}

void jpeg_decode(const uint8_t* bytes, uint64_t num_bytes, int64_t& width, int64_t& height, int64_t& channels) {
  struct jpeg_decompress_struct cinfo;
  my_error_mgr jerr;
  JSAMPARRAY buffer; /* Output row buffer */
  int row_stride;    /* physical row width in output buffer */

  /* Step 1: allocate and initialize JPEG decompression object */

  /* We set up the normal JPEG error routines, then override error_exit. */
  cinfo.err = jpeg_std_error(&jerr.pub);
  jerr.pub.error_exit = my_error_exit;

  /* Establish the setjmp return context for my_error_exit to use. */
  if (setjmp(jerr.setjmp_buffer)) {
    /* If we get here, the JPEG code has signaled an error.
     * We need to clean up the JPEG object, close the input file, and return.
     */
    jpeg_destroy_decompress(&cinfo);
  }

  /* Now we can initialize the JPEG decompression object. */
  jpeg_create_decompress(&cinfo);

  /* Step 2: specify data source (eg, a file) */

  jpeg_mem_src(&cinfo, bytes, num_bytes);

  /* Step 3: read file parameters with jpeg_read_header() */

  (void)jpeg_read_header(&cinfo, TRUE);
  /* We can ignore the return value from jpeg_read_header since
   *   (a) suspension is not possible with the stdio data source, and
   *   (b) we passed TRUE to reject a tables-only JPEG file as an error.
   * See libjpeg.txt for more info.
   */

  /* Step 4: set parameters for decompression */

  /* In this example, we don't need to change any of the defaults set by
   * jpeg_read_header(), so we do nothing here.
   */

  /* Step 5: Start decompressor */

  (void)jpeg_start_decompress(&cinfo);
  /* We can ignore the return value since suspension is not possible
   * with the stdio data source.
   */

  /* We may need to do some setup of our own at this point before reading
   * the data.  After jpeg_start_decompress() we have the correct scaled
   * output image dimensions available, as well as the output colormap
   * if we asked for color quantization.
   * In this example, we need to make an output work buffer of the right size.
   */
  /* JSAMPLEs per row in output buffer */
  row_stride = cinfo.output_width * cinfo.output_components;

  /* Make a one-row-high sample array that will go away when done with image */
  buffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE, row_stride, 1);

  /* Step 6: while (scan lines remain to be read) */
  /*           jpeg_read_scanlines(...); */

  /* Here we use the library's state variable cinfo.output_scanline as the
   * loop counter, so that we don't have to keep track ourselves.
   */
  while (cinfo.output_scanline < cinfo.output_height) {
    /* jpeg_read_scanlines expects an array of pointers to scanlines.
     * Here the array is only one element long, but you could ask for
     * more than one scanline at a time if that's more convenient.
     */
    (void)jpeg_read_scanlines(&cinfo, buffer, 1);
    /* Assume put_scanline_someplace wants a pointer and sample count. */
    // TODO: This needs to write the output.
  }

  /* Step 7: Finish decompression */

  (void)jpeg_finish_decompress(&cinfo);
  /* We can ignore the return value since suspension is not possible
   * with the stdio data source.
   */

  /* Step 8: Release JPEG decompression object */

  /* This is an important step since it will release a good deal of memory. */
  jpeg_destroy_decompress(&cinfo);
}
}  // namespace

DecodeImage::DecodeImage(const OpKernelInfo& info) : OpKernel(info) {
    pixel_format_ = info.GetAttrOrDefault<std::string>("pixel_format", "RGB");
}

void DecodeImage::Compute(OpKernelContext* context) {
  const Tensor* encoded_stream = context->Input<Tensor>(0);
  const uint8_t* encoded_stream_data = encoded_stream->Data<uint8_t>();
  const auto& dims = encoded_stream->Shape().GetDims();

  if (dims.size() != 1) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "Input is expected to have 1 dimension, got ", dims.size());
  }

  int width;
  int height;
  int channel;

  OrtTensorTypeAndShapeInfo* input_info = ort_.GetTensorTypeAndShape(inputs);
  const int64_t encoded_image_data_len = ort_.GetTensorShapeElementCount(input_info);
  ort_.ReleaseTensorTypeAndShapeInfo(input_info);

  const uint8_t* encoded_image_data = ort_.GetTensorData<uint8_t>(inputs);  // uint8 data

  if (PngDecoder::IsPng(encoded_image_data, encoded_image_data_len)) {
    auto decoder = PngDecoder(encoded_image_data, encoded_image_data_len);
    const auto& shape = decoder.Shape();
    OrtValue* output_value = ort_.KernelContext_GetOutput(context, 0, shape.data(), shape.size());
    uint8_t* decoded_image_data = ort_.GetTensorMutableData<uint8_t>(output_value);

    decoder.Decode(decoded_image_data, decoder.NumDecodedBytes());
  } else {
    jpeg_decode(
        ort_.GetTensorData<uint8_t>(inputs),
        encoded_image_data_len,
        width;
        height;
        channel);
  }

  // Decode the image
  // const std::vector<int32_t> encoded_image_sizes{1, static_cast<int32_t>(encoded_image_data_len)};
  // const void* encoded_image_data = ort_.GetTensorData<uint8_t>(inputs);  // uint8 data
  // const cv::Mat encoded_image(encoded_image_sizes, CV_8UC1, const_cast<void*>(encoded_image_data));
  // const cv::Mat decoded_image = cv::imdecode(encoded_image, cv::IMREAD_COLOR);

  // if (decoded_image.data == nullptr) {
  //   ORT_CXX_API_THROW("[DecodeImage] Invalid input. Failed to decode image.", ORT_INVALID_ARGUMENT);
  // };

  //// Setup output & copy to destination
  // const cv::Size decoded_image_size = decoded_image.size();
  // const int64_t colors = decoded_image.elemSize();  //  == 3 as it's BGR

  // const std::vector<int64_t> output_dims{decoded_image_size.height, decoded_image_size.width, colors};
  // OrtValue* output_value = ort_.KernelContext_GetOutput(context, 0, output_dims.data(), output_dims.size());
  // uint8_t* decoded_image_data = ort_.GetTensorMutableData<uint8_t>(output_value);
  // memcpy(decoded_image_data, decoded_image.data, decoded_image_size.height * decoded_image_size.width * colors);
}
}  // namespace ort_extensions

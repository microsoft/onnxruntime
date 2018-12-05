// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#include "core/session/onnxruntime_c_api.h"
#include "providers.h"
#include <stdio.h>
#include <assert.h>
#include <png.h>

#define ONNXRUNTIME_ABORT_ON_ERROR(expr)                         \
  do {                                                           \
    ONNXStatus* onnx_status = (expr);                            \
    if (onnx_status != NULL) {                                   \
      const char* msg = ONNXRuntimeGetErrorMessage(onnx_status); \
      fprintf(stderr, "%s\n", msg);                              \
      ReleaseONNXStatus(onnx_status);                            \
      abort();                                                   \
    }                                                            \
  } while (0);

/**
 * convert input from HWC format to CHW format
 * \param input A single image. The byte array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A float array. should be freed by caller after use
 * \param output_count Array length of the `output` param
 */
static void hwc_to_chw(const png_byte* input, size_t h, size_t w, float** output, size_t* output_count) {
  size_t stride = h * w;
  *output_count = stride * 3;
  float* output_data = (float*)malloc(*output_count * sizeof(float));
  for (size_t i = 0; i != stride; ++i) {
    for (size_t c = 0; c != 3; ++c) {
      output_data[c * stride + i] = input[i * 3 + c];
    }
  }
  *output = output_data;
}

/**
 * convert input from CHW format to HWC format
 * \param input A single image. This float array has length of 3*h*w
 * \param h image height
 * \param w image width
 * \param output A byte array. should be freed by caller after use
 */
static void chw_to_hwc(const float* input, size_t h, size_t w, png_bytep* output) {
  size_t stride = h * w;
  png_bytep output_data = (png_bytep)malloc(stride * 3);
  for (int c = 0; c != 3; ++c) {
    size_t t = c * stride;
    for (size_t i = 0; i != stride; ++i) {
      float f = input[t + i];
      if (f < 0.f || f > 255.0f)
        f = 0;
      output_data[i * 3 + c] = (png_byte)f;
    }
  }
  *output = output_data;
}

/**
 * \param out should be freed by caller after use
 * \param output_count Array length of the `out` param
 */
static int read_png_file(const char* input_file, size_t* height, size_t* width, float** out, size_t* output_count) {
  png_image image; /* The control structure used by libpng */
  /* Initialize the 'png_image' structure. */
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  if (png_image_begin_read_from_file(&image, input_file) == 0) {
    return -1;
  }
  png_bytep buffer;
  image.format = PNG_FORMAT_BGR;
  size_t input_data_length = PNG_IMAGE_SIZE(image);
  if (input_data_length != 720 * 720 * 3) {
    printf("input_data_length:%zd\n", input_data_length);
    return -1;
  }
  buffer = (png_bytep)malloc(input_data_length);
  memset(buffer, 0, input_data_length);
  if (png_image_finish_read(&image, NULL /*background*/, buffer,
                            0 /*row_stride*/, NULL /*colormap*/) == 0) {
    return -1;
  }
  hwc_to_chw(buffer, image.height, image.width, out, output_count);
  free(buffer);
  *width = image.width;
  *height = image.height;
  return 0;
}

/**
 * \param tensor should be a float tensor in [N,C,H,W] format
 */
static int write_tensor_to_png_file(ONNXValue* tensor, const char* output_file) {
  struct ONNXRuntimeTensorTypeAndShapeInfo* shape_info;
  ONNXRUNTIME_ABORT_ON_ERROR(ONNXRuntimeGetTensorShapeAndType(tensor, &shape_info));
  size_t dim_count = ONNXRuntimeGetNumOfDimensions(shape_info);
  if (dim_count != 4) {
    printf("output tensor must have 4 dimensions");
    return -1;
  }
  int64_t dims[4];
  ONNXRuntimeGetDimensions(shape_info, dims, sizeof(dims) / sizeof(dims[0]));
  if (dims[0] != 1 || dims[1] != 3) {
    printf("output tensor shape error");
    return -1;
  }
  float* f;
  ONNXRUNTIME_ABORT_ON_ERROR(ONNXRuntimeGetTensorMutableData(tensor, (void**)&f));
  png_bytep model_output_bytes;
  png_image image;
  memset(&image, 0, (sizeof image));
  image.version = PNG_IMAGE_VERSION;
  image.format = PNG_FORMAT_BGR;
  image.height = dims[2];
  image.width = dims[3];
  chw_to_hwc(f, image.height, image.width, &model_output_bytes);
  int ret = 0;
  if (png_image_write_to_file(&image, output_file, 0 /*convert_to_8bit*/,
                              model_output_bytes, 0 /*row_stride*/, NULL /*colormap*/) == 0) {
    printf("write to '%s' failed:%s\n", output_file, image.message);
    ret = -1;
  }
  free(model_output_bytes);
  return ret;
}

static void usage() {
  printf("usage: <model_path> <input_file> <output_file> \n");
}

int run_inference(ONNXSession* session, const char* input_file, const char* output_file) {
  size_t input_height;
  size_t input_width;
  float* model_input;
  size_t model_input_ele_count;
  if (read_png_file(input_file, &input_height, &input_width, &model_input, &model_input_ele_count) != 0) {
    return -1;
  }
  if (input_height != 720 || input_width != 720) {
    printf("please resize to image to 720x720\n");
    free(model_input);
    return -1;
  }
  ONNXRuntimeAllocatorInfo* allocator_info;
  ONNXRUNTIME_ABORT_ON_ERROR(ONNXRuntimeCreateCpuAllocatorInfo(ONNXRuntimeArenaAllocator, ONNXRuntimeMemTypeDefault, &allocator_info));
  const size_t input_shape[] = {1, 3, 720, 720};
  const size_t input_shape_len = sizeof(input_shape) / sizeof(input_shape[0]);
  const size_t model_input_len = model_input_ele_count * sizeof(float);

  ONNXValue* input_tensor = NULL;
  ONNXRUNTIME_ABORT_ON_ERROR(ONNXRuntimeCreateTensorWithDataAsONNXValue(allocator_info, model_input, model_input_len, input_shape, input_shape_len, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor));
  assert(input_tensor != NULL);
  assert(ONNXRuntimeIsTensor(input_tensor) != 0);
  ReleaseONNXRuntimeAllocatorInfo(allocator_info);
  const char* input_names[] = {"inputImage"};
  const char* output_names[] = {"outputImage"};
  ONNXValue* output_tensor = NULL;
  ONNXRUNTIME_ABORT_ON_ERROR(ONNXRuntimeRunInference(session, NULL, input_names, (const ONNXValue* const*)&input_tensor, 1, output_names, 1, &output_tensor));
  assert(output_tensor != NULL);
  assert(ONNXRuntimeIsTensor(output_tensor) != 0);
  int ret = 0;
  if (write_tensor_to_png_file(output_tensor, output_file) != 0) {
    ret = -1;
  }
  ReleaseONNXValue(output_tensor);
  ReleaseONNXValue(input_tensor);
  free(model_input);
  return ret;
}

void verify_input_output_count(ONNXSession* session) {
  size_t count;
  ONNXRUNTIME_ABORT_ON_ERROR(ONNXRuntimeInferenceSessionGetInputCount(session, &count));
  assert(count == 1);
  ONNXRUNTIME_ABORT_ON_ERROR(ONNXRuntimeInferenceSessionGetOutputCount(session, &count));
  assert(count == 1);
}

#ifdef USE_CUDA
void enable_cuda(ONNXRuntimeSessionOptions* session_option) {
  ONNXRuntimeProviderFactoryInterface** factory;
  ONNXRUNTIME_ABORT_ON_ERROR(ONNXRuntimeCreateCUDAExecutionProviderFactory(0, &factory));
  ONNXRuntimeSessionOptionsAppendExecutionProvider(session_option, factory);
  ONNXRuntimeReleaseObject(factory);
}
#endif

int main(int argc, char* argv[]) {
  if (argc < 4) {
    usage();
    return -1;
  }
  char* model_path = argv[1];
  char* input_file = argv[2];
  char* output_file = argv[3];
  ONNXRuntimeEnv* env;
  ONNXRUNTIME_ABORT_ON_ERROR(ONNXRuntimeInitialize(ONNXRUNTIME_LOGGING_LEVEL_kWARNING, "test", &env));
  ONNXRuntimeSessionOptions* session_option = ONNXRuntimeCreateSessionOptions();
#ifdef USE_CUDA
  enable_cuda(session_option);
#endif
  ONNXSession* session;
  ONNXRUNTIME_ABORT_ON_ERROR(ONNXRuntimeCreateInferenceSession(env, model_path, session_option, &session));
  verify_input_output_count(session);
  int ret = run_inference(session, input_file, output_file);
  ONNXRuntimeReleaseObject(session_option);
  ReleaseONNXSession(session);
  ONNXRuntimeReleaseObject(env);
  if (ret != 0) {
    fprintf(stderr, "fail\n");
  }
  return ret;
}

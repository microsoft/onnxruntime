#include "testPch.h"

#include "filehelpers.h"
#include "imageTestHelper.h"
#include "robuffer.h"

#include <d3dx12.h>
#include <MemoryBuffer.h>
#include <string>
#include <utility>

#ifndef BUILD_GOOGLE_TEST
#error Must use googletest for value-parameterized tests
#endif

using namespace winrt;
using namespace winml;
using namespace wfc;
using namespace wm;
using namespace wgi;
using namespace wgdx;
using namespace ws;
using namespace wss;

enum BindingLocation {
  CPU,
  GPU
};

class ImageTests : public ::testing::Test {
 protected:
  winml::LearningModel m_model = nullptr;
  winml::LearningModelDevice m_device = nullptr;
  winml::LearningModelSession m_session = nullptr;
  winml::LearningModelBinding m_model_binding = nullptr;
  winml::LearningModelEvaluationResult m_result = nullptr;

  static void SetUpTestSuite() {
    init_apartment();
#ifdef BUILD_INBOX
    winrt_activation_handler = WINRT_RoGetActivationFactory;
#endif
  }

  void LoadModel(const std::wstring& model_path) {
    std::wstring full_path = FileHelpers::GetModulePath() + model_path;
    WINML_EXPECT_NO_THROW(m_model = LearningModel::LoadFromFilePath(full_path));
  }

  void ImageTests::PrepareModelSessionBinding(
    const std::wstring& model_file_name,
    LearningModelDeviceKind device_kind,
    std::optional<uint32_t> optimized_batch_size
  ) {
    LoadModel(std::wstring(model_file_name));
    WINML_EXPECT_NO_THROW(m_device = LearningModelDevice(device_kind));
    if (optimized_batch_size.has_value()) {
      LearningModelSessionOptions options;
      options.BatchSizeOverride(optimized_batch_size.value());
      WINML_EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device, options));
    } else {
      WINML_EXPECT_NO_THROW(m_session = LearningModelSession(m_model, m_device));
    }
    WINML_EXPECT_NO_THROW(m_model_binding = LearningModelBinding(m_session));
  }

  bool BindInputValue(
    const std::wstring& image_file_name,
    const std::wstring& input_pixel_format,
    const std::wstring& model_pixel_format,
    InputImageSource input_image_source,
    LearningModelDeviceKind device_kind
  ) {
    std::wstring full_image_path = FileHelpers::GetModulePath() + image_file_name;
    StorageFile image_file = StorageFile::GetFileFromPathAsync(full_image_path).get();
    IRandomAccessStream stream = image_file.OpenAsync(FileAccessMode::Read).get();
    BitmapDecoder bitmap_decoder = BitmapDecoder::CreateAsync(stream).get();
    SoftwareBitmap software_bitmap = bitmap_decoder.GetSoftwareBitmapAsync().get();

    // Convert the input image to PixelFormat specified
    software_bitmap = SoftwareBitmap::Convert(software_bitmap, ImageTestHelper::GetPixelFormat(input_pixel_format));

    auto input_feature = m_model.InputFeatures().First();
    if (InputImageSource::FromImageFeatureValue == input_image_source) {
      VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(software_bitmap);
      ImageFeatureValue image_input_tensor = ImageFeatureValue::CreateFromVideoFrame(frame);
      WINML_EXPECT_NO_THROW(m_model_binding.Bind(input_feature.Current().Name(), image_input_tensor));
    } else if (InputImageSource::FromVideoFrame == input_image_source) {
      VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(software_bitmap);
      WINML_EXPECT_NO_THROW(m_model_binding.Bind(input_feature.Current().Name(), frame));
    } else if (InputImageSource::FromCPUResource == input_image_source) {
      TensorFloat tensor_float = ImageTestHelper::LoadInputImageFromCPU(software_bitmap, model_pixel_format);
      WINML_EXPECT_NO_THROW(m_model_binding.Bind(input_feature.Current().Name(), tensor_float));
    } else if (InputImageSource::FromGPUResource == input_image_source) {
      TensorFloat tensor_float = ImageTestHelper::LoadInputImageFromGPU(software_bitmap, model_pixel_format);
      if (LearningModelDeviceKind::Cpu == device_kind) {
        WINML_EXPECT_THROW_SPECIFIC(
          m_model_binding.Bind(input_feature.Current().Name(), tensor_float),
          winrt::hresult_error,
          [](const winrt::hresult_error& e) -> bool { return e.code() == WINML_ERR_INVALID_BINDING; }
        );
        return false;
      }
      WINML_EXPECT_NO_THROW(m_model_binding.Bind(input_feature.Current().Name(), tensor_float));
    }
    return true;
  }

  VideoFrame BindImageOutput(
    ModelInputOutputType model_input_output_type,
    OutputBindingStrategy output_binding_strategy,
    const std::wstring& model_pixel_format
  ) {
    std::wstring output_data_binding_name = std::wstring(m_model.OutputFeatures().First().Current().Name());
    VideoFrame frame = nullptr;
    if (OutputBindingStrategy::Bound == output_binding_strategy) {
      if (ModelInputOutputType::Image == model_input_output_type) {
        ImageFeatureDescriptor output_image_descriptor = nullptr;
        WINML_EXPECT_NO_THROW(m_model.OutputFeatures().First().Current().as(output_image_descriptor));
        SoftwareBitmap bitmap(
          ImageTestHelper::GetPixelFormat(model_pixel_format),
          output_image_descriptor.Height(),
          output_image_descriptor.Width()
        );
        frame = VideoFrame::CreateWithSoftwareBitmap(bitmap);

      } else if (ModelInputOutputType::Tensor == model_input_output_type) {
        TensorFeatureDescriptor output_tensor_descriptor = nullptr;
        WINML_EXPECT_NO_THROW(m_model.OutputFeatures().First().Current().as(output_tensor_descriptor));
        auto output_tensor_shape = output_tensor_descriptor.Shape();
        SoftwareBitmap bitmap(
          ImageTestHelper::GetPixelFormat(model_pixel_format),
          static_cast<int32_t>(output_tensor_shape.GetAt(3)),
          static_cast<int32_t>(output_tensor_shape.GetAt(2))
        );
        frame = VideoFrame::CreateWithSoftwareBitmap(bitmap);
      }
      auto output_tensor = ImageFeatureValue::CreateFromVideoFrame(frame);
      WINML_EXPECT_NO_THROW(m_model_binding.Bind(output_data_binding_name, output_tensor));
    }

    // Else for Unbound
    return frame;
  }

  IVector<VideoFrame> BindImageOutput(
    ModelInputOutputType model_input_output_type,
    OutputBindingStrategy output_binding_strategy,
    VideoFrameSource output_video_frame_source,
    const std::wstring& model_pixel_format,
    const uint32_t& batch_size
  ) {
    std::wstring output_data_binding_name = std::wstring(m_model.OutputFeatures().First().Current().Name());
    uint32_t width = 0, height = 0;
    if (ModelInputOutputType::Image == model_input_output_type) {
      ImageFeatureDescriptor output_image_descriptor = nullptr;
      WINML_EXPECT_NO_THROW(m_model.OutputFeatures().First().Current().as(output_image_descriptor));
      width = output_image_descriptor.Width();
      height = output_image_descriptor.Height();
    } else {
      TensorFeatureDescriptor output_tensor_descriptor = nullptr;
      WINML_EXPECT_NO_THROW(m_model.OutputFeatures().First().Current().as(output_tensor_descriptor));
      auto output_tensor_shape = output_tensor_descriptor.Shape();
      width = static_cast<uint32_t>(output_tensor_shape.GetAt(3));
      height = static_cast<uint32_t>(output_tensor_shape.GetAt(2));
    }
    IVector<VideoFrame> output_video_frames;
    if (OutputBindingStrategy::Bound == output_binding_strategy) {
      std::vector<VideoFrame> output_frames = {};
      for (uint32_t i = 0; i < batch_size; ++i) {
        VideoFrame video_frame = nullptr;
        if (VideoFrameSource::FromSoftwareBitmap == output_video_frame_source) {
          video_frame = VideoFrame::CreateWithSoftwareBitmap(
            SoftwareBitmap(ImageTestHelper::GetPixelFormat(model_pixel_format), width, height)
          );
        } else if (VideoFrameSource::FromDirect3DSurface == output_video_frame_source) {
          video_frame =
            VideoFrame::CreateAsDirect3D11SurfaceBacked(DirectXPixelFormat::B8G8R8A8UIntNormalized, width, height);
        } else if (VideoFrameSource::FromUnsupportedD3DSurface == output_video_frame_source) {
          video_frame =
            VideoFrame::CreateAsDirect3D11SurfaceBacked(DirectXPixelFormat::B8G8R8X8UIntNormalized, width, height);
        }
        output_frames.emplace_back(video_frame);
      }
      output_video_frames = winrt::single_threaded_vector(std::move(output_frames));
      WINML_EXPECT_NO_THROW(m_model_binding.Bind(output_data_binding_name, output_video_frames));
    }

    // Else for Unbound
    return output_video_frames;
  }

  void VerifyResults(
    VideoFrame output_tensor, const std::wstring& bm_image_path, const std::wstring& model_pixel_format
  ) {
    SoftwareBitmap bm_software_bitmap = FileHelpers::GetSoftwareBitmapFromFile(bm_image_path);
    bm_software_bitmap =
      SoftwareBitmap::Convert(bm_software_bitmap, ImageTestHelper::GetPixelFormat(model_pixel_format));
    VideoFrame bm_video_frame = VideoFrame::CreateWithSoftwareBitmap(bm_software_bitmap);
    WINML_EXPECT_TRUE(ImageTestHelper::VerifyHelper(output_tensor, bm_video_frame));
  }

  void EvaluateTest(EvaluationStrategy strategy) {
    if (EvaluationStrategy::Async == strategy) {
      WINML_EXPECT_NO_THROW(m_result = m_session.EvaluateAsync(m_model_binding, L"").get());
    } else if (EvaluationStrategy::Sync == strategy) {
      WINML_EXPECT_NO_THROW(m_result = m_session.Evaluate(m_model_binding, L""));
    }
  }

  bool ShouldSkip(
    const std::wstring& model_file_name, const std::wstring& image_file_name, const InputImageSource input_image_source
  ) {
    // Case that the tensor's shape doesn't match model's shape should be skiped
    if ((L"1080.jpg" == image_file_name || L"kitten_224.png" == image_file_name) && (InputImageSource::FromGPUResource == input_image_source || InputImageSource::FromCPUResource == input_image_source)) {
      return true;
    }

    // Case that the images's shape doesn't match model's shape which expects free dimension should be skiped.
    // Because the fns-candy is not real model that can handle free dimensional input
    if ((L"1080.jpg" == image_file_name || L"kitten_224.png" == image_file_name) && L"fns-candy_Bgr8_freeDimInput.onnx" == model_file_name) {
      return true;
    }

    return false;
  }

  void ValidateOutputImageMetaData(
    const std::wstring& path, BitmapAlphaMode expected_mode, BitmapPixelFormat expected_format, bool supported
  ) {
    WINML_EXPECT_NO_THROW(LoadModel(path));
    //input does not have image metadata and output does

    WINML_EXPECT_TRUE(m_model.OutputFeatures().First().HasCurrent());

    std::wstring name(m_model.OutputFeatures().First().Current().Name());
    std::wstring expected_tensor_name = L"add_3";
    WINML_EXPECT_EQUAL(name, expected_tensor_name);

    ImageFeatureDescriptor image_descriptor = nullptr;
    if (supported) {
      WINML_EXPECT_NO_THROW(m_model.OutputFeatures().First().Current().as(image_descriptor));
      WINML_EXPECT_TRUE(image_descriptor != nullptr);

      auto tensor_name = image_descriptor.Name();
      WINML_EXPECT_EQUAL(tensor_name, expected_tensor_name);

      auto model_data_kind = image_descriptor.Kind();
      WINML_EXPECT_EQUAL(model_data_kind, LearningModelFeatureKind::Image);

      WINML_EXPECT_TRUE(image_descriptor.IsRequired());

      WINML_EXPECT_EQUAL(image_descriptor.Width(), 1920u);
      WINML_EXPECT_EQUAL(image_descriptor.Height(), 1080u);
      WINML_EXPECT_EQUAL(image_descriptor.BitmapAlphaMode(), expected_mode);
      WINML_EXPECT_EQUAL(image_descriptor.BitmapPixelFormat(), expected_format);
    } else {
      //not an image descriptor. a regular tensor
      WINML_EXPECT_THROW_SPECIFIC(
        m_model.OutputFeatures().First().Current().as(image_descriptor),
        winrt::hresult_no_interface,
        [](const winrt::hresult_no_interface& e) -> bool { return e.code() == E_NOINTERFACE; }
      );
      TensorFeatureDescriptor tensor_descriptor = nullptr;
      WINML_EXPECT_NO_THROW(m_model.OutputFeatures().First().Current().as(tensor_descriptor));

      // Make sure we fail binding ImageFeatureValue
      LearningModelSession session(m_model);
      LearningModelBinding binding(session);
      auto ifv = FileHelpers::LoadImageFeatureValue(L"1080.jpg");
      WINML_EXPECT_THROW_SPECIFIC(
        binding.Bind(L"add_3", ifv),
        winrt::hresult_error,
        [](const winrt::hresult_error& e) -> bool { return e.code() == WINML_ERR_INVALID_BINDING; }
      );
    }
  }

  void RunConsecutiveImageBindingOnGpu(ImageFeatureValue& image1, ImageFeatureValue& image2) {
    static const wchar_t* model_file_name = L"Add_ImageNet1920.onnx";
    std::wstring module_path = FileHelpers::GetModulePath();

    // WinML model creation
    LearningModel model(nullptr);
    std::wstring full_model_path = module_path + model_file_name;
    WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(full_model_path));
    LearningModelDeviceKind device_kind = LearningModelDeviceKind::DirectX;
    LearningModelSession model_session(model, LearningModelDevice(device_kind));
    LearningModelBinding model_binding(model_session);

    //Input Binding
    auto feature = model.InputFeatures().First();
    WINML_EXPECT_NO_THROW(model_binding.Bind(feature.Current().Name(), image1));
    feature.MoveNext();
    WINML_EXPECT_NO_THROW(model_binding.Bind(feature.Current().Name(), image2));
  }

  static ImageFeatureValue CreateImageFeatureValue(const std::wstring& full_image_path) {
    StorageFile image_file = StorageFile::GetFileFromPathAsync(full_image_path).get();
    IRandomAccessStream stream = image_file.OpenAsync(FileAccessMode::Read).get();
    SoftwareBitmap software_bitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
    VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(software_bitmap);
    ImageFeatureValue image_input_tensor = ImageFeatureValue::CreateFromVideoFrame(frame);
    return image_input_tensor;
  }
};

// MNIST model expects 28x28 data with
class MnistImageTest : public ImageTests, public testing::WithParamInterface<std::pair<std::wstring, unsigned int>> {};
TEST_P(MnistImageTest, Evaluates) {
  GPUTEST;

  PrepareModelSessionBinding(L"mnist.onnx", LearningModelDeviceKind::Cpu, std::nullopt);

  std::wstring filename;
  unsigned int label;
  std::tie(filename, label) = GetParam();
  auto image_feature_value = FileHelpers::LoadImageFeatureValue(std::wstring(filename));
  m_model_binding.Bind(L"Input3", image_feature_value);

  auto result = m_session.EvaluateAsync(m_model_binding, L"0").get();
  auto vector = result.Outputs().Lookup(L"Plus214_Output_0").as<TensorFloat>().GetAsVectorView();

  unsigned int max_label = 0;
  float max_val = 0;
  for (unsigned int i = 0; i < vector.Size(); ++i) {
    float val = vector.GetAt(i);
    if (val > max_val) {
      max_val = val;
      max_label = i;
    }
  }
  std::cerr << "Expected Label " << label;
  std::cerr << "Evaluated Label " << max_label;
  WINML_EXPECT_TRUE(max_label == label);
}
INSTANTIATE_TEST_SUITE_P(
  MnistInputOutput,
  MnistImageTest,
  testing::Values(
    std::make_pair(L"vertical-crop.png", 5),
    std::make_pair(L"horizontal-crop.png", 2),
    std::make_pair(L"big.png", 8),
    std::make_pair(L"RGB_5.png", 5)
  )
);

#if defined(NDEBUG) || defined(RUN_MODELTEST_IN_DEBUG_MODE)
typedef std::tuple<
  std::tuple<std::wstring, ModelInputOutputType, std::wstring>,
  std::wstring,
  std::wstring,
  InputImageSource,
  EvaluationStrategy,
  OutputBindingStrategy,
  LearningModelDeviceKind>
  ImageTestParamTuple;
struct ImageTestParam {
  std::wstring model_file_name, model_pixel_format, image_file_name, input_pixel_format;
  ModelInputOutputType model_input_output_type;
  InputImageSource input_image_source;
  EvaluationStrategy evaluation_strategy;
  OutputBindingStrategy output_binding_strategy;
  LearningModelDeviceKind device_kind;

  ImageTestParam(ImageTestParamTuple param) {
    std::tuple<std::wstring, ModelInputOutputType, std::wstring> model_info;
    tie(
      model_info,
      image_file_name,
      input_pixel_format,
      input_image_source,
      evaluation_strategy,
      output_binding_strategy,
      device_kind
    ) = param;
    tie(model_file_name, model_input_output_type, model_pixel_format) = model_info;
  }
};
class ImageTest : public ImageTests, public testing::WithParamInterface<ImageTestParamTuple> {};

TEST_P(ImageTest, ImageTest) {
  const auto param = ImageTestParam(GetParam());
  if (ShouldSkip(param.model_file_name, param.image_file_name, param.input_image_source)) {
    GTEST_SKIP() << "This test is disabled";
  }

  if (LearningModelDeviceKind::Cpu != param.device_kind || InputImageSource::FromGPUResource == param.input_image_source) {
    GPUTEST;
  }

  PrepareModelSessionBinding(param.model_file_name, param.device_kind, {});

  bool toContinue = BindInputValue(
    param.image_file_name,
    param.input_pixel_format,
    param.model_pixel_format,
    param.input_image_source,
    param.device_kind
  );

  if (!toContinue)
    return;

  VideoFrame output_video_frame = BindImageOutput(
    param.model_input_output_type, param.output_binding_strategy, std::wstring(param.model_pixel_format)
  );

  EvaluateTest(param.evaluation_strategy);

  // benchmark used to compare with the output from model
  std::wstring benchmark_file_name =
    std::wstring(param.model_pixel_format + L'_' + param.input_pixel_format + L'_' + param.image_file_name);

  // Verify the output by comparing with the benchmark image
  std::wstring bm_image_path = FileHelpers::GetModulePath() + L"groundTruth\\" + benchmark_file_name;
  if (OutputBindingStrategy::Unbound == param.output_binding_strategy) {
    std::wstring output_data_binding_name = std::wstring(m_model.OutputFeatures().First().Current().Name());
    auto image_FV = m_result.Outputs().Lookup(output_data_binding_name).try_as<ImageFeatureValue>();
    if (image_FV == nullptr) {
      return;
    }
    output_video_frame = image_FV.VideoFrame();
  }
  VerifyResults(output_video_frame, bm_image_path, param.model_pixel_format);
}
INSTANTIATE_TEST_SUITE_P(
  ImageTest,
  ImageTest,
  testing::Combine(
    testing::Values(
      std::make_tuple(L"fns-candy_Bgr8.onnx", Image, L"Bgr8"),
      std::make_tuple(L"fns-candy_Rgb8.onnx", Image, L"Rgb8"),
      std::make_tuple(L"fns-candy_tensor.onnx", Tensor, L"Bgr8"),
      std::make_tuple(L"fns-candy_Bgr8_freeDimInput.onnx", Image, L"Bgr8")
    ),
    testing::Values(L"1080.jpg", L"kitten_224.png", L"fish_720.png", L"fish_720_Gray.png"),
    testing::Values(L"Bgra8", L"Rgba8", L"Gray8"),
    testing::Values(FromVideoFrame, FromImageFeatureValue, FromCPUResource, FromGPUResource),
    testing::Values(Async, Sync),
    testing::Values(Bound, Unbound),
    testing::Values(LearningModelDeviceKind::DirectX, LearningModelDeviceKind::Cpu)
  )
);

typedef std::tuple<
  std::tuple<std::wstring, ModelInputOutputType, std::vector<std::wstring>, int, bool>,
  OutputBindingStrategy,
  EvaluationStrategy,
  VideoFrameSource,
  VideoFrameSource,
  LearningModelDeviceKind>
  BatchTestParamTuple;
struct BatchTestParam {
  std::wstring model_file_name, model_pixel_format, image_file_name, input_pixel_format;
  ModelInputOutputType model_input_output_type;
  std::vector<std::wstring> input_images;
  int batch_size;
  bool use_session_options;
  OutputBindingStrategy output_binding_strategy;
  EvaluationStrategy evaluation_strategy;
  VideoFrameSource video_frame_source, output_video_frame_source;
  LearningModelDeviceKind device_kind;

  BatchTestParam(BatchTestParamTuple param) {
    std::tuple<std::wstring, ModelInputOutputType, std::vector<std::wstring>, int, bool> model_info;
    tie(
      model_info,
      output_binding_strategy,
      evaluation_strategy,
      video_frame_source,
      output_video_frame_source,
      device_kind
    ) = param;
    tie(model_file_name, model_input_output_type, input_images, batch_size, use_session_options) = model_info;
  }
};
class BatchTest : public ImageTests, public testing::WithParamInterface<BatchTestParamTuple> {};
TEST_P(BatchTest, BatchSupport) {
  const auto param = BatchTestParam(GetParam());
  std::optional<uint32_t> optimized_batch_size;
  if (param.use_session_options) {
    optimized_batch_size = param.use_session_options;
  }
  if (VideoFrameSource::FromDirect3DSurface == param.video_frame_source && LearningModelDeviceKind::Cpu == param.device_kind) {
    return;
  }
  if (LearningModelDeviceKind::Cpu != param.device_kind ||
        VideoFrameSource::FromDirect3DSurface == param.video_frame_source ||
        VideoFrameSource::FromDirect3DSurface == param.output_video_frame_source ||
        VideoFrameSource::FromUnsupportedD3DSurface == param.output_video_frame_source) {
    GPUTEST;
  }

  // create model, device and session
  PrepareModelSessionBinding(param.model_file_name, param.device_kind, optimized_batch_size);

  // create the input video_frames
  std::vector<VideoFrame> input_frames = {};
  if (param.input_images.empty()) {
    for (int i = 0; i < param.batch_size; ++i) {
      if (VideoFrameSource::FromDirect3DSurface == param.video_frame_source) {
        VideoFrame video_frame =
          VideoFrame::CreateAsDirect3D11SurfaceBacked(DirectXPixelFormat::B8G8R8X8UIntNormalized, 720, 720);
        input_frames.emplace_back(video_frame);
      } else {
        VideoFrame video_frame =
          VideoFrame::CreateWithSoftwareBitmap(SoftwareBitmap(BitmapPixelFormat::Bgra8, 720, 720));
        input_frames.emplace_back(video_frame);
      }
    }
  } else {
    for (int i = 0; i < param.batch_size; ++i) {
      std::wstring full_image_path = FileHelpers::GetModulePath() + param.input_images[i];
      StorageFile image_file = StorageFile::GetFileFromPathAsync(full_image_path).get();
      IRandomAccessStream stream = image_file.OpenAsync(FileAccessMode::Read).get();
      SoftwareBitmap software_bitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
      VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(software_bitmap);

      if (VideoFrameSource::FromDirect3DSurface == param.video_frame_source) {
        uint32_t width = software_bitmap.PixelWidth();
        uint32_t height = software_bitmap.PixelHeight();
        auto D3D_video_frame =
          VideoFrame::CreateAsDirect3D11SurfaceBacked(DirectXPixelFormat::B8G8R8X8UIntNormalized, width, height);
        frame.CopyToAsync(D3D_video_frame);
        input_frames.emplace_back(D3D_video_frame);
      } else {
        input_frames.emplace_back(frame);
      }
    }
  }
  auto video_frames = winrt::single_threaded_vector(std::move(input_frames));

  auto input_feature_descriptor = m_model.InputFeatures().First();
  WINML_EXPECT_NO_THROW(m_model_binding.Bind(input_feature_descriptor.Current().Name(), video_frames));

  auto output_video_frames = BindImageOutput(
    param.model_input_output_type,
    param.output_binding_strategy,
    param.output_video_frame_source,
    L"Bgra8",
    param.batch_size
  );

  EvaluateTest(param.evaluation_strategy);

  // benchmark used to compare with the output from model
  if (OutputBindingStrategy::Unbound == param.output_binding_strategy) {
    std::wstring output_data_binding_name = std::wstring(m_model.OutputFeatures().First().Current().Name());
    output_video_frames = m_result.Outputs().Lookup(output_data_binding_name).try_as<IVector<VideoFrame>>();
    if (output_video_frames == nullptr) {
      return;
    }
  }
  if (!param.input_images.empty()) {
    for (int i = 0; i < param.batch_size; ++i) {
      std::wstring bm_image_path = FileHelpers::GetModulePath() + L"batchGroundTruth\\" + param.input_images[i];
      if (VideoFrameSource::FromSoftwareBitmap != param.output_video_frame_source &&
                OutputBindingStrategy::Unbound != param.output_binding_strategy) {
        VideoFrame D3D_video_frame = output_video_frames.GetAt(i);
        VideoFrame SB_video_frame(BitmapPixelFormat::Bgra8, 720, 720);
        D3D_video_frame.as<IVideoFrame>().CopyToAsync(SB_video_frame).get();
        VerifyResults(SB_video_frame, bm_image_path, L"Bgra8");
      } else {
        VerifyResults(output_video_frames.GetAt(i), bm_image_path, L"Bgra8");
      }
    }
  }
}
// TODO: Reenable failing tests (Bug 299)
INSTANTIATE_TEST_SUITE_P(
  BatchTest,
  BatchTest,
  testing::Combine(
    testing::Values(
      std::make_tuple(
        L"fns-candy_Bgr8_Batch2.onnx", Image, std::vector<std::wstring>({L"fish_720.png", L"fish_720.png"}), 2, false
      ),
      std::make_tuple(
        L"fns-candy_Bgr8_Batch2.onnx", Image, std::vector<std::wstring>({L"1080.jpg", L"fish_720.png"}), 2, false
      ),
      std::make_tuple(
        L"fns-candy_Bgr8_Batch2.onnx",
        Image,
        std::vector<std::wstring>({L"fish_720_Gray.png", L"fish_720.png"}),
        2,
        false
      )
      // std::make_tuple(L"fns-candy_Bgr8_Batch3.onnx", Image, std::vector<std::wstring>({L"1080.jpg", L"fish_720_Gray.png", L"fish_720.png"}), 3, false),
      // std::make_tuple(L"fns-candy_Bgr8_Batch3.onnx", Image, std::vector<std::wstring>({L"1080.jpg", L"kitten_224.png", L"fish_720.png"}), 3, false),
      // std::make_tuple(L"fns-candy_Bgr8_tensor_Batch3.onnx", Tensor, std::vector<std::wstring>({L"1080.jpg", L"fish_720_Gray.png", L"fish_720.png"}), 3, false),
      // std::make_tuple(L"fns-candy_Bgr8_freeDimInput_Batch10.onnx", Image, std::vector<std::wstring>({}), 10, false),
      // std::make_tuple(L"fns-candy_Bgr8.onnx", Image, std::vector<std::wstring>({L"1080.jpg", L"fish_720_Gray.png", L"fish_720.png"}), 3, false),
      // std::make_tuple(L"fns-candy_Bgr8.onnx", Image, std::vector<std::wstring>({L"1080.jpg", L"fish_720_Gray.png", L"fish_720.png"}), 3, true)
    ),
    testing::Values(Bound, Unbound),
    testing::Values(Async, Sync),
    testing::Values(FromSoftwareBitmap, FromDirect3DSurface),
    testing::Values(FromSoftwareBitmap, FromDirect3DSurface, FromUnsupportedD3DSurface),
    testing::Values(LearningModelDeviceKind::DirectX, LearningModelDeviceKind::Cpu)
  )
);
#endif

TEST_F(ImageTests, LoadBindEvalModelWithoutImageMetadata) {
  GPUTEST;

  LoadModel(L"squeezenet_tensor_input.onnx");

  auto feature_value = FileHelpers::LoadImageFeatureValue(L"227x227.png");

  LearningModelSession model_session(m_model);
  LearningModelBinding model_binding(model_session);

  model_binding.Bind(L"data", feature_value);
  auto result = model_session.Evaluate(model_binding, L"");
}

TEST_F(ImageTests, LoadBindModelWithoutImageMetadata) {
  GPUTEST;

  // Model expecting a tensor instead of an image
  LoadModel(L"squeezenet_tensor_input.onnx");

  LearningModelSession model_session(m_model);
  LearningModelBinding model_binding(model_session);

  // Should work on images (by falling back to RGB8)
  auto feature_value = FileHelpers::LoadImageFeatureValue(L"227x227.png");
  model_binding.Bind(L"data", feature_value);

  // Should work on tensors
  auto tensor = TensorFloat::CreateFromIterable(
    {1, 3, 227, 227}, winrt::single_threaded_vector<float>(std::vector<float>(3 * 227 * 227))
  );
  model_binding.Bind(L"data", tensor);
}

TEST_F(ImageTests, LoadInvalidBindModelWithoutImageMetadata) {
  GPUTEST;

  LoadModel(L"squeezenet_tensor_input.onnx");

  LearningModelSession model_session(m_model);
  LearningModelBinding model_binding(model_session);

  // expect not fail if image dimensions are bigger than required
  auto feature_value = FileHelpers::LoadImageFeatureValue(L"1080.jpg");
  WINML_EXPECT_NO_THROW(model_binding.Bind(L"data", feature_value));

  // expect fail if tensor is of wrong type
  auto tensor_uint8 = TensorUInt8Bit::CreateFromIterable(
    {1, 3, 227, 227}, winrt::single_threaded_vector<uint8_t>(std::vector<uint8_t>(3 * 227 * 227))
  );
  WINML_EXPECT_THROW_SPECIFIC(
    model_binding.Bind(L"data", tensor_uint8),
    winrt::hresult_error,
    [](const winrt::hresult_error& e) -> bool { return e.code() == WINML_ERR_INVALID_BINDING; }
  );

  // Should fail if tensor has smaller dimensions/type
  auto tensor = TensorFloat::CreateFromIterable(
    {1, 3, 22, 22}, winrt::single_threaded_vector<float>(std::vector<float>(3 * 22 * 22))
  );
  WINML_EXPECT_THROW_SPECIFIC(
    model_binding.Bind(L"data", tensor),
    winrt::hresult_error,
    [](const winrt::hresult_error& e) -> bool { return e.code() == WINML_ERR_SIZE_MISMATCH; }
  );
}

TEST_F(ImageTests, ImageMetaDataTest) {
  // supported image metadata
  ValidateOutputImageMetaData(
    L"Add_ImageNet1920WithImageMetadataBgr8_SRGB_0_255.onnx",
    BitmapAlphaMode::Premultiplied,
    BitmapPixelFormat::Bgra8,
    true
  );
  ValidateOutputImageMetaData(
    L"Add_ImageNet1920WithImageMetadataRgb8_SRGB_0_255.onnx",
    BitmapAlphaMode::Premultiplied,
    BitmapPixelFormat::Rgba8,
    true
  );
  ValidateOutputImageMetaData(
    L"Add_ImageNet1920WithImageMetadataBgr8_SRGB_0_1.onnx",
    BitmapAlphaMode::Premultiplied,
    BitmapPixelFormat::Bgra8,
    true
  );
  ValidateOutputImageMetaData(
    L"Add_ImageNet1920WithImageMetadataBgr8_SRGB_1_1.onnx",
    BitmapAlphaMode::Premultiplied,
    BitmapPixelFormat::Bgra8,
    true
  );

  // unsupported image metadata
  ValidateOutputImageMetaData(
    L"Add_ImageNet1920WithImageMetadataBgra8_SRGB_0_255.onnx",
    BitmapAlphaMode::Straight,
    BitmapPixelFormat::Bgra8,
    false
  );
  ValidateOutputImageMetaData(
    L"Add_ImageNet1920WithImageMetadataRgba8_SRGB_0_255.onnx",
    BitmapAlphaMode::Straight,
    BitmapPixelFormat::Rgba8,
    false
  );
  ValidateOutputImageMetaData(
    L"Add_ImageNet1920WithImageMetadataBgr8_SRGB_16_235.onnx",
    BitmapAlphaMode::Straight,
    BitmapPixelFormat::Bgra8,
    false
  );
  ValidateOutputImageMetaData(
    L"Add_ImageNet1920WithImageMetadataBgr8_LINEAR_0_255.onnx",
    BitmapAlphaMode::Straight,
    BitmapPixelFormat::Bgra8,
    false
  );
}

//Tests if GPU will throw TDR if the same image feature value is binded back to back for two different inputs to a model
TEST_F(ImageTests, ImageBindingTwiceSameFeatureValueOnGpu) {
  GPUTEST;
  std::wstring module_path = FileHelpers::GetModulePath();
  static const wchar_t* input_data_image_filename = L"1080.jpg";

  std::wstring full_image_path = module_path + input_data_image_filename;
  ImageFeatureValue input_norm = CreateImageFeatureValue(full_image_path);

  RunConsecutiveImageBindingOnGpu(input_norm, input_norm);
}

//Tests if GPU will throw TDR if 2 different image feature values are binded back to back for two different inputs to a model
TEST_F(ImageTests, ImageBindingTwiceDifferentFeatureValueOnGpu) {
  GPUTEST;
  std::wstring module_path = FileHelpers::GetModulePath();
  static const wchar_t* input_data_image_filename = L"1080.jpg";

  std::wstring full_image_path = module_path + input_data_image_filename;
  ImageFeatureValue input_norm = CreateImageFeatureValue(full_image_path);
  ImageFeatureValue input_norm_1 = CreateImageFeatureValue(full_image_path);

  RunConsecutiveImageBindingOnGpu(input_norm, input_norm_1);
}

static void RunImageBindingInputAndOutput(bool bindInputAsIInspectable) {
  static const wchar_t* model_file_name = L"Add_ImageNet1920.onnx";
  std::wstring module_path = FileHelpers::GetModulePath();
  static const wchar_t* input_data_image_filename = L"1080.jpg";
  static const wchar_t* output_data_image_filename = L"out_Add_ImageNet_1080.jpg";

  // WinML model creation
  LearningModel model(nullptr);
  std::wstring full_model_path = module_path + model_file_name;
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(full_model_path));
  LearningModelDeviceKind device_kind = LearningModelDeviceKind::DirectX;
  LearningModelSession model_session(model, LearningModelDevice(device_kind));
  LearningModelBinding model_binding(model_session);

  std::wstring full_image_path = module_path + input_data_image_filename;

  StorageFile image_file = StorageFile::GetFileFromPathAsync(full_image_path).get();
  IRandomAccessStream stream = image_file.OpenAsync(FileAccessMode::Read).get();
  SoftwareBitmap software_bitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
  VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(software_bitmap);

  if (bindInputAsIInspectable) {
    auto feature = model.InputFeatures().First();
    WINML_EXPECT_NO_THROW(model_binding.Bind(feature.Current().Name(), frame));
    feature.MoveNext();
    WINML_EXPECT_NO_THROW(model_binding.Bind(feature.Current().Name(), frame));
  } else {
    ImageFeatureValue input_image_tensor = ImageFeatureValue::CreateFromVideoFrame(frame);
    auto feature = model.InputFeatures().First();
    WINML_EXPECT_NO_THROW(model_binding.Bind(feature.Current().Name(), input_image_tensor));
    feature.MoveNext();
    WINML_EXPECT_NO_THROW(model_binding.Bind(feature.Current().Name(), input_image_tensor));
  }

  auto output_tensor_descriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
  auto output_tensor_shape = output_tensor_descriptor.Shape();
  VideoFrame output_image(
    BitmapPixelFormat::Rgba8,
    static_cast<int32_t>(output_tensor_shape.GetAt(3)),
    static_cast<int32_t>(output_tensor_shape.GetAt(2))
  );
  ImageFeatureValue output_tensor = ImageFeatureValue::CreateFromVideoFrame(output_image);

  WINML_EXPECT_NO_THROW(model_binding.Bind(model.OutputFeatures().First().Current().Name(), output_tensor));

  // Evaluate the model
  winrt::hstring correlation_id;
  model_session.EvaluateAsync(model_binding, correlation_id).get();

  //check the output video frame object
  StorageFolder current_folder = StorageFolder::GetFolderFromPathAsync(module_path).get();
  StorageFile out_image_file =
    current_folder.CreateFileAsync(output_data_image_filename, CreationCollisionOption::ReplaceExisting).get();
  IRandomAccessStream write_stream = out_image_file.OpenAsync(FileAccessMode::ReadWrite).get();

  BitmapEncoder encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), write_stream).get();
  // Set the software bitmap
  encoder.SetSoftwareBitmap(output_image.SoftwareBitmap());

  encoder.FlushAsync().get();

  BYTE* data = nullptr;
  UINT32 ui_capacity = 0;
  wgi::BitmapBuffer bitmap_buffer(output_image.SoftwareBitmap().LockBuffer(wgi::BitmapBufferAccessMode::Read));
  wf::IMemoryBufferReference reference = bitmap_buffer.CreateReference();
  auto spByteAccess = reference.as<::Windows::Foundation::IMemoryBufferByteAccess>();
  WINML_EXPECT_HRESULT_SUCCEEDED(spByteAccess->GetBuffer(&data, &ui_capacity));
  WINML_EXPECT_NOT_EQUAL(data[0], 0);
}

TEST_F(ImageTests, ImageBindingInputAndOutput) {
  GPUTEST;
  RunImageBindingInputAndOutput(false /*bindInputAsIInspectable*/);
}

TEST_F(ImageTests, ImageBindingInputAndOutput_BindInputTensorAsInspectable) {
  GPUTEST;
  RunImageBindingInputAndOutput(true /*bindInputAsIInspectable*/);
}

static void TestImageBindingStyleTransfer(
  const wchar_t* model_file_name, const wchar_t* input_data_image_filename, wchar_t* output_data_image_filename
) {
  GPUTEST;

  //this test only checks that the operation completed successfully without crashing

  std::wstring module_path = FileHelpers::GetModulePath();

  // WinML model creation
  LearningModel model(nullptr);
  std::wstring full_model_path = module_path + model_file_name;
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(full_model_path));
  LearningModelDeviceKind device_kind = LearningModelDeviceKind::DirectX;
  LearningModelDevice device = nullptr;
  WINML_EXPECT_NO_THROW(device = LearningModelDevice(device_kind));
  LearningModelSession model_session = nullptr;
  WINML_EXPECT_NO_THROW(model_session = LearningModelSession(model, device));
  LearningModelBinding model_binding = nullptr;
  WINML_EXPECT_NO_THROW(model_binding = LearningModelBinding(model_session));

  std::wstring full_image_path = module_path + input_data_image_filename;

  StorageFile image_file = StorageFile::GetFileFromPathAsync(full_image_path).get();
  IRandomAccessStream stream = image_file.OpenAsync(FileAccessMode::Read).get();
  SoftwareBitmap software_bitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
  VideoFrame frame = VideoFrame::CreateWithSoftwareBitmap(software_bitmap);
  //aizBUG:3762 Cannot bind the same tensor to 2 different input. will deal with this in a later check in
  ImageFeatureValue input_1_image_tensor = ImageFeatureValue::CreateFromVideoFrame(frame);

  auto feature = model.InputFeatures().First();
  WINML_EXPECT_NO_THROW(model_binding.Bind(feature.Current().Name(), input_1_image_tensor));

  auto output_tensor_descriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
  auto output_tensor_shape = output_tensor_descriptor.Shape();
  VideoFrame output_image(
    BitmapPixelFormat::Rgba8,
    static_cast<int32_t>(output_tensor_shape.GetAt(3)),
    static_cast<int32_t>(output_tensor_shape.GetAt(2))
  );
  ImageFeatureValue output_tensor = ImageFeatureValue::CreateFromVideoFrame(output_image);

  WINML_EXPECT_NO_THROW(model_binding.Bind(model.OutputFeatures().First().Current().Name(), output_tensor));

  // Evaluate the model
  winrt::hstring correlation_id;
  WINML_EXPECT_NO_THROW(model_session.EvaluateAsync(model_binding, correlation_id).get());

  //check the output video frame object
  StorageFolder current_folder = StorageFolder::GetFolderFromPathAsync(module_path).get();
  StorageFile out_image_file =
    current_folder.CreateFileAsync(output_data_image_filename, CreationCollisionOption::ReplaceExisting).get();
  IRandomAccessStream write_stream = out_image_file.OpenAsync(FileAccessMode::ReadWrite).get();

  BitmapEncoder encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), write_stream).get();
  // Set the software bitmap
  encoder.SetSoftwareBitmap(output_image.SoftwareBitmap());

  encoder.FlushAsync().get();
}

TEST_F(ImageTests, ImageBindingStyleTransfer) {
  //this test only checks that the operation completed successfully without crashing
  TestImageBindingStyleTransfer(L"fns-candy.onnx", L"fish_720.png", L"out_fish_720_StyleTransfer.jpg");
}

TEST_F(ImageTests, ImageBindingAsGPUTensor) {
  GPUTEST;

  static const wchar_t* model_file_name = L"fns-candy.onnx";
  std::wstring module_path = FileHelpers::GetModulePath();
  static const wchar_t* input_data_image_filename = L"fish_720.png";
  static const wchar_t* output_data_image_filename = L"out_fish_720_StyleTransfer.jpg";

  // WinML model creation
  LearningModel model(nullptr);
  std::wstring full_model_path = module_path + model_file_name;
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(full_model_path));

  ID3D12Device* D3D12_device = nullptr;
  WINML_EXPECT_NO_THROW(D3D12CreateDevice(
    nullptr, D3D_FEATURE_LEVEL::D3D_FEATURE_LEVEL_11_0, __uuidof(ID3D12Device), reinterpret_cast<void**>(&D3D12_device)
  ));
  ID3D12CommandQueue* dx_queue = nullptr;
  D3D12_COMMAND_QUEUE_DESC command_queue_desc = {};
  command_queue_desc.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
  D3D12_device->CreateCommandQueue(
    &command_queue_desc, __uuidof(ID3D12CommandQueue), reinterpret_cast<void**>(&dx_queue)
  );
  auto device_factory = get_activation_factory<LearningModelDevice, ILearningModelDeviceFactoryNative>();
  auto tensor_factory = get_activation_factory<TensorFloat, ITensorStaticsNative>();

  com_ptr<::IUnknown> p_unk;
  device_factory->CreateFromD3D12CommandQueue(dx_queue, p_unk.put());

  LearningModelDevice dml_device_custom = nullptr;
  WINML_EXPECT_NO_THROW(p_unk.as(dml_device_custom));
  LearningModelSession dml_session_custom = nullptr;
  WINML_EXPECT_NO_THROW(dml_session_custom = LearningModelSession(model, dml_device_custom));

  LearningModelBinding model_binding(dml_session_custom);

  std::wstring full_image_path = module_path + input_data_image_filename;

  StorageFile image_file = StorageFile::GetFileFromPathAsync(full_image_path).get();
  IRandomAccessStream stream = image_file.OpenAsync(FileAccessMode::Read).get();
  SoftwareBitmap software_bitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();

  UINT64 buffer_byte_size =
    static_cast<uint64_t>(software_bitmap.PixelWidth()) * software_bitmap.PixelHeight() * 3 * sizeof(float);
  D3D12_HEAP_PROPERTIES heap_properties = {
    D3D12_HEAP_TYPE_DEFAULT, D3D12_CPU_PAGE_PROPERTY_UNKNOWN, D3D12_MEMORY_POOL_UNKNOWN, 0, 0};
  D3D12_RESOURCE_DESC resource_desc = {
    D3D12_RESOURCE_DIMENSION_BUFFER,
    0,
    buffer_byte_size,
    1,
    1,
    1,
    DXGI_FORMAT_UNKNOWN,
    {1, 0},
    D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS
  };

  com_ptr<ID3D12Resource> GPU_resource = nullptr;
  D3D12_device->CreateCommittedResource(
    &heap_properties,
    D3D12_HEAP_FLAG_NONE,
    &resource_desc,
    D3D12_RESOURCE_STATE_COMMON,
    nullptr,
    __uuidof(ID3D12Resource),
    GPU_resource.put_void()
  );
  com_ptr<::IUnknown> sp_unk_tensor;
  TensorFloat input_1_image_tensor(nullptr);
  __int64 shape[4] = {1, 3, software_bitmap.PixelWidth(), software_bitmap.PixelHeight()};
  tensor_factory->CreateFromD3D12Resource(GPU_resource.get(), shape, 4, sp_unk_tensor.put());
  sp_unk_tensor.try_as(input_1_image_tensor);

  auto feature = model.InputFeatures().First();
  WINML_EXPECT_NO_THROW(model_binding.Bind(feature.Current().Name(), input_1_image_tensor));

  auto output_tensor_descriptor = model.OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
  auto output_tensor_shape = output_tensor_descriptor.Shape();
  VideoFrame output_image(
    BitmapPixelFormat::Rgba8,
    static_cast<int32_t>(output_tensor_shape.GetAt(3)),
    static_cast<int32_t>(output_tensor_shape.GetAt(2))
  );
  ImageFeatureValue output_tensor = ImageFeatureValue::CreateFromVideoFrame(output_image);

  WINML_EXPECT_NO_THROW(model_binding.Bind(model.OutputFeatures().First().Current().Name(), output_tensor));

  // Evaluate the model
  winrt::hstring correlation_id;
  dml_session_custom.EvaluateAsync(model_binding, correlation_id).get();

  //check the output video frame object
  StorageFolder current_folder = StorageFolder::GetFolderFromPathAsync(module_path).get();
  StorageFile out_image_file =
    current_folder.CreateFileAsync(output_data_image_filename, CreationCollisionOption::ReplaceExisting).get();
  IRandomAccessStream write_stream = out_image_file.OpenAsync(FileAccessMode::ReadWrite).get();

  BitmapEncoder encoder = BitmapEncoder::CreateAsync(BitmapEncoder::JpegEncoderId(), write_stream).get();
  // Set the software bitmap
  encoder.SetSoftwareBitmap(output_image.SoftwareBitmap());

  encoder.FlushAsync().get();
}

static void GetCleanSession(
  LearningModelDeviceKind device_kind,
  std::wstring modelFilePath,
  LearningModelDevice& device,
  LearningModelSession& session
) {
  LearningModel model(nullptr);
  std::wstring full_model_path = modelFilePath;
  WINML_EXPECT_NO_THROW(model = LearningModel::LoadFromFilePath(full_model_path));
  WINML_EXPECT_NO_THROW(device = LearningModelDevice(device_kind));
  WINML_EXPECT_NO_THROW(session = LearningModelSession(model, device));
}

static void BindInputToSession(
  BindingLocation bind_location,
  std::wstring input_data_location,
  LearningModelSession& session,
  LearningModelBinding& binding
) {
  StorageFile image_file = StorageFile::GetFileFromPathAsync(input_data_location).get();
  IRandomAccessStream stream = image_file.OpenAsync(FileAccessMode::Read).get();
  SoftwareBitmap software_bitmap = (BitmapDecoder::CreateAsync(stream).get()).GetSoftwareBitmapAsync().get();
  VideoFrame cpu_video_frame = VideoFrame::CreateWithSoftwareBitmap(software_bitmap);
  if (bind_location == BindingLocation::CPU) {
    ImageFeatureValue input_image_feature_value = ImageFeatureValue::CreateFromVideoFrame(cpu_video_frame);
    WINML_EXPECT_NO_THROW(
      binding.Bind(session.Model().InputFeatures().First().Current().Name(), input_image_feature_value)
    );
  } else {
    DirectXPixelFormat format = DirectXPixelFormat::B8G8R8X8UIntNormalized;
    VideoFrame gpu_video_frame = VideoFrame::CreateAsDirect3D11SurfaceBacked(
      format, software_bitmap.PixelWidth(), software_bitmap.PixelHeight(), session.Device().Direct3D11Device()
    );
    cpu_video_frame.CopyToAsync(gpu_video_frame).get();
    ImageFeatureValue input_image_feature_value = ImageFeatureValue::CreateFromVideoFrame(gpu_video_frame);
    WINML_EXPECT_NO_THROW(
      binding.Bind(session.Model().InputFeatures().First().Current().Name(), input_image_feature_value)
    );
  }
}

static void BindOutputToSession(
  BindingLocation bind_location, LearningModelSession& session, LearningModelBinding& binding
) {
  auto output_tensor_descriptor = session.Model().OutputFeatures().First().Current().as<ITensorFeatureDescriptor>();
  auto output_tensor_shape = output_tensor_descriptor.Shape();
  if (bind_location == BindingLocation::CPU) {
    VideoFrame output_image(
      BitmapPixelFormat::Rgba8,
      static_cast<int32_t>(output_tensor_shape.GetAt(3)),
      static_cast<int32_t>(output_tensor_shape.GetAt(2))
    );
    ImageFeatureValue output_tensor = ImageFeatureValue::CreateFromVideoFrame(output_image);
    WINML_EXPECT_NO_THROW(binding.Bind(session.Model().OutputFeatures().First().Current().Name(), output_tensor));
  } else {
    VideoFrame output_image = VideoFrame::CreateAsDirect3D11SurfaceBacked(
      DirectXPixelFormat::B8G8R8X8UIntNormalized,
      static_cast<int32_t>(output_tensor_shape.GetAt(3)),
      static_cast<int32_t>(output_tensor_shape.GetAt(2))
    );
    ImageFeatureValue output_tensor = ImageFeatureValue::CreateFromVideoFrame(output_image);
    WINML_EXPECT_NO_THROW(binding.Bind(session.Model().OutputFeatures().First().Current().Name(), output_tensor));
  }
}

static void SynchronizeGPUWorkloads(const wchar_t* model_file_name, const wchar_t* input_data_image_filename) {
  //this test only checks that the operations complete successfully without crashing
  GPUTEST;
  std::wstring module_path = FileHelpers::GetModulePath();
  LearningModelDevice device = nullptr;
  LearningModelSession session = nullptr;
  LearningModelBinding binding = nullptr;

  /*
     * lazy dx11 loading scenarios:
     */

  // Scenario 1
  GetCleanSession(LearningModelDeviceKind::DirectX, module_path + model_file_name, device, session);

  // input: CPU, output: CPU
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  BindInputToSession(BindingLocation::CPU, module_path + input_data_image_filename, session, binding);
  BindOutputToSession(BindingLocation::CPU, session, binding);
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));

  // ---> verify that 11 stack is not initialized
  WINML_EXPECT_FALSE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

  // input: CPU, output: GPU
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  BindInputToSession(BindingLocation::CPU, module_path + input_data_image_filename, session, binding);
  BindOutputToSession(BindingLocation::GPU, session, binding);
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));

  // Scenario 2
  GetCleanSession(LearningModelDeviceKind::DirectX, module_path + model_file_name, device, session);

  // input: CPU, output: CPU
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  BindInputToSession(BindingLocation::CPU, module_path + input_data_image_filename, session, binding);
  BindOutputToSession(BindingLocation::CPU, session, binding);
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));

  // ---> verify that 11 stack is not initialized
  WINML_EXPECT_FALSE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

  // input: GPU, output: CPU
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  BindInputToSession(BindingLocation::GPU, module_path + input_data_image_filename, session, binding);
  BindOutputToSession(BindingLocation::CPU, session, binding);
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));

  // ---> verify that 11 stack is initialized
  WINML_EXPECT_TRUE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

  // Scenario 3
  GetCleanSession(LearningModelDeviceKind::DirectX, module_path + model_file_name, device, session);

  // input: CPU, output: CPU
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  BindInputToSession(BindingLocation::CPU, module_path + input_data_image_filename, session, binding);
  BindOutputToSession(BindingLocation::CPU, session, binding);
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));

  // ---> verify that 11 stack is not initialized
  WINML_EXPECT_FALSE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

  // input: GPU, output: GPU
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  BindInputToSession(BindingLocation::GPU, module_path + input_data_image_filename, session, binding);
  BindOutputToSession(BindingLocation::GPU, session, binding);
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));

  // ---> verify that 11 stack is initialized
  WINML_EXPECT_TRUE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

  /*
     * non lazy dx11 loading scenarios:
     */

  // Scenario 1
  GetCleanSession(LearningModelDeviceKind::DirectX, module_path + model_file_name, device, session);

  // input: GPU, output: CPU
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  BindInputToSession(BindingLocation::GPU, module_path + input_data_image_filename, session, binding);
  BindOutputToSession(BindingLocation::CPU, session, binding);
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));

  // ---> verify that 11 stack is initialized
  WINML_EXPECT_TRUE(device.as<IDeviceFenceValidator>()->SharedHandleInitialized());

  // input : CPU, output : CPU
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  BindInputToSession(BindingLocation::CPU, module_path + input_data_image_filename, session, binding);
  BindOutputToSession(BindingLocation::CPU, session, binding);
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));

  // Scenario 2
  GetCleanSession(LearningModelDeviceKind::DirectX, module_path + model_file_name, device, session);

  // input: CPU, output: GPU
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  BindInputToSession(BindingLocation::CPU, module_path + input_data_image_filename, session, binding);
  BindOutputToSession(BindingLocation::GPU, session, binding);
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));

  // input : CPU, output : CPU
  WINML_EXPECT_NO_THROW(binding = LearningModelBinding(session));
  BindInputToSession(BindingLocation::CPU, module_path + input_data_image_filename, session, binding);
  BindOutputToSession(BindingLocation::CPU, session, binding);
  WINML_EXPECT_NO_THROW(session.Evaluate(binding, L""));
}

TEST_F(ImageTests, SynchronizeGPUWorkloads) {
  SynchronizeGPUWorkloads(L"fns-candy.onnx", L"fish_720.png");
}

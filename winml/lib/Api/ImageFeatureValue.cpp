// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "pch.h"
#include "ImageFeatureValue.h"
#include "LearningModelBinding.h"
#include "LearningModelDevice.h"
#include "LearningModelSession.h"
#include <windows.media.h>
#include <wrl\wrappers\corewrappers.h>
#include "LearningModelBinding.h"
#include "LearningModelSession.h"
#include "LearningModelDevice.h"
#include "ImageConversionTypes.h"
#include "ConverterResourceStore.h"
#include "ImageFeatureDescriptor.h"

#include "core/session/onnxruntime_c_api.h"

#include "D3DDeviceCache.h"
#include "TensorFeatureDescriptor.h"

using namespace WinML;
using namespace winrt::Windows::Graphics::Imaging;
using namespace winrt::Windows::Graphics::DirectX::Direct3D11;
using namespace winrt::Windows::Graphics::DirectX;
using namespace Windows::AI::MachineLearning::Internal;
using namespace winrt::Windows::Foundation::Collections;

namespace winrt::Windows::AI::MachineLearning::implementation {

struct ImageFeatureValue::ImageResourceMetadata {
  std::vector<Windows::Graphics::Imaging::BitmapBounds> Bounds;
  ::Windows::AI::MachineLearning::Internal::ImageTensorDescription TensorDescriptor;
};

Windows::AI::MachineLearning::ImageFeatureValue ImageFeatureValue::Create(
    uint32_t batchSize,
    BitmapPixelFormat format,
    uint32_t width,
    uint32_t height) {
  std::vector<Windows::Media::VideoFrame> videoFrames = {};
  for (uint32_t i = 0; i < batchSize; ++i) {
    SoftwareBitmap bitmap(format, width, height);
    Windows::Media::VideoFrame frame = Windows::Media::VideoFrame::CreateWithSoftwareBitmap(bitmap);
    videoFrames.emplace_back(frame);
  }
  return make<ImageFeatureValue>(winrt::single_threaded_vector(std::move(videoFrames)));
}

Windows::AI::MachineLearning::ImageFeatureValue ImageFeatureValue::CreateFromVideoFrame(Windows::Media::VideoFrame const& image) try {
  return make<ImageFeatureValue>(image);
}
WINML_CATCH_ALL

void ImageFeatureValue::Initialize() {
  m_batchSize = m_videoFrames.Size();
  for (auto videoFrame : m_videoFrames) {
    // TODO: Check all videoFrames come from either CPU or GPU.
    if (auto surface = videoFrame.Direct3DSurface()) {
      Direct3DSurfaceDescription description = surface.Description();
      m_widths.emplace_back(description.Width);
      m_heights.emplace_back(description.Height);
    } else {
      ISoftwareBitmap softwarebitmap(videoFrame.SoftwareBitmap());
      m_widths.emplace_back(softwarebitmap.PixelWidth());
      m_heights.emplace_back(softwarebitmap.PixelHeight());
    }
  }
}

ImageFeatureValue::ImageFeatureValue(Windows::Media::VideoFrame const& image) {
  std::vector<Windows::Media::VideoFrame> frame = {image};
  m_videoFrames = winrt::single_threaded_vector(std::move(frame));
  Initialize();
}

ImageFeatureValue::ImageFeatureValue(IVector<Windows::Media::VideoFrame> const& images) : m_videoFrames(images) {
  Initialize();
}

ImageFeatureValue::ImageFeatureValue(IVectorView<Windows::Media::VideoFrame> const& images) {
  std::vector<Windows::Media::VideoFrame> videoFrames = {};
  for (uint32_t i = 0; i < images.Size(); ++i) {
    videoFrames.emplace_back(images.GetAt(i));
  }
  m_videoFrames = winrt::single_threaded_vector(std::move(videoFrames));
  Initialize();
}

static std::optional<BitmapPixelFormat> GetBitmapPixelFormatFromMetadata(const IPropertySet& properties) {
  if (properties != nullptr && properties.HasKey(L"BitmapPixelFormat")) {
    if (auto pixelFormatInspectable = properties.Lookup(L"BitmapPixelFormat")) {
      auto pixelFormatValue = pixelFormatInspectable.as<Windows::Foundation::IPropertyValue>();
      auto pixelFormat = static_cast<BitmapPixelFormat>(pixelFormatValue.GetInt32());
      WINML_THROW_HR_IF_FALSE_MSG(
          WINML_ERR_INVALID_BINDING,
          pixelFormat == BitmapPixelFormat::Rgba8 ||
              pixelFormat == BitmapPixelFormat::Bgra8 ||
              pixelFormat == BitmapPixelFormat::Gray8,
          "BitmapPixelFormat must be either Rgba8, Bgra8, or Gray8");

      return pixelFormat;
    }
  }

  return {};
}

static std::optional<BitmapBounds> GetBoundsFromMetadata(const IPropertySet& properties) {
  if (properties != nullptr && properties.HasKey(L"BitmapBounds")) {
    if (auto boundsInspectable = properties.Lookup(L"BitmapBounds")) {
      auto boundsPropertyValue = boundsInspectable.as<Windows::Foundation::IPropertyValue>();
      WINML_THROW_HR_IF_FALSE_MSG(
          WINML_ERR_INVALID_BINDING,
          boundsPropertyValue.Type() == Windows::Foundation::PropertyType::UInt32Array,
          "BitmapBounds must reference a property value with type UInt32Array with 4 elements.");

      com_array<uint32_t> bounds;
      boundsPropertyValue.GetUInt32Array(bounds);
      WINML_THROW_HR_IF_FALSE_MSG(
          WINML_ERR_INVALID_BINDING,
          bounds.size() == 4,
          "BitmapBounds must reference a property value with type UInt32Array with 4 elements.");

      return Windows::Graphics::Imaging::BitmapBounds{bounds[0], bounds[1], bounds[2], bounds[3]};
    }
  }

  return {};
}

BitmapBounds ImageFeatureValue::CenterAndCropBounds(
    uint32_t idx,
    uint32_t desiredWidth,
    uint32_t desiredHeight) {
  BitmapBounds bounds = {};
  float RequiredAspectRatio = static_cast<float>(desiredWidth) / static_cast<float>(desiredHeight);

  // crop to center while maintaining size
  if (RequiredAspectRatio * m_heights[idx] < m_widths[idx]) {
    // actual width is too wide. Cut off left and right of image
    bounds.Width = std::min((UINT)(RequiredAspectRatio * m_heights[idx] + 0.5f), m_widths[idx]);
    bounds.Height = m_heights[idx];
    bounds.X = (m_widths[idx] - bounds.Width) / 2;
    bounds.Y = 0;
  } else {
    // actual height is too long. Cut off top and bottom
    bounds.Width = m_widths[idx];
    bounds.Height = std::min((UINT)(m_widths[idx] / RequiredAspectRatio + 0.5f), m_heights[idx]);
    bounds.X = 0;
    bounds.Y = (m_heights[idx] - bounds.Height) / 2;
  }

  // TODO: Do we allow smaller images?
  WINML_THROW_HR_IF_FALSE_MSG(
      WINML_ERR_INVALID_BINDING,
      (bounds.X >= 0 && bounds.X <= m_widths[idx]) &&
          (bounds.Y >= 0 && bounds.Y <= m_heights[idx]),
      "Failed to center crop the provided input image. The calculated bounds exceed the dimensions of the image, or do not match the model inputs dimensions.");

  return bounds;
}

static ImageTensorDataType GetTensorDataTypeFromTensorKind(TensorKind kind) {
  switch (kind) {
    case TensorKind::Float:
      return kImageTensorDataTypeFloat32;
    case TensorKind::Float16:
      return kImageTensorDataTypeFloat16;
    default:
      WINML_THROW_HR_IF_FALSE_MSG(WINML_ERR_INVALID_BINDING, false, "Model image inputs must have tensor type of Float or Float16.");
  }

  FAIL_FAST_HR(E_INVALIDARG);
}

static unsigned GetSizeFromTensorDataType(ImageTensorDataType type) {
  switch (type) {
    case kImageTensorDataTypeFloat32:
      return sizeof(float);
    case kImageTensorDataTypeFloat16:
      return sizeof(uint16_t);
    default:
      WINML_THROW_HR_IF_FALSE_MSG(WINML_ERR_INVALID_BINDING, false, "Model image inputs must have tensor type of Float or Float16.");
  }

  FAIL_FAST_HR(E_INVALIDARG);
}

static ImageTensorDescription CreateImageTensorDescriptor(TensorKind tensorKind, BitmapPixelFormat pixelFormat, uint32_t batchSize, uint32_t width, uint32_t height) {
  ImageTensorDescription tensorDescription = {};
  tensorDescription.dataType = GetTensorDataTypeFromTensorKind(tensorKind);
  tensorDescription.sizes[0] = batchSize;

  if (pixelFormat == Windows::Graphics::Imaging::BitmapPixelFormat::Rgba8) {
    tensorDescription.channelType = kImageTensorChannelTypeRGB8;
    tensorDescription.sizes[1] = 3;
  } else if (pixelFormat == Windows::Graphics::Imaging::BitmapPixelFormat::Bgra8) {
    tensorDescription.channelType = kImageTensorChannelTypeBGR8;
    tensorDescription.sizes[1] = 3;
  } else if (pixelFormat == Windows::Graphics::Imaging::BitmapPixelFormat::Gray8) {
    tensorDescription.channelType = kImageTensorChannelTypeGRAY8;
    tensorDescription.sizes[1] = 1;
  } else {
    THROW_HR(E_NOTIMPL);
  }
  tensorDescription.sizes[2] = height;
  tensorDescription.sizes[3] = width;

  return tensorDescription;
}

static void CPUTensorize(
    Windows::Media::IVideoFrame videoFrame,
    BitmapBounds bounds,
    ImageTensorDescription tensorDescriptor,
    com_ptr<LearningModelSession> spSession,
    void* pResource) {
  auto spDevice = spSession->Device().as<LearningModelDevice>();

  ConverterResourceDescription descriptor = {};
  descriptor.pixel_format = static_cast<DWORD>(BitmapPixelFormat::Bgra8);
  descriptor.width = static_cast<int>(tensorDescriptor.sizes[3]);
  descriptor.height = static_cast<int>(tensorDescriptor.sizes[2]);
  descriptor.luid = {};  // Converted image on CPU

  auto pooledConverter = PoolObjectWrapper::Create(spDevice->TensorizerStore()->Fetch(descriptor));

  //apply tensorization
  pooledConverter->Get()->Tensorizer->VideoFrameToSoftwareTensor(
      videoFrame,
      bounds,
      tensorDescriptor,
      reinterpret_cast<BYTE*>(pResource));

  // Software tensorization doesnt need to hold onto any resources beyond its scope, so we can
  // return the converter to the pool on tensorization completion.
  // (This happens automatically in the destruction of PoolObjectWrapper)
}

static void CPUTensorize(
    IVector<Windows::Media::VideoFrame> videoFrames,
    std::vector<BitmapBounds> bounds,
    ImageTensorDescription tensorDescriptor,
    com_ptr<LearningModelSession> spSession,
    BYTE* resource,
    unsigned int singleFrameBufferSize) {
  // Tensorize video frames one by one without extra copy.
  for (uint32_t batchIdx = 0; batchIdx < videoFrames.Size(); ++batchIdx) {
    CPUTensorize(videoFrames.GetAt(batchIdx), bounds[batchIdx], tensorDescriptor, spSession, resource);
    resource += singleFrameBufferSize;
  }
}

static void GPUTensorize(
    IVector<Windows::Media::VideoFrame> videoFrames,
    std::vector<BitmapBounds> bounds,
    ImageTensorDescription tensorDescriptor,
    com_ptr<LearningModelSession> spSession,
    ID3D12Resource* d3dResource,
    WinML::BindingContext& context) {
  auto spDevice = spSession->Device().as<LearningModelDevice>();

  ConverterResourceDescription descriptor = {};
  descriptor.pixel_format = static_cast<DWORD>(DirectXPixelFormat::B8G8R8X8UIntNormalized);
  descriptor.width = static_cast<int>(tensorDescriptor.sizes[3]);
  descriptor.height = static_cast<int>(tensorDescriptor.sizes[2]);
  descriptor.luid = spDevice->GetD3DDevice()->GetAdapterLuid();  // Converted image on GPU

  // Tensorize video frames one by one without extra copy.
  for (uint32_t batchIdx = 0; batchIdx < videoFrames.Size(); ++batchIdx) {
    auto pooledConverter = PoolObjectWrapper::Create(spDevice->TensorizerStore()->Fetch(descriptor));
    {
      // Apply tensorization
      auto session = spSession.as<winrt::Windows::AI::MachineLearning::LearningModelSession>();
      pooledConverter->Get()->Tensorizer->VideoFrameToDX12Tensor(
          batchIdx,
          session,
          videoFrames.GetAt(batchIdx),
          bounds[batchIdx],
          tensorDescriptor,
          d3dResource);

      // Tensorization to a GPU tensor will run asynchronously and associated resources
      // need to be kept alive until the gpu resources have been used in the queue.
      //
      // The PoolObjectWrapper needs to stay alive so that the underlying resources are
      // not released to the cache.
      //
      // This object will be returned to the cache when evaluate has completed. So we cache this
      // on the binding context.
      context.converter = pooledConverter;
    }
  }
}

std::optional<ImageFeatureValue::ImageResourceMetadata> ImageFeatureValue::GetInputMetadata(const WinML::BindingContext& context) {
  uint32_t descriptorWidth;
  uint32_t descriptorHeight;

  TensorKind tensorKind = TensorKind::Undefined;
  auto spImageDescriptor = context.descriptor.try_as<ImageFeatureDescriptor>();
  auto spTensorDescriptor = context.descriptor.try_as<TensorFeatureDescriptor>();

  // Set up descriptorWidth and descriptorHeight
  if (spImageDescriptor) {
    // If model expects free dimensions the descritpr will have MAXUINT32, and we use the supplied image

    // If the width or height in model metadata is -1, which means free dimension.
    // The the widths and heights of input data must be the same. Or the
    // tensorDescriptor cannot describ the shape of the inputs.
    if (spImageDescriptor->Width() == MAXUINT32 &&
        !(std::adjacent_find(m_widths.begin(), m_widths.end(), std::not_equal_to<uint32_t>()) == m_widths.end())) {
      THROW_HR(E_INVALIDARG);
    }
    if (spImageDescriptor->Height() == MAXUINT32 &&
        !(std::adjacent_find(m_heights.begin(), m_heights.end(), std::not_equal_to<uint32_t>()) == m_heights.end())) {
      THROW_HR(E_INVALIDARG);
    }
    descriptorWidth = (spImageDescriptor->Width() == MAXUINT32) ? m_widths[0] : spImageDescriptor->Width();
    descriptorHeight = (spImageDescriptor->Height() == MAXUINT32) ? m_heights[0] : spImageDescriptor->Height();
    tensorKind = spImageDescriptor->TensorKind();
  } else if (spTensorDescriptor) {
    // If model expects a tensor, use its shape
    auto shape = spTensorDescriptor->Shape();

    if (shape.Size() != 4) {
      return {};
    }
    bool hasAccecptableChannelSize = (shape.GetAt(1) == 3 || shape.GetAt(1) == 1);
    if (!hasAccecptableChannelSize) {
      return {};
    }
    if (-1 == shape.GetAt(3) &&
        !(std::adjacent_find(m_widths.begin(), m_widths.end(), std::not_equal_to<uint32_t>()) == m_widths.end())) {
      THROW_HR(E_INVALIDARG);
    }
    if (-1 == shape.GetAt(2) &&
        !(std::adjacent_find(m_heights.begin(), m_heights.end(), std::not_equal_to<uint32_t>()) == m_heights.end())) {
      THROW_HR(E_INVALIDARG);
    }
    descriptorWidth = (-1 == shape.GetAt(3)) ? m_widths[0] : static_cast<uint32_t>(shape.GetAt(3));
    descriptorHeight = (-1 == shape.GetAt(2)) ? m_heights[0] : static_cast<uint32_t>(shape.GetAt(2));
    tensorKind = spTensorDescriptor->TensorKind();
  } else {
    return {};
  }

  // Set up BitmapBounds
  // For batch of images with different sizes, like { {1, 3, 1080, 1080}, {1, 3, 720, 720} },
  // a vector of bounds is to record the result after cropped.
  std::vector<BitmapBounds> bounds = {};
  for (uint32_t i = 0; i < m_batchSize; ++i) {
    auto tempBounds = GetBoundsFromMetadata(context.properties);
    if (!tempBounds.has_value()) {
      // If the user has not specified bounds, we need to infer the bounds
      // from the combination of descriptor, and input value or output value
      if (context.type == BindingType::kInput) {
        // If unspecified output, get the crop with correct aspect ratio
        tempBounds = CenterAndCropBounds(i, descriptorWidth, descriptorHeight);
      } else {
        // If given an unspecified output region, write into the top left portion of the output image.
        tempBounds = BitmapBounds{0, 0, m_widths[i], m_heights[i]};
      }
    }
    bounds.emplace_back(tempBounds.value());
  }
  // TODO: Validate Bounds

  // Set up BitmapPixelFormat

  auto pixelFormat = std::optional<BitmapPixelFormat>{};
  pixelFormat = GetBitmapPixelFormatFromMetadata(context.properties);
  if (!pixelFormat.has_value() && spImageDescriptor) {
    pixelFormat = spImageDescriptor->BitmapPixelFormat();
  } else if (!pixelFormat.has_value() && spTensorDescriptor) {
    auto shape = spTensorDescriptor->Shape();
    int channelCount = static_cast<uint32_t>(shape.GetAt(1));
    if (channelCount == 1) {
      // Assume Gray if no image descriptor is given and channelcount 1
      pixelFormat = BitmapPixelFormat::Gray8;

    } else if (channelCount == 3) {
      // Assume Bgra8 if no image descriptor is given
      pixelFormat = BitmapPixelFormat::Bgra8;
    } else {
      THROW_HR(WINML_ERR_SIZE_MISMATCH);
    }
  }
  //NCHW layout
  auto imageTensorDescriptor = CreateImageTensorDescriptor(tensorKind, pixelFormat.value(), m_batchSize, descriptorWidth, descriptorHeight);

  return ImageResourceMetadata{bounds, imageTensorDescriptor};
}

HRESULT ImageFeatureValue::GetValue(WinML::BindingContext& context, IValue** out) try {
  FAIL_FAST_IF(!(std::all_of(m_widths.begin(), m_widths.end(), [](int i) { return i != 0; })));
  FAIL_FAST_IF(!(std::all_of(m_heights.begin(), m_heights.end(), [](int i) { return i != 0; })));

  // Get image metadata from the binding context
  auto metadata = GetInputMetadata(context);
  RETURN_HR_IF(E_INVALIDARG, !metadata);
  ImageResourceMetadata resourceMetadata = metadata.value();

  // Get the session
  auto spSession = context.session.as<LearningModelSession>();
  auto spDevice = spSession->Device().as<LearningModelDevice>();
  auto engine = spSession->GetEngine();

  // create the OrtValue
  winrt::com_ptr<IValue> value;
  RETURN_IF_FAILED(engine->CreateTensorValue(
      resourceMetadata.TensorDescriptor.sizes,
      sizeof(resourceMetadata.TensorDescriptor.sizes) / sizeof(resourceMetadata.TensorDescriptor.sizes[0]),
      resourceMetadata.TensorDescriptor.dataType == kImageTensorDataTypeFloat32 ? winml::TensorKind::Float : winml::TensorKind::Float16,
      value.put()));

  // Get the tensor raw data
  WinML::Resource void_resource;
  RETURN_IF_FAILED(value->GetResource(void_resource));

  if (context.type == BindingType::kInput) {
    // Only tensorize inputs
    auto bufferSize = std::accumulate(std::begin(resourceMetadata.TensorDescriptor.sizes), std::end(resourceMetadata.TensorDescriptor.sizes), static_cast<int64_t>(1), std::multiplies<int64_t>());
    auto bufferByteSize = GetSizeFromTensorDataType(resourceMetadata.TensorDescriptor.dataType) * bufferSize;
    auto singleFrameBufferSize = bufferByteSize / m_batchSize;
    if (spDevice->IsCpuDevice()) {
      auto resource = reinterpret_cast<BYTE*>(void_resource.get());
      CPUTensorize(m_videoFrames, resourceMetadata.Bounds, resourceMetadata.TensorDescriptor, spSession, resource, static_cast<unsigned int>(singleFrameBufferSize));
    } else {
      auto resource = reinterpret_cast<ID3D12Resource*>(void_resource.get());
      GPUTensorize(m_videoFrames, resourceMetadata.Bounds, resourceMetadata.TensorDescriptor, spSession, resource, context);
    }
  }

  *out = value.detach();
  return S_OK;
}
WINML_CATCH_ALL_COM

HRESULT ImageFeatureValue::IsPlaceholder(bool* pIsPlaceHolder) {
  FAIL_FAST_IF_NULL(pIsPlaceHolder);
  *pIsPlaceHolder = false;
  return S_OK;
}

HRESULT ImageFeatureValue::UpdateSourceResourceData(BindingContext& context, IValue* value) try {
  // Get the device
  auto spSession = context.session.as<LearningModelSession>();
  auto spDevice = spSession->Device().as<LearningModelDevice>();

  // Get the output tensor raw data
  WinML::Resource void_resource;
  RETURN_IF_FAILED(value->GetResource(void_resource));

  // Get the run context
  auto metadata = GetInputMetadata(context);
  ImageResourceMetadata resourceMetadata = metadata.value();

  ConverterResourceDescription descriptor = {};
  descriptor.width = static_cast<int>(resourceMetadata.TensorDescriptor.sizes[3]);
  descriptor.height = static_cast<int>(resourceMetadata.TensorDescriptor.sizes[2]);

  bool out;
  if (SUCCEEDED(value->IsCpu(&out)) && out) {
    descriptor.pixel_format = static_cast<DWORD>(BitmapPixelFormat::Bgra8);
    descriptor.luid = {};  // Converted image on CPU

    auto pooledConverter = PoolObjectWrapper::Create(spDevice->DetensorizerStore()->Fetch(descriptor));

    auto bufferSize = std::accumulate(std::begin(resourceMetadata.TensorDescriptor.sizes), std::end(resourceMetadata.TensorDescriptor.sizes), static_cast<int64_t>(1), std::multiplies<int64_t>());
    auto bufferByteSize = GetSizeFromTensorDataType(resourceMetadata.TensorDescriptor.dataType) * bufferSize / m_batchSize;

    BYTE* resource = reinterpret_cast<BYTE*>(void_resource.get());
    for (uint32_t batchIdx = 0; batchIdx < m_batchSize; ++batchIdx) {
      // Convert Software Tensor to VideoFrame one by one based on the buffer size.
      auto videoFrame = m_videoFrames.GetAt(batchIdx);
      pooledConverter->Get()->Detensorizer->SoftwareTensorToVideoFrame(context.session, resource, resourceMetadata.TensorDescriptor, videoFrame);
      resource += bufferByteSize;
    }
  } else {
    descriptor.pixel_format = static_cast<DWORD>(DirectXPixelFormat::B8G8R8X8UIntNormalized);
    descriptor.luid = spDevice->GetD3DDevice()->GetAdapterLuid();  // Converted image on GPU

    auto pooledConverter = PoolObjectWrapper::Create(spDevice->DetensorizerStore()->Fetch(descriptor));

    auto d3dResource = reinterpret_cast<ID3D12Resource*>(void_resource.get());

    for (uint32_t batchIdx = 0; batchIdx < m_batchSize; ++batchIdx) {
      auto videoFrame = m_videoFrames.GetAt(batchIdx);
      pooledConverter->Get()->Detensorizer->DX12TensorToVideoFrame(
          batchIdx,
          context.session,
          d3dResource,
          resourceMetadata.TensorDescriptor,
          videoFrame);

      // Reset the Allocator before return to the Cache. Must Sync this background thread to that completion before we do.
      spDevice->GetD3DDeviceCache()->SyncD3D12ToCPU();
      pooledConverter->Get()->Detensorizer->ResetAllocator();
    }
  }

  // Release any converters back to the pool by nulling out the wrapper.
  context.converter = nullptr;
  return S_OK;
}
WINML_CATCH_ALL_COM

HRESULT ImageFeatureValue::AbiRepresentation(winrt::Windows::Foundation::IInspectable& abiRepresentation) {
  if (IsBatch()) {
    m_videoFrames.as(abiRepresentation);
  } else {
    winrt::Windows::AI::MachineLearning::ImageFeatureValue to = nullptr;
    RETURN_IF_FAILED(this->QueryInterface(
        winrt::guid_of<winrt::Windows::AI::MachineLearning::ImageFeatureValue>(),
        reinterpret_cast<void**>(winrt::put_abi(to))));

    to.as(abiRepresentation);
  }
  return S_OK;
}

Windows::AI::MachineLearning::LearningModelFeatureKind ImageFeatureValue::Kind() try {
  return LearningModelFeatureKind::Image;
}
WINML_CATCH_ALL

Windows::Media::VideoFrame ImageFeatureValue::VideoFrame() try {
  return m_videoFrames.GetAt(0);
}
WINML_CATCH_ALL

IIterable<Windows::Media::VideoFrame> ImageFeatureValue::VideoFrames() try {
  return m_videoFrames.try_as<IIterable<Windows::Media::VideoFrame>>();
}
WINML_CATCH_ALL
}  // namespace winrt::Windows::AI::MachineLearning::implementation

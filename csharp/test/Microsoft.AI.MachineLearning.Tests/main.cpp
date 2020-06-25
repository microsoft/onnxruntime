#include "winrt/microsoft.ai.machinelearning.h"
#include "winrt/windows.storage.h"
#include "winrt/windows.foundation.h"
#include "winrt/windows.foundation.collections.h"
#include "winrt/Windows.Graphics.h"
#include "winrt/Windows.Graphics.Imaging.h"
#include "winrt/Windows.Media.h"

#include <windows.h>

EXTERN_C IMAGE_DOS_HEADER __ImageBase;

using namespace winrt::Microsoft::AI::MachineLearning;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::Media;
using namespace winrt::Windows::Graphics::Imaging;

std::wstring GetModulePath() {
  std::wstring val;
  wchar_t modulePath[MAX_PATH] = {0};
  GetModuleFileNameW((HINSTANCE)&__ImageBase, modulePath, _countof(modulePath));
  wchar_t drive[_MAX_DRIVE];
  wchar_t dir[_MAX_DIR];
  wchar_t filename[_MAX_FNAME];
  wchar_t ext[_MAX_EXT];
  _wsplitpath_s(modulePath, drive, _MAX_DRIVE, dir, _MAX_DIR, filename, _MAX_FNAME, ext, _MAX_EXT);

  val = drive;
  val += dir;

  return val;
}

int main() {
  auto model = LearningModel::LoadFromFilePath(L"squeezenet.onnx");
  auto name = GetModulePath() + L"kitten_224.png";
  auto image = StorageFile::GetFileFromPathAsync(name).get();
  auto stream = image.OpenAsync(FileAccessMode::Read).get();
  auto softwareBitmap = BitmapDecoder::CreateAsync(stream).get().GetSoftwareBitmapAsync().get();
  auto frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);

  auto session = LearningModelSession(model);
  auto binding = LearningModelBinding(session);
  binding.Bind(L"data_0", frame);
  auto results = session.Evaluate(binding, L"");
}
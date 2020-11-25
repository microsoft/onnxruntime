#include <stdio.h>
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
  printf("Load squeezenet.onnx.\n");
  auto model = LearningModel::LoadFromFilePath(L"squeezenet.onnx");
  printf("Load kitten_224.png as StorageFile.\n");
  auto name = GetModulePath() + L"kitten_224.png";
  auto image = StorageFile::GetFileFromPathAsync(name).get();
  printf("Load StorageFile into Stream.\n");
  auto stream = image.OpenAsync(FileAccessMode::Read).get();
  printf("Create SoftwareBitmap from decoded Stream.\n");
  auto softwareBitmap = BitmapDecoder::CreateAsync(stream).get().GetSoftwareBitmapAsync().get();
  printf("Create VideoFrame.\n");
  auto frame = VideoFrame::CreateWithSoftwareBitmap(softwareBitmap);
  printf("Create LearningModelSession.\n");
  auto session = LearningModelSession(model);
  printf("Create LearningModelBinding.\n");
  auto binding = LearningModelBinding(session);
  printf("Bind data_0.\n");
  binding.Bind(L"data_0", frame);
  printf("Evaluate.\n");
  auto results = session.Evaluate(binding, L"");
  printf("Success!\n");
  return 0;
}
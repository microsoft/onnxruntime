// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#define UNICODE
#include <windows.h>
#include <onnxruntime_cxx_api.h>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "onnxruntime.lib")

Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "test"};

// This is the structure to interface with the NMIST model
// After instantiation, set the input_image_ data to be the 28x28 pixel image of the number to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
struct NMIST {
  NMIST() {
    auto allocator_info = Ort::AllocatorInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(allocator_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
  }

  int Run() {
    const char* input_names[] = {"Input3"};
    const char* output_names[] = {"Plus214_Output_0"};

    session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);

    result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
    return result_;
  }

  static constexpr const int width_ = 28;
  static constexpr const int height_ = 28;

  std::array<float, width_ * height_> input_image_{};
  std::array<float, 10> results_{};
  int result_{0};

 private:
  Ort::Session session_{env, L"mnist\\model.onnx", Ort::SessionOptions{nullptr}};

  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 4> input_shape_{1, 1, width_, height_};

  Ort::Value output_tensor_{nullptr};
  std::array<int64_t, 2> output_shape_{1, 10};
};

const constexpr unsigned scale_{1};
const constexpr unsigned scale_ui_{4};

NMIST nmist_;
HBITMAP dib_;
HDC hdc_dib_;
bool painting_{};

struct DIBInfo : DIBSECTION {
  DIBInfo(HBITMAP hBitmap) noexcept { ::GetObject(hBitmap, sizeof(DIBSECTION), this); }

  int Width() const noexcept { return dsBm.bmWidth; }
  int Height() const noexcept { return dsBm.bmHeight; }

  void* Bits() const noexcept { return dsBm.bmBits; }
  int Pitch() const noexcept { return dsBmih.biSizeImage / abs(dsBmih.biHeight); }
};

// We need to convert the true-color data in the DIB into the model's floating point format
// TODO: (also scales down the image and smooths the values, but this is not working properly)
void ConvertDibToNmist() {
  DIBInfo info{dib_};

  const DWORD* input = reinterpret_cast<const DWORD*>(info.Bits());
  float* output = nmist_.input_image_.data();

  std::fill(nmist_.input_image_.begin(), nmist_.input_image_.end(), 0.f);

  for (unsigned y = 0; y < NMIST::height_; y++) {
    for (unsigned yblock = 0; yblock < scale_; yblock++) {
      for (unsigned x = 0; x < NMIST::width_; x++) {
        for (unsigned xblock = 0; xblock < scale_; xblock++)
          output[x] += input[x * scale_ + xblock] == 0 ? 1.0f : 0.0f;
      }
      input = reinterpret_cast<const DWORD*>(reinterpret_cast<const BYTE*>(input) + info.Pitch());
    }
    output += NMIST::width_;
  }

  // Normalize the resulting sums
  for (auto& v : nmist_.input_image_)
    v /= scale_ * scale_;
}

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// The Windows entry point function
int APIENTRY wWinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPTSTR lpCmdLine, int nCmdShow) {
  {
    WNDCLASSEX wc{};
    wc.cbSize = sizeof(WNDCLASSEX);
    wc.style = CS_HREDRAW | CS_VREDRAW;
    wc.lpfnWndProc = WndProc;
    wc.hInstance = hInstance;
    wc.hCursor = LoadCursor(NULL, IDC_ARROW);
    wc.hbrBackground = (HBRUSH)(COLOR_WINDOW + 1);
    wc.lpszClassName = L"ONNXTest";
    RegisterClassEx(&wc);
  }
  {
    BITMAPINFO bmi{};
    bmi.bmiHeader.biSize = sizeof(bmi.bmiHeader);
    bmi.bmiHeader.biWidth = NMIST::width_ * scale_;
    bmi.bmiHeader.biHeight = -NMIST::height_ * scale_;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits;
    dib_ = CreateDIBSection(nullptr, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
  }

  hdc_dib_ = CreateCompatibleDC(nullptr);
  SelectObject(hdc_dib_, dib_);
  SelectObject(hdc_dib_, CreatePen(PS_SOLID, 2, RGB(0, 0, 0)));
  FillRect(hdc_dib_, &RECT{0, 0, NMIST::width_, NMIST::height_}, (HBRUSH)GetStockObject(WHITE_BRUSH));

  HWND hWnd = CreateWindow(L"ONNXTest", L"ONNX Runtime Sample - NMIST", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 512, 256, nullptr, nullptr, hInstance, nullptr);
  if (!hWnd)
    return FALSE;

  ShowWindow(hWnd, nCmdShow);

  MSG msg;
  while (GetMessage(&msg, NULL, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }
  return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
  switch (message) {
    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC hdc = BeginPaint(hWnd, &ps);

      // Draw the image
      StretchBlt(hdc, 0, 0, NMIST::width_ * scale_ui_, NMIST::height_ * scale_ui_, hdc_dib_, 0, 0, NMIST::width_, NMIST::height_, SRCCOPY);
      SelectObject(hdc, GetStockObject(BLACK_PEN));
      MoveToEx(hdc, NMIST::width_ * scale_ui_, 0, nullptr);
      LineTo(hdc, NMIST::width_ * scale_ui_, NMIST::height_ * scale_ui_);
      LineTo(hdc, 0, NMIST::height_ * scale_ui_);

      constexpr int graphs_left = NMIST::width_ * scale_ui_ + 5;
      constexpr int graph_width = 64;
      SelectObject(hdc, GetStockObject(GRAY_BRUSH));

      auto least = *std::min_element(nmist_.results_.begin(), nmist_.results_.end());
      auto greatest = nmist_.results_[nmist_.result_];
      auto range = greatest - least;

      auto graphs_zero = graphs_left - least * graph_width / range;

      // Hilight the winner
      RECT rc{graphs_left, nmist_.result_ * 16, graphs_left + graph_width + 128, (nmist_.result_ + 1) * 16};
      FillRect(hdc, &rc, (HBRUSH)GetStockObject(LTGRAY_BRUSH));

      // For every entry, draw the odds and the graph for it
      SetBkMode(hdc, TRANSPARENT);
      wchar_t value[80];
      for (unsigned i = 0; i < 10; i++) {
        int y = 16 * i;
        float result = nmist_.results_[i];

        auto length = wsprintf(value, L"%2d: %d.%02d", i, int(result), abs(int(result * 100) % 100));
        TextOut(hdc, graphs_left + graph_width + 5, y, value, length);

        Rectangle(hdc, graphs_zero, y + 1, graphs_zero + result * graph_width / range, y + 14);
      }

      // Draw the zero line
      MoveToEx(hdc, graphs_zero, 0, nullptr);
      LineTo(hdc, graphs_zero, 16 * 10);

      EndPaint(hWnd, &ps);
      return 0;
    }

    case WM_LBUTTONDOWN: {
      SetCapture(hWnd);
      painting_ = true;
      int x = LOWORD(lParam);
      int y = HIWORD(lParam);
      MoveToEx(hdc_dib_, x / scale_ui_, y / scale_ui_, nullptr);
      return 0;
    }

    case WM_MOUSEMOVE:
      if (painting_) {
        int x = LOWORD(lParam);
        int y = HIWORD(lParam);
        LineTo(hdc_dib_, x / scale_ui_, y / scale_ui_);
        InvalidateRect(hWnd, nullptr, false);
      }
      return 0;

    case WM_CAPTURECHANGED:
      painting_ = false;
      return 0;

    case WM_LBUTTONUP:
      ReleaseCapture();
      ConvertDibToNmist();
      nmist_.Run();
      InvalidateRect(hWnd, nullptr, true);
      return 0;

    case WM_RBUTTONDOWN:  // Erase the image
      FillRect(hdc_dib_, &RECT{0, 0, NMIST::width_ * scale_, NMIST::height_ * scale_}, (HBRUSH)GetStockObject(WHITE_BRUSH));
      InvalidateRect(hWnd, nullptr, false);
      return 0;

    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}

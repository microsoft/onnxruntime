// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#define UNICODE
#include <windows.h>
#include <windowsx.h>
#include <onnxruntime_cxx_api.h>
#include <array>
#include <cmath>
#include <algorithm>

#pragma comment(lib, "user32.lib")
#pragma comment(lib, "gdi32.lib")
#pragma comment(lib, "onnxruntime.lib")

template <typename T>
static void softmax(T& input) {
  float rowmax = *std::max_element(input.begin(), input.end());
  std::vector<float> y(input.size());
  float sum = 0.0f;
  for (size_t i = 0; i != input.size(); ++i) {
    sum += y[i] = std::exp(input[i] - rowmax);
  }
  for (size_t i = 0; i != input.size(); ++i) {
    input[i] = y[i] / sum;
  }
}

// This is the structure to interface with the MNIST model
// After instantiation, set the input_image_ data to be the 28x28 pixel image of the number to recognize
// Then call Run() to fill in the results_ data with the probabilities of each
// result_ holds the index with highest probability (aka the number the model thinks is in the image)
struct MNIST {
  MNIST() {
    auto memory_info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
    input_tensor_ = Ort::Value::CreateTensor<float>(memory_info, input_image_.data(), input_image_.size(), input_shape_.data(), input_shape_.size());
    output_tensor_ = Ort::Value::CreateTensor<float>(memory_info, results_.data(), results_.size(), output_shape_.data(), output_shape_.size());
  }

  std::ptrdiff_t Run() {
    const char* input_names[] = {"Input3"};
    const char* output_names[] = {"Plus214_Output_0"};

    session_.Run(Ort::RunOptions{nullptr}, input_names, &input_tensor_, 1, output_names, &output_tensor_, 1);
    softmax(results_);
    result_ = std::distance(results_.begin(), std::max_element(results_.begin(), results_.end()));
    return result_;
  }

  static constexpr const int width_ = 28;
  static constexpr const int height_ = 28;

  std::array<float, width_ * height_> input_image_{};
  std::array<float, 10> results_{};
  int64_t result_{0};

 private:
  Ort::Env env;
  Ort::Session session_{env, L"model.onnx", Ort::SessionOptions{nullptr}};

  Ort::Value input_tensor_{nullptr};
  std::array<int64_t, 4> input_shape_{1, 1, width_, height_};

  Ort::Value output_tensor_{nullptr};
  std::array<int64_t, 2> output_shape_{1, 10};
};

const constexpr int drawing_area_inset_{4};  // Number of pixels to inset the top left of the drawing area
const constexpr int drawing_area_scale_{4};  // Number of times larger to make the drawing area compared to the shape inputs
const constexpr int drawing_area_width_{MNIST::width_ * drawing_area_scale_};
const constexpr int drawing_area_height_{MNIST::height_ * drawing_area_scale_};

std::unique_ptr<MNIST> mnist_;
HBITMAP dib_;
HDC hdc_dib_;
bool painting_{};

HBRUSH brush_winner_{CreateSolidBrush(RGB(128, 255, 128))};
HBRUSH brush_bars_{CreateSolidBrush(RGB(128, 128, 255))};

struct DIBInfo : DIBSECTION {
  DIBInfo(HBITMAP hBitmap) noexcept { ::GetObject(hBitmap, sizeof(DIBSECTION), this); }

  int Width() const noexcept { return dsBm.bmWidth; }
  int Height() const noexcept { return dsBm.bmHeight; }

  void* Bits() const noexcept { return dsBm.bmBits; }
  int Pitch() const noexcept { return dsBmih.biSizeImage / abs(dsBmih.biHeight); }
};

// We need to convert the true-color data in the DIB into the model's floating point format
// TODO: (also scales down the image and smooths the values, but this is not working properly)
void ConvertDibToMnist() {
  DIBInfo info{dib_};

  const DWORD* input = reinterpret_cast<const DWORD*>(info.Bits());
  float* output = mnist_->input_image_.data();

  std::fill(mnist_->input_image_.begin(), mnist_->input_image_.end(), 0.f);

  for (unsigned y = 0; y < MNIST::height_; y++) {
    for (unsigned x = 0; x < MNIST::width_; x++) {
      output[x] += input[x] == 0 ? 1.0f : 0.0f;
    }
    input = reinterpret_cast<const DWORD*>(reinterpret_cast<const BYTE*>(input) + info.Pitch());
    output += MNIST::width_;
  }
}

LRESULT CALLBACK WndProc(HWND, UINT, WPARAM, LPARAM);

// The Windows entry point function
int APIENTRY wWinMain(_In_ HINSTANCE hInstance, _In_opt_ HINSTANCE /*hPrevInstance*/, _In_ LPTSTR /*lpCmdLine*/,
                      _In_ int nCmdShow) {
  try {
    mnist_ = std::make_unique<MNIST>();
  } catch (const Ort::Exception& exception) {
    MessageBoxA(nullptr, exception.what(), "Error:", MB_OK);
    return 0;
  }

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
    bmi.bmiHeader.biWidth = MNIST::width_;
    bmi.bmiHeader.biHeight = -MNIST::height_;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biBitCount = 32;
    bmi.bmiHeader.biPlanes = 1;
    bmi.bmiHeader.biCompression = BI_RGB;

    void* bits;
    dib_ = CreateDIBSection(nullptr, &bmi, DIB_RGB_COLORS, &bits, nullptr, 0);
  }
  if (dib_ == nullptr) return -1;
  hdc_dib_ = CreateCompatibleDC(nullptr);
  SelectObject(hdc_dib_, dib_);
  SelectObject(hdc_dib_, CreatePen(PS_SOLID, 2, RGB(0, 0, 0)));
  RECT rect{0, 0, MNIST::width_, MNIST::height_};
  FillRect(hdc_dib_, &rect, (HBRUSH)GetStockObject(WHITE_BRUSH));

  HWND hWnd = CreateWindow(L"ONNXTest", L"ONNX Runtime Sample - MNIST", WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT, 512, 256, nullptr, nullptr, hInstance, nullptr);
  if (!hWnd)
    return FALSE;

  ShowWindow(hWnd, nCmdShow);

  MSG msg;
  while (GetMessage(&msg, NULL, 0, 0)) {
    TranslateMessage(&msg);
    DispatchMessage(&msg);
  }

  DeleteObject(dib_);
  DeleteDC(hdc_dib_);

  DeleteObject(brush_winner_);
  DeleteObject(brush_bars_);

  return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT message, WPARAM wParam, LPARAM lParam) {
  switch (message) {
    case WM_PAINT: {
      PAINTSTRUCT ps;
      HDC hdc = BeginPaint(hWnd, &ps);

      // Draw the image
      StretchBlt(hdc, drawing_area_inset_, drawing_area_inset_, drawing_area_width_, drawing_area_height_, hdc_dib_, 0, 0, MNIST::width_, MNIST::height_, SRCCOPY);
      SelectObject(hdc, GetStockObject(BLACK_PEN));
      SelectObject(hdc, GetStockObject(NULL_BRUSH));
      Rectangle(hdc, drawing_area_inset_, drawing_area_inset_, drawing_area_inset_ + drawing_area_width_, drawing_area_inset_ + drawing_area_height_);

      constexpr int graphs_left = drawing_area_inset_ + drawing_area_width_ + 5;
      constexpr int graph_width = 64;
      SelectObject(hdc, brush_bars_);

      auto least = *std::min_element(mnist_->results_.begin(), mnist_->results_.end());
      auto greatest = mnist_->results_[mnist_->result_];
      auto range = greatest - least;

      int graphs_zero = static_cast<int>(graphs_left - least * graph_width / range);

      // Hilight the winner
      RECT rc{graphs_left, static_cast<LONG>(mnist_->result_) * 16, graphs_left + graph_width + 128, static_cast<LONG>(mnist_->result_ + 1) * 16};
      FillRect(hdc, &rc, brush_winner_);

      // For every entry, draw the odds and the graph for it
      SetBkMode(hdc, TRANSPARENT);
      wchar_t value[80];
      for (unsigned i = 0; i < 10; i++) {
        int y = 16 * i;
        float result = mnist_->results_[i];

        auto length = wsprintf(value, L"%2d: %d.%02d", i, int(result), abs(int(result * 100) % 100));
        TextOut(hdc, graphs_left + graph_width + 5, y, value, length);

        Rectangle(hdc, graphs_zero, y + 1, static_cast<int>(graphs_zero + result * graph_width / range), y + 14);
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
      int x = (GET_X_LPARAM(lParam) - drawing_area_inset_) / drawing_area_scale_;
      int y = (GET_Y_LPARAM(lParam) - drawing_area_inset_) / drawing_area_scale_;
      MoveToEx(hdc_dib_, x, y, nullptr);
      return 0;
    }

    case WM_MOUSEMOVE:
      if (painting_) {
        int x = (GET_X_LPARAM(lParam) - drawing_area_inset_) / drawing_area_scale_;
        int y = (GET_Y_LPARAM(lParam) - drawing_area_inset_) / drawing_area_scale_;
        LineTo(hdc_dib_, x, y);
        InvalidateRect(hWnd, nullptr, false);
      }
      return 0;

    case WM_CAPTURECHANGED:
      painting_ = false;
      return 0;

    case WM_LBUTTONUP:
      ReleaseCapture();
      ConvertDibToMnist();
      mnist_->Run();
      InvalidateRect(hWnd, nullptr, true);
      return 0;

    case WM_RBUTTONDOWN:  // Erase the image
    {
      RECT rect{0, 0, MNIST::width_, MNIST::height_};
      FillRect(hdc_dib_, &rect, (HBRUSH)GetStockObject(WHITE_BRUSH));
      InvalidateRect(hWnd, nullptr, false);
      return 0;
    }

    case WM_DESTROY:
      PostQuitMessage(0);
      return 0;
  }
  return DefWindowProc(hWnd, message, wParam, lParam);
}

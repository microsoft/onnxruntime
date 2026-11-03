#include <onnxruntime_cxx_api.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

struct Frame {
  std::array<Ort::Float16_t, 64> a;
  std::array<uint8_t, 256> b;
  std::array<Ort::Float16_t, 16> scales;
};
static_assert(sizeof(Frame) == 416 && sizeof(Ort::Float16_t) == 2);
static_assert(std::is_trivially_copyable_v<Frame> && std::endian::native == std::endian::little);

int main(int argc, char** argv) {
  try {
    if (argc != 6) throw std::runtime_error("Usage: comparator capture|compare provider model inputs reference");
    const std::string mode = argv[1];
    if (mode != "capture" && mode != "compare") throw std::runtime_error("Invalid mode");
    const bool capture = mode == "capture";
    std::ifstream data(argv[4], std::ios::binary);
    std::array<char, 4> magic{};
    uint32_t version = 0, count = 0;
    if (!data.read(magic.data(), magic.size()) ||
        !data.read(reinterpret_cast<char*>(&version), sizeof(version)) ||
        !data.read(reinterpret_cast<char*>(&count), sizeof(count)) ||
        magic != std::array<char, 4>{'F', '1', '6', 'S'} || version != 1 || count == 0 || count > 1000)
      throw std::runtime_error("Invalid frame header");
    std::vector<Frame> frames(count);
    if (!data.read(reinterpret_cast<char*>(frames.data()), frames.size() * sizeof(Frame)) ||
        data.peek() != std::char_traits<char>::eof())
      throw std::runtime_error("Wrong frame payload size");
    for (const auto& frame : frames) {
      for (const auto& v : frame.a)
        if (!std::isfinite(v.ToFloat())) throw std::runtime_error("Non-finite input");
      for (const auto& v : frame.scales)
        if (!std::isfinite(v.ToFloat())) throw std::runtime_error("Non-finite scale");
    }
    std::vector<Ort::Float16_t> reference(frames.size() * 256);
    if (!capture) {
      std::ifstream input(argv[5], std::ios::binary);
      if (!input.read(reinterpret_cast<char*>(reference.data()), reference.size() * sizeof(reference[0])) ||
          input.peek() != std::char_traits<char>::eof())
        throw std::runtime_error("Wrong reference payload size");
      for (const auto& v : reference)
        if (!std::isfinite(v.ToFloat())) throw std::runtime_error("Non-finite reference");
    }
    Ort::Env env{ORT_LOGGING_LEVEL_WARNING, "dynamic-dot4-sweep"};
    env.RegisterExecutionProviderLibrary("webgpu", argv[2]);
    std::vector<Ort::ConstEpDevice> devices;
    for (const auto& device : env.GetEpDevices()) {
      if (std::string(device.EpName()) == "WebGpuExecutionProvider") {
        devices.push_back(device);
        break;
      }
    }
    if (devices.empty()) throw std::runtime_error("No WebGPU device");
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_DISABLE_ALL);
    options.AddConfigEntry("session.disable_cpu_ep_fallback", "1");
    options.AppendExecutionProvider_V2(env, devices,
        {{"enableInt64", "1"}, {"validationMode", "full"},
         {"maxNumPendingDispatches", "1"}, {"enableGraphCapture", "0"}});
    Ort::Session session{env, argv[3], options};
    auto memory = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    const int columns = capture ? 1552 : 160;
    const std::array<int64_t, 2> a_shape{16, 4}, scale_shape{columns, 1};
    const std::array<int64_t, 3> b_shape{columns, 1, 16};
    std::vector<uint8_t> b(columns * 16);
    std::vector<Ort::Float16_t> scales(columns);
    const char* input_names[]{"a", "b", "scales"};
    const char* output_names[]{"y"};
    size_t comparisons = 0, both_native = 0, corrected = 0;
    std::array<size_t, 5> mismatches{};
    for (size_t index = 0; index < frames.size(); ++index) {
      auto& frame = frames[index];
      for (int col = 0; col < columns; ++col) {
        std::copy_n(frame.b.data() + (col % 16) * 16, 16, b.data() + col * 16);
        scales[col] = frame.scales[col % 16];
      }
      std::array<Ort::Value, 3> inputs{
          Ort::Value::CreateTensor<Ort::Float16_t>(memory, frame.a.data(), frame.a.size(), a_shape.data(), a_shape.size()),
          Ort::Value::CreateTensor<uint8_t>(memory, b.data(), b.size(), b_shape.data(), b_shape.size()),
          Ort::Value::CreateTensor<Ort::Float16_t>(memory, scales.data(), scales.size(), scale_shape.data(), scale_shape.size())};
      for (int round = 0; round < 4; ++round) {
        auto outputs = session.Run(Ort::RunOptions{}, input_names, inputs.data(), inputs.size(), output_names, 1);
        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        if (info.GetShape() != std::vector<int64_t>{16, columns} ||
            info.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16)
          throw std::runtime_error("Unexpected output shape or dtype");
        const auto* values = outputs[0].GetTensorData<Ort::Float16_t>();
        for (int row = 0; row < 16; ++row) {
          for (int col = 0; col < 16; ++col) {
            auto& expected = reference[index * 256 + row * 16 + col];
            if (capture) {
              const auto& scalar = values[row * columns + 1536 + col];
              if (!std::isfinite(scalar.ToFloat())) throw std::runtime_error("Non-finite capture");
              if (round == 0) expected = scalar;
              if (std::memcmp(&expected, &scalar, sizeof(scalar)) != 0)
                throw std::runtime_error("Capture changed between repeats");
              for (int pattern : {47, 95})
                if (std::memcmp(&scalar, &values[row * columns + pattern * 16 + col], sizeof(scalar)) != 0)
                  throw std::runtime_error("Original enumeration exact pattern no longer matches");
            } else {
              const float mask_value = values[row * columns + 48 + col].ToFloat();
              if (!std::isfinite(mask_value) || mask_value < 0 || mask_value > 63 ||
                  std::trunc(mask_value) != mask_value)
                throw std::runtime_error("Invalid correction mask");
              const auto mask = static_cast<unsigned>(mask_value);
              if ((mask & 3u) != (((mask >> 2u) | (mask >> 4u)) & 3u))
                throw std::runtime_error("Mask disagrees with correction reasons");
              both_native += (mask & 3u) == 0;
              corrected += (mask & 1u) != 0;
              corrected += (mask & 2u) != 0;
              constexpr std::array<int, 5> planes{0, 1, 8, 9, 2};
              for (size_t kind = 0; kind < planes.size(); ++kind) {
                const auto& value = values[row * columns + planes[kind] * 16 + col];
                if (!std::isfinite(value.ToFloat())) throw std::runtime_error("Non-finite candidate");
                if (std::memcmp(&value, &expected, sizeof(value)) != 0) {
                  if (mismatches[kind] < 4)
                    std::cerr << "MISMATCH kind=" << kind << " frame=" << index << " row=" << row
                              << " col=" << col << " value=" << value.ToFloat()
                              << " original=" << expected.ToFloat() << '\n';
                  ++mismatches[kind];
                }
              }
            }
            ++comparisons;
          }
        }
      }
      if ((index + 1) % 100 == 0)
        std::cout << "PROGRESS frames=" << index + 1 << " comparisons=" << comparisons << std::endl;
    }
    if (capture) {
      std::ofstream output(argv[5], std::ios::binary | std::ios::trunc);
      output.write(reinterpret_cast<const char*>(reference.data()), reference.size() * sizeof(reference[0]));
      output.flush();
      if (!output.good()) throw std::runtime_error("Failed to persist capture");
      std::cout << "CAPTURE comparisons=" << comparisons << " elements=" << reference.size() << std::endl;
      return 0;
    }
    for (size_t kind = 0; kind < mismatches.size(); ++kind)
      std::cout << "CROSS kind=" << kind << " comparisons=" << comparisons
                << " mismatches=" << mismatches[kind] << std::endl;
    std::cout << "PATH both_native=" << both_native << " corrected_pairs=" << corrected << std::endl;
    return mismatches[1] == 0 ? 0 : 2;
  } catch (const std::exception& error) {
    std::cerr << "FAIL " << error.what() << std::endl;
    return 1;
  }
}

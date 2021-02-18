#pragma once

namespace Ort {
struct Env;
struct Session;
}  // namespace Ort

class Example {
 public:
  Example() = default;
  ~Example() = default;

  bool Load(const std::string& model_path);
  std::vector<Ort::Value> Run(const char* const* input_names,
                              const Ort::Value* input_values,
                              size_t input_count,
                              const char* const* output_names,
                              size_t output_count);

 private:
  Example(const Example&) = delete;
  Example& operator=(const Example&) = delete;
  Example(Example&&) = delete;
  Example& operator=(Example&&) = delete;

  std::unique_ptr<Ort::Env> env_;
  std::unique_ptr<Ort::Session> session_;
};

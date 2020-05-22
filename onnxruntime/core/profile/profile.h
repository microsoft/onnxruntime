// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cinttypes>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace onnxruntime {
namespace profile {

// Color in ARGB space.
// A: first 8 bit.
// R: later 8 bit.
// G: later 8 bit.
// B: last 8 bits
// All colo channels has range [0, 255].
enum class Color : uint32_t {
  Black = 0x00000000,
  Red = 0x00ff0000,
  DarkGreen = 0x00009900,
  Green = 0x0000ff00,
  LightGreen = 0x00ccffcc,
  Blue = 0x000000ff,
  Amber = 0x00ffbf00,
  LightAmber = 0x00fff2cc,
  White = 0x00ffffff,
  Cyan = 0x0000ffff,
  Magenta = 0x00ff00ff
};

class RangeCreatorBase {
 public:
  RangeCreatorBase(const std::string message, const Color color)
      : message_(message), color_(color),
        is_begin_called_(false), is_end_called_(false) {};

  // Check if Begin and End are both called.
  // It's pointless if not all of them are called.
  ~RangeCreatorBase() {
    if (!is_begin_called_) {
      std::cerr << "Begin must be called once." << std::endl;
      std::exit(-1);
    }
    if (!is_end_called_) {
      std::cerr << "End must be called once." << std::endl;
      std::exit(-1);
    }
  }

  // Mark the beginning of a range.
  void Begin() {
    if (is_begin_called_) {
      throw std::runtime_error("Begin cannot be called more than once.");
    }
    if (is_end_called_) {
      throw std::runtime_error("Begin cannot be called after calling End.");
    }
    BeginImpl();
    is_begin_called_ = true;
  }

  // Mark the end of a range.
  void End() {
    if (!is_begin_called_) {
      throw std::runtime_error("End must be called after calling Begin.");
    }
    if (is_end_called_) {
      throw std::runtime_error("End cannot be called more than once.");
    }
    EndImpl();
    is_end_called_ = true;
  }

  bool IsBeginCalled() const {
    return is_begin_called_;
  } 

  bool IsEndCalled() const {
    return is_end_called_;
  }

  virtual void BeginImpl() = 0;

  virtual void EndImpl() = 0;

 protected:
  void Show() {
    std::cout << "Range message: " << message_
              << ", color: " << static_cast<uint32_t>(color_)
              << ", is_begin_called: " << is_begin_called_
              << ", is_end_called: " << is_end_called_ << std::endl;
  }

  // Text on this event.
  const std::string message_;

  // Color of event in ARGB space.
  const Color color_;

  bool is_begin_called_;
  bool is_end_called_;
};

class NvtxRangeCreator final : public RangeCreatorBase {
 public:
  NvtxRangeCreator(const std::string message, const Color color)
      : RangeCreatorBase(message, color) {};

  void BeginImpl() override;
  void EndImpl() override;

 private:
#if !defined(NDEBUG) && defined(USE_CUDA) && !defined(_WIN32)
  // It records the event ID created by BeginImpl.
  // EndImpl needs this value to end the right event.
  uint64_t range_id_;
#endif
};

class NvtxNestedRangeCreator final : public RangeCreatorBase {
 public:
  NvtxNestedRangeCreator(const std::string message, const Color color)
      : RangeCreatorBase(message, color) {};

  void BeginImpl() override;
  void EndImpl() override;
};

class NvtxMarkerCreator final {
 public:
  NvtxMarkerCreator(const std::string message, const Color color)
      : message_(message), color_(color) {};
  void Mark();

 private:
  void Show() {
    std::cout << "Range message: " << message_
              << ", color: " << static_cast<uint32_t>(color_) << std::endl;
  }

  // Text on this marker.
  const std::string message_;

  // See nvtxRangeCreator.color_.
  const Color color_;
};

}  // namespace profile
}  // namespace onnxruntime
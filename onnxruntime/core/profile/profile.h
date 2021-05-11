// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// These enclosed classes are wrappers over
// generating Nvidia's visual profile APIs.
// They can be used to plot the time intervals of forward and backward passes.
// They can also be used to plot the time span of a specific operator.
// When writing this file, Nvidia only supports this tool on Linux.
#ifdef ENABLE_NVTX_PROFILE

#pragma once

#include <cinttypes>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include "core/common/common.h"

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
  Magenta = 0x00ff00ff,
  Yellow = 0x00ffff00,
};

class RangeCreatorBase {
 public:
  RangeCreatorBase(const std::string message, const Color color)
      : message_(message), color_(color), is_begin_called_(false), is_end_called_(false){};

  // Check if Begin and End are both called.
  // It's pointless if not all of them are called.
  ~RangeCreatorBase() {
    if (!is_begin_called_) {
      std::cerr << "Begin must be called once." << std::endl;
    }
    if (!is_end_called_) {
      std::cerr << "End must be called once." << std::endl;
    }
  }

  // Mark the beginning of a range.
  void Begin() {
    ORT_ENFORCE(!is_begin_called_, "Begin cannot be called more than once.");
    ORT_ENFORCE(!is_end_called_, "Begin cannot be called after calling End.");
    BeginImpl();
    is_begin_called_ = true;
  }

  // Mark the end of a range.
  void End() {
    ORT_ENFORCE(is_begin_called_, "End must be called after calling Begin.");
    ORT_ENFORCE(!is_end_called_, "End cannot be called more than once.");
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
  // Text on this event.
  const std::string message_;

  // Color of event in ARGB space.
  const Color color_;

  bool is_begin_called_;
  bool is_end_called_;
};

class NvtxRangeCreator final : public RangeCreatorBase {
 public:
  NvtxRangeCreator(const std::string message, const Color color);
  ~NvtxRangeCreator();

  void BeginImpl() override;
  void EndImpl() override;

 private:
  // It records the event ID created by BeginImpl.
  // EndImpl needs this value to end the right event.
  uint64_t range_id_;
};

class NvtxNestedRangeCreator final : public RangeCreatorBase {
 public:
  NvtxNestedRangeCreator(const std::string message, const Color color)
      : RangeCreatorBase(message, color){};

  void BeginImpl() override;
  void EndImpl() override;
};

class NvtxMarkerCreator final {
 public:
  NvtxMarkerCreator(const std::string message, const Color color)
      : message_(message), color_(color){};
  void Mark();

 private:
  // Text on this marker.
  const std::string message_;

  // See nvtxRangeCreator.color_.
  const Color color_;
};

}  // namespace profile
}  // namespace onnxruntime

#endif

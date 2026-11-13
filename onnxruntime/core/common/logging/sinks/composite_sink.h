// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <algorithm>
#include <atomic>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/common/logging/isink.h"
#include "core/common/logging/logging.h"

namespace onnxruntime {
namespace logging {
/// <summary>
/// Class that abstracts multiple ISink instances being written to.
/// </summary>
/// <seealso cref="ISink" />
class CompositeSink : public ISink {
 public:
  /// <summary>
  /// Initializes a new instance of the <see cref="CompositeSink"/> class.
  /// Use AddSink to add sinks.
  /// </summary>
  CompositeSink() : ISink(SinkType::CompositeSink) {}

  /// <summary>
  /// Check if the composite sink contains a sink of the specified type.
  /// </summary>
  bool HasType(SinkType sink_type) const {
    return std::any_of(sinks_with_severity_.begin(), sinks_with_severity_.end(),
                       [&](const auto& entry) {
                         return entry.sink->GetType() == sink_type;
                       });
  }

  /// <summary>
  /// Adds a sink. Takes ownership of the sink (so pass unique_ptr by value).
  /// </summary>
  /// <param name="sink">The sink.</param>
  /// <param name="severity">The min severity to send a message to that sink</param>
  /// <returns>This instance to allow chaining.</returns>
  CompositeSink& AddSink(std::unique_ptr<ISink> sink, logging::Severity severity) {
    if (sink && sink->GetType() == SinkType::EtwSink) {
      has_etw_sink_.store(true, std::memory_order_relaxed);
    }
    sinks_with_severity_.emplace_back(std::move(sink), severity);
    return *this;
  }

  /// <summary>
  /// Returns the minimum configured severity across sinks of the specified type, if any exist.
  /// </summary>
  std::optional<logging::Severity> GetMinSeverityOfType(SinkType sink_type) const {
    std::optional<logging::Severity> min_severity;
    for (const auto& entry : sinks_with_severity_) {
      if (entry.sink->GetType() == sink_type) {
        const auto entry_severity = entry.severity.load(std::memory_order_relaxed);
        min_severity = min_severity.has_value() ? std::min(*min_severity, entry_severity) : entry_severity;
      }
    }
    return min_severity;
  }

  /// <summary>
  /// Remove a sink of the specified type.
  /// </summary>
  /// <param name="sink_type">Sink type to remove</param>
  /// <returns>Minimum severity of the remaining sinks</returns>
  logging::Severity RemoveSink(SinkType sink_type) {
    logging::Severity severity = Severity::kFATAL;  // default if we end up with no sinks
    bool has_etw_sink = false;

    // find entries to remove and the minimum severity of the remaining sinks
    auto entries_to_remove = std::remove_if(sinks_with_severity_.begin(), sinks_with_severity_.end(),
                                            [&](const auto& entry) {
                                              if (entry.sink->GetType() == sink_type) {
                                                return true;
                                              } else {
                                                if (entry.sink->GetType() == SinkType::EtwSink) {
                                                  has_etw_sink = true;
                                                }
                                                severity = std::min(severity,
                                                                    entry.severity.load(std::memory_order_relaxed));
                                                return false;
                                              }
                                            });

    sinks_with_severity_.erase(entries_to_remove, sinks_with_severity_.end());
    has_etw_sink_.store(has_etw_sink, std::memory_order_relaxed);

    return severity;
  }

  /// <summary>
  /// Check if there's only one sink left
  /// </summary>
  /// <returns> True if only 1 sink remaining </returns>
  bool HasOnlyOneSink() const {
    return sinks_with_severity_.size() == 1;
  }

  /// <summary>
  /// If one sink is remaining then returns it and empties the composite sink
  /// </summary>
  /// <returns> If one sink remains then returns the sink, otherwise nullptr </returns>
  std::unique_ptr<ISink> GetRemoveSingleSink() {
    if (HasOnlyOneSink()) {
      auto single_sink = std::move(sinks_with_severity_.begin()->sink);
      sinks_with_severity_.clear();
      has_etw_sink_.store(false, std::memory_order_relaxed);
      return single_sink;
    }
    return nullptr;
  }

  void SendProfileEvent(profiling::EventRecord& event_record) const override {
    for (const auto& entry : sinks_with_severity_) {
      entry.sink->SendProfileEvent(event_record);
    }
  }

 private:
  struct SinkWithSeverity {
    SinkWithSeverity(std::unique_ptr<ISink> sink_in, logging::Severity severity_in)
        : sink(std::move(sink_in)), severity(severity_in) {}

    SinkWithSeverity(SinkWithSeverity&& other) noexcept
        : sink(std::move(other.sink)),
          severity(other.severity.load(std::memory_order_relaxed)) {}

    SinkWithSeverity& operator=(SinkWithSeverity&& other) noexcept {
      if (this != &other) {
        sink = std::move(other.sink);
        severity.store(other.severity.load(std::memory_order_relaxed), std::memory_order_relaxed);
      }
      return *this;
    }

    ORT_DISALLOW_COPY_AND_ASSIGNMENT(SinkWithSeverity);

    std::unique_ptr<ISink> sink;
    std::atomic<logging::Severity> severity;
  };

  void SendImpl(const Timestamp& timestamp, const std::string& logger_id, const Capture& message) override {
    // When an ETW sink is combined with a primary BaseSink, each Logger maintains its own
    // configured severity (e.g. existing session loggers or explicit session/run overrides)
    // while Logger::OutputIsEnabled() also admits messages required by ETW. Gate BaseSink by
    // the originating Logger's configured severity rather than a single global threshold so
    // ETW presence never changes BaseSink/callback delivery, while preserving EtwSink filtering.
#ifndef ORT_MINIMAL_BUILD
    const bool use_logger_severity_for_base =
        has_etw_sink_.load(std::memory_order_relaxed) && message.GetLogger() != nullptr;
#else
    constexpr bool use_logger_severity_for_base = false;
#endif
    for (auto& entry : sinks_with_severity_) {
      const logging::Severity min_severity =
          (use_logger_severity_for_base && entry.sink->GetType() == SinkType::BaseSink)
              ? message.GetLogger()->GetSeverity()
              : entry.severity.load(std::memory_order_relaxed);
      if (message.Severity() >= min_severity) {
        entry.sink->Send(timestamp, logger_id, message);
      }
    }
  }

  std::vector<SinkWithSeverity> sinks_with_severity_;
  std::atomic<bool> has_etw_sink_{false};
};
}  // namespace logging
}  // namespace onnxruntime

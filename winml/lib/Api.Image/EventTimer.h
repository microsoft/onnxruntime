#include "pch.h"

class EventTimer {
 public:
  bool Start() {
    auto now = std::chrono::high_resolution_clock::now();
    if (!_started || std::chrono::duration_cast<std::chrono::microseconds>(now - _startTime).count() > _kDurationBetweenSendingEvents) {
      _started = true;
      _startTime = std::chrono::high_resolution_clock::now();
      return true;
    }

    return false;
  }

 private:
  bool _started = false;
  std::chrono::steady_clock::time_point _startTime;
  constexpr static long long _kDurationBetweenSendingEvents =
    1000 * 50;  // duration in (us). send an Event every 50 ms;
};

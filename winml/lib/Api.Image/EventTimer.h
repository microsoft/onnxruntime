#include "pch.h"

class EventTimer
{
public:
  EventTimer(long long durationInMicroSeconds)
    : _durationInMicroSeconds(durationInMicroSeconds)
  {
  }
  
  bool Start()
  {
    auto now = std::chrono::high_resolution_clock::now();
    if (!_started || 
         std::chrono::duration_cast<std::chrono::microseconds>(now - _startTime).count() > _durationInMicroSeconds)
    {
      _started = true;
      _startTime = std::chrono::high_resolution_clock::now();
      return true;
    }

    return false;
  }

private:
  bool _started = false;
  long long _durationInMicroSeconds;
  std::chrono::steady_clock::time_point _startTime;
};


// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#ifndef __TEST_LOG_H__
#define __TEST_LOG_H__

#include <iostream>
#include <sstream>
#include <vector>
#include <map>
#include <fstream>
#include <unordered_map>

// Forward declarartions
//
class OnnxPrediction;
std::wostream& operator<<(std::wostream& out, OnnxPrediction& pred);

namespace Logger
{
  // Alias for end of line
  //
  using LogEndln = void*;

  // Used to format the output of the logs.
  //
  class TestLog
  {
    public:    
      // Flush out all output streams
      // used by logger
      //
      void flush();

      // Print out output of a prediction.
      //
      TestLog& operator<<(OnnxPrediction& pred);

      // Generic log output that appends timing
      // information.
      //
      template<typename T>
      TestLog& operator<<(const T& info);

      // Disable logging
      //
      inline void disable();

      // Enable logging
      //
      inline void enable();

      // Ends the current line so that the
      // next line can start with time information.
      //
      void operator<<(LogEndln info);

      // Minimize log
      //
      inline void minLog();

      // Maintain ring buffer
      // Note:
      // This is only used for minimun logging
      // if normal logging is being used this map
      // must be constrained.
      //
      void insert(std::wstring data);

      // Singleton constructor only one object exists
      // Hence this resource is not thread-safe
      //
      TestLog();

    private:
      bool print_time_info = true;
      bool logging_on = true;
      bool min_log = false;
      std::wofstream logFile;
      std::wstring logFileName;
      std::map<size_t, std::pair<size_t, std::wstring>> ring_buffer;
      static constexpr int logFileLineWidth{128};
      static constexpr int logFileLen{1000};
  };
  
  // Reference to initialized logger
  // Not this resource is not thread safe and only one
  // exists for the entire process.
  //
  extern TestLog testLog;

  // Object used to mark end of line format
  // for testLog
  //
  static constexpr LogEndln endl = nullptr;

  // Utility function to convert from char to wchar
  //
  std::wstring towstr(const char *pStr);
}

// Inline Functions

// Minimize log
//
inline void Logger::TestLog::minLog()
{
  min_log = true;
}

// Enable logging
//
inline void Logger::TestLog::enable()
{
  logging_on = true;
}

inline void Logger::TestLog::disable()
{
  logging_on = false;
}

// Template functions 

// Generic log output that appends timing
// information.
//
template<typename T>
Logger::TestLog& Logger::TestLog::operator<<(const T& info)
{
  if (!logging_on)
  {
    return *this;
  }

  std::chrono::system_clock::time_point today{std::chrono::system_clock::now()};
  time_t tt{std::chrono::system_clock::to_time_t( today ) };
  constexpr int length_time_str = 28;
  char buf[length_time_str];

  if (0 == ctime_s(buf, sizeof(buf), &tt))
  {    
    wchar_t wbuf[length_time_str];
    char const *ptr = buf;
    std::mbstate_t ps;
    size_t retVal;
    mbsrtowcs_s(&retVal, wbuf, length_time_str, &ptr, length_time_str, &ps );
    std::wstring_view temp(wbuf, retVal - 2);
    std::wstringstream stream;
    if (print_time_info)
    {
      stream << L"[" << temp << L"]" << L"\t";
    }

    if constexpr (std::is_same<T, std::string>())
    {
      stream << towstr(info.data());
    }
    else
    {
      stream << info;
    }
    
    if (min_log)
    {
      insert(stream.str());
    }
    else
    {
      std::wcout << stream.str();
    }
    print_time_info = false;
  }
  return *this;
}

namespace Logger
{
  template <typename CharT>
  class cache_streambuf : public std::basic_streambuf<CharT>
  {
    public:
      using Base = std::basic_streambuf<CharT>;
      using char_type = typename Base::char_type;
      using int_type = typename Base::int_type;

      // Get the total number of unique errors found
      //
      inline size_t get_unique_errors();
    
    protected:
      virtual int_type overflow( int_type ch = Traits::eof() );

    private:
      std::basic_stringstream<char_type> buffer;
      std::unordered_map<std::basic_string<char_type>, size_t> exception_count;
  };

  using ccstream = cache_streambuf<char>;
  using wcstream = cache_streambuf<wchar_t>;

  template<typename CharT>
  inline size_t cache_streambuf<CharT>::get_unique_errors()
  {
    return exception_count.size();
  }

  template<typename CharT>
  auto cache_streambuf<CharT>::overflow( int_type ch ) -> int_type 
  {
    // if not end of file
    //
    if( ! Base::traits_type::eq_int_type(ch,
            Base::traits_type::eof()))
    {
      if(ch > 255)
      {
        if constexpr (std::is_same_v<char_type, char>)
        {
          std::cout << "Yikes";
        }
        else
        {
          std::wcout << L"Yikes"; 
        } 
      }
      if(ch != int_type{'\n'})
      {
        buffer << static_cast<char>(ch);
      }
      else
      {
        buffer << static_cast<char>(ch);
        exception_count[buffer.str()]++;
        if constexpr (std::is_same_v<char_type, char>)
        {
          std::cout << buffer.str();
        }
        else
        {
          std::wcout << buffer.str(); 
        } 
        buffer = std::basic_stringstream<char_type>{};
      }
    }
    return Base::traits_type::not_eof(ch);  
  }
}

#endif
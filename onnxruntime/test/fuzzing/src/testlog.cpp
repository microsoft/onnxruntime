// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "testlog.h"
#include <filesystem>

namespace Logger
{
    // Initialize static variable for the whole
    // process
    //
    TestLog testLog{};

    // Maintain ring buffer
    // Note:
    // This is only used for minimun logging
    // if normal logging is being used this map
    // must be constrained.
    //
    void TestLog::insert(std::wstring data)
    {
        static size_t index = 0;
        ring_buffer[index] = std::pair{index, data};
        index++;
    }

    // Output Representation of pred to console out
    //
    TestLog& TestLog::operator<<(OnnxPrediction& pred)
    {
        if (!this->logging_on)
        {
            return *this;
        }
        std::wcout << pred;
        return *this;
    }

    // Flush out all output streams
    // used by logger
    //
    void TestLog::flush()
    {
        bool newLine = true;
        static size_t line_count = 0;
        std::wstringstream line{};

        for(auto& str_ : ring_buffer)
        {
            auto str = str_.second;
            // Create a line that terminates with a new line
            //
            newLine ?  
                (line << str.first << str.second)
                : str.second != L"\n" ? (line << str.second)
                : (line << L"");

            newLine = str.second == L"\n";

            // Process current line
            //
            if (newLine)
            {
                line_count++;
                auto count{line.str().size()};
                if (count > TestLog::logFileLineWidth)
                {
                    logFile << line.str().substr(TestLog::logFileLineWidth);
                    std::wcout << line.str().substr(TestLog::logFileLineWidth);
                }
                else
                {
                    size_t chDiff = TestLog::logFileLineWidth - count;
                    auto padding = std::wstring(chDiff, L' ');
                    line << padding << str.second;
                    logFile << line.str();
                    std::wcout << line.str();
                }
                line = std::wstringstream{};

                if (line_count > TestLog::logFileLen)
                {
                    logFile.seekp(0, std::fstream::beg);
                    line_count = 0;
                }
            }
        }

        ring_buffer.clear();
        logFile.flush();
        std::cout << std::flush;
        std::wcout << std::flush;
    }

    // Ends the current line so that the
    // next line can start with time information.
    //
    void TestLog::operator<<(LogEndln info)
    {
        if (!logging_on)
        {
            return;
        }
        
        if (min_log)
        {
            insert(std::wstring(L"\n"));
        }
        else
        {
            std::wcout << L"\n";
        }
        
        print_time_info = true;
        (void) info;
    }

    // Singleton constructor only one object exists
    // Hence this resource is not thread-safe
    //
    TestLog::TestLog()
    :logFileName{L"out"}
    {
        static bool init{false};
        if(!init)
        {
            init = true;
            
            std::filesystem::path mutateModelDir{logFileName};
            if ( !std::filesystem::exists(mutateModelDir) )
            {
                std::filesystem::create_directory(mutateModelDir);
            }
            logFile.open(L"out/log", std::ios::ate);
        }
        else
        {
            throw std::runtime_error("TestLog has already been initialized. Call GetTestLog() to use it");
        }
    }

    // Helper function to convert string to wstring
    //
    std::wstring towstr(const char *pStr)
    {
        std::mbstate_t ps;
        size_t retVal;
        size_t length_str = std::strnlen(pStr, 65535);
        mbsrtowcs_s(&retVal, nullptr, 0, &pStr, length_str, &ps );
        retVal += 1;
        auto ptr = std::make_unique<wchar_t[]>(retVal);
        if (ptr == nullptr)
        {
            std::stringstream str;
            str << "Failed to allocate memory: " << __func__ << __LINE__ <<"\n"; 
            throw std::exception{str.str().data()};
        }
        mbsrtowcs_s(&retVal, ptr.get(), retVal, &pStr, length_str, &ps );
        return std::wstring{ptr.get()};
    }
}

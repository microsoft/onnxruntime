// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <string_view>
#include <locale>
#include <codecvt>
        

namespace Dml
{
    static inline std::wstring ConvertToWString(std::string_view str)
    {
        std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>,wchar_t> g_converterToUtf16;
        return g_converterToUtf16.from_bytes(str.data());
    }

    static inline std::wstring GetModelName(const onnxruntime::Path& modelPath)
    {
        if (modelPath.GetComponents().empty())
        {
            return L"";
        }
        
        const onnxruntime::PathString& pathString = modelPath.GetComponents().back();
        size_t dotPosition = pathString.find_last_of('.');
        if (dotPosition == std::string::npos)
        {
            return L"";
        }

        return pathString.substr(0, dotPosition);
    }

    static inline std::wstring GetSanitizedFileName(std::wstring_view name)
    {
        std::wstring newName(name);
        for (wchar_t& c : newName)
        {
            switch (c)
            {
            case '\\':
            case '/':
            case '\"':
            case '|':
            case '<':
            case '>':
            case ':':
            case '?':
            case '*':
                c = '_';
                break;
            }
        }
        return newName;
    }

    static inline std::string GetSanitizedFileName(std::string_view name)
    {
        std::string newName(name);
        for (char& c : newName)
        {
            switch (c)
            {
            case '\\':
            case '/':
            case '\"':
            case '|':
            case '<':
            case '>':
            case ':':
            case '?':
            case '*':
                c = '_';
                break;
            }
        }
        return newName;
    }

    static inline void WriteToFile(std::wstring_view directoryName, std::wstring_view fileName, std::uint8_t* data, size_t dataSize)
    {
        std::wstring sanitizedFileName = GetSanitizedFileName(fileName);
        std::filesystem::create_directory(directoryName);
        std::wstring fullSanitizedFileName = std::wstring(directoryName) +
                                (directoryName.empty() ? L"" : L"/") +
                                sanitizedFileName;
        std::ofstream file(fullSanitizedFileName, std::ios::binary);
        if (!file.is_open()) 
        {
            std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>,wchar_t> g_converterToUtf16;
            std::stringstream errorMessage;
            errorMessage << "File named: " << g_converterToUtf16.to_bytes(fileName.data()) << " could not be opened\n";
            throw std::ios::failure(errorMessage.str());
        }
        file.write(reinterpret_cast<const char*>(data), dataSize);
    }

namespace StringUtil
{
    struct NameAndIndex
    {
        const char* name; // Null terminated.
        uint32_t index;
    };

    struct WideNameAndIndex
    {
        const wchar_t* name; // Null terminated.
        uint32_t index;
    };

    inline std::optional<uint32_t> MapToIndex(std::string_view mode, gsl::span<const NameAndIndex> nameAndIndexList)
    {
        for (auto& nameAndIndex : nameAndIndexList)
        {
            if (strncmp(nameAndIndex.name, mode.data(), mode.size()) == 0)
            {
                return nameAndIndex.index;
            }
        }

        return {};
    }

    inline std::optional<uint32_t> MapToIndex(std::wstring_view mode, gsl::span<const WideNameAndIndex> nameAndIndexList)
    {
        for (auto& nameAndIndex : nameAndIndexList)
        {
            if (wcsncmp(nameAndIndex.name, mode.data(), mode.size()) == 0)
            {
                return nameAndIndex.index;
            }
        }

        return {};
    }
}
}
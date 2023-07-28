// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <string>
#include <string_view>

namespace Dml
{
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

    static inline void WriteToFile(std::string_view fileName, std::uint8_t* data, size_t dataSize)
    {
        std::string sanitizedFileName = GetSanitizedFileName(fileName);
        std::ofstream file(sanitizedFileName, std::ios::binary);
        if (!file.is_open()) 
        {
            std::stringstream errMsg;
            errMsg << "File named: " << fileName << " could not be opened\n";
            throw std::ios::failure(errMsg.str());
        }
        file.write(reinterpret_cast<const char*>(data), dataSize);
    }
}

//-----------------------------------------------------------------------------
//
//  Copyright (c) Microsoft Corporation. All rights reserved.
//
//-----------------------------------------------------------------------------

#pragma once

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
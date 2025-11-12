// Forked from
// https://devicesasg.visualstudio.com/PerceptiveShell/_git/PerceptiveShell?version=GC9676d0b06e55ed2443206f475ea846293e9ee130&path=/src/perceptiveshell_private/perceptiveshell_private/include/perceptiveshell_private/crypto.h
#pragma once

#include "crypto.h"

#include <cstddef>
#include <string_view>
#include <vector>

extern "C" void decrypt_impl(const char* key, long long key_size, char* data, long long size);

void Crypto::decrypt(std::string_view key, std::vector<uint8_t>& data)
{
    decrypt_impl(key.data(), key.size(), reinterpret_cast<char*>(data.data()), data.size());
}

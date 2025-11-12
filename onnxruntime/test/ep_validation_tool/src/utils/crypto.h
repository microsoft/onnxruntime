// Forked from
// https://devicesasg.visualstudio.com/PerceptiveShell/_git/PerceptiveShell?version=GC9676d0b06e55ed2443206f475ea846293e9ee130&path=/src/perceptiveshell_private/perceptiveshell_private/include/perceptiveshell_private/crypto.h
#pragma once

#include <cstddef>
#include <string_view>
#include <vector>

namespace Crypto
{
void decrypt(std::string_view key, std::vector<uint8_t>& data);
} // namespace Crypto

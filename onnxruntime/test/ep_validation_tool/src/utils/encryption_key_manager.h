// Forked from
// https://devicesasg.visualstudio.com/PerceptiveShell/_git/PerceptiveShell?version=GC2e88aa051dc10c8a8434a6240e25be4ca6963c9c&path=/src/perceptiveshell_private/perceptiveshell_private/include/perceptiveshell_private/encryption_key_manager.h
#pragma once

#include <string>
#include <unordered_map>

namespace EncryptionKeyManager
{
/*!
 *   \brief Get the encryption key from the key id.
 *
 *   This function retrieves the encryption key associated with a given key id.
 *
 *   \param[in] keyId Key id.
 *   \return The encryption key for the key id as a null-terminated string.
 */
const char* GetEncryptionKeyFromKeyId(const std::string& keyId);
} // namespace EncryptionKeyManager

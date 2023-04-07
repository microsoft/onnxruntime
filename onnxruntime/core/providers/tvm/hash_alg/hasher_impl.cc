// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "hasher_impl.h"  // NOLINT(build/include_subdir)

namespace onnxruntime {
namespace tvm {

std::string HasherSHA256Impl::hash(const char* src, size_t size) const {
  return hexdigest(src, size);
}

void HasherSHA256Impl::digest(const Ipp8u* src, int size, Ipp8u* dst) {
    IppStatus status = ippStsNoErr;
    const IppsHashMethod* hashMethod = ippsHashMethod_SHA256();
    status = ippsHashMessage_rmf(src, size, dst, hashMethod);
    if (ippStsNoErr != status) {
        ORT_THROW("Can't get SHA-256...");
    }
}

std::string HasherSHA256Impl::digest(const char* src, size_t size) {
    const int digest_size_byte = IPP_SHA256_DIGEST_BITSIZE / 8;
    auto dst = std::unique_ptr<char>(new char[digest_size_byte]);
    digest(reinterpret_cast<const Ipp8u*>(src), static_cast<int>(size), reinterpret_cast<Ipp8u*>(dst.get()));
    return std::string(dst.get(), digest_size_byte);
}

std::string HasherSHA256Impl::hexdigest(const char* src, size_t size) {
    std::string byte_digest = digest(src, size);
    std::stringstream ss;
    for (char c : byte_digest) {
        ss << std::hex << std::setw(2) << std::setfill('0') << (0xff & c);
    }
    return ss.str();
}

}   // namespace tvm
}   // namespace onnxruntime

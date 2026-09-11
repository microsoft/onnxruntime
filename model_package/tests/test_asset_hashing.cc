// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

/// \file test_asset_hashing.cc
/// \brief Tests for the directory Merkle hash and SHA-256 implementation.

#include "model_package.h"
#include "model_package_api.h"
#include "sha256.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>

namespace fs = std::filesystem;
using model_package::Sha256;

namespace {

int g_failed = 0;
int g_passed = 0;
const char* g_current = "<none>";

#define CHECK(cond)                                                                       \
  do {                                                                                    \
    if (!(cond)) {                                                                        \
      std::fprintf(stderr, "[FAIL] %s line %d: CHECK(%s)\n", g_current, __LINE__, #cond); \
      return false;                                                                       \
    }                                                                                     \
  } while (0)

#define CHECK_OK(status)                                                 \
  do {                                                                   \
    ModelPackageStatus* _s = (status);                                   \
    if (_s != nullptr) {                                                 \
      std::fprintf(stderr, "[FAIL] %s line %d: expected OK, got: %s\n",  \
                   g_current, __LINE__, ModelPackageStatus_Message(_s)); \
      ModelPackageStatus_Release(_s);                                    \
      return false;                                                      \
    }                                                                    \
  } while (0)

class Sandbox {
 public:
  Sandbox() {
    std::random_device rd;
    std::mt19937_64 g(rd());
    char buf[32];
    std::snprintf(buf, sizeof(buf), "mp_hash_%016llx", static_cast<unsigned long long>(g()));
    root_ = fs::temp_directory_path() / buf;
    fs::create_directories(root_);
  }
  ~Sandbox() {
    std::error_code ec;
    fs::remove_all(root_, ec);
  }
  Sandbox(const Sandbox&) = delete;
  Sandbox& operator=(const Sandbox&) = delete;
  const fs::path& root() const { return root_; }
  void Write(const std::string& relpath, const std::string& contents) {
    fs::path full = root_ / relpath;
    fs::create_directories(full.parent_path());
    std::ofstream f(full, std::ios::binary);
    f << contents;
  }

 private:
  fs::path root_;
};

// FIPS-180-4 known-answer test vectors.
bool test_sha256_known_vectors() {
  CHECK(Sha256::HashStringHex("") ==
        "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
  CHECK(Sha256::HashStringHex("abc") ==
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
  // Long message: 1,000,000 'a' characters.
  std::string a_million(1000000, 'a');
  CHECK(Sha256::HashStringHex(a_million) ==
        "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0");
  return true;
}

bool test_sha256_incremental_matches_oneshot() {
  std::string msg = "the quick brown fox jumps over the lazy dog";
  std::string oneshot = Sha256::HashStringHex(msg);
  Sha256 h;
  for (char c : msg) h.Update(&c, 1);
  CHECK(h.FinalHex() == oneshot);
  return true;
}

bool test_directory_hash_basic() {
  Sandbox s;
  s.Write("a.txt", "alpha");
  s.Write("b.txt", "beta");

  const char* uri = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s.root().c_str(), &uri));
  CHECK(uri != nullptr);
  std::string u(uri);
  CHECK(u.substr(0, 7) == "sha256:");
  CHECK(u.size() == 7 + 64);
  return true;
}

bool test_directory_hash_reproducible() {
  Sandbox s1;
  s1.Write("a.txt", "alpha");
  s1.Write("nested/b.txt", "beta");

  Sandbox s2;
  s2.Write("a.txt", "alpha");
  s2.Write("nested/b.txt", "beta");

  const char* u1 = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s1.root().c_str(), &u1));
  std::string copy1(u1);

  const char* u2 = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s2.root().c_str(), &u2));
  CHECK(copy1 == std::string(u2));
  return true;
}

bool test_directory_hash_name_change_differs() {
  Sandbox s1;
  s1.Write("a.txt", "alpha");

  Sandbox s2;
  s2.Write("b.txt", "alpha");  // same content, different name

  const char* u1 = nullptr;
  const char* u2 = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s1.root().c_str(), &u1));
  std::string copy1(u1);
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s2.root().c_str(), &u2));
  CHECK(copy1 != std::string(u2));
  return true;
}

bool test_directory_hash_swapped_names_differ() {
  Sandbox s1;
  s1.Write("a.txt", "alpha");
  s1.Write("b.txt", "beta");

  Sandbox s2;
  s2.Write("a.txt", "beta");  // swapped contents
  s2.Write("b.txt", "alpha");

  const char* u1 = nullptr;
  const char* u2 = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s1.root().c_str(), &u1));
  std::string copy1(u1);
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s2.root().c_str(), &u2));
  CHECK(copy1 != std::string(u2));
  return true;
}

bool test_directory_hash_content_change_differs() {
  Sandbox s1;
  s1.Write("a.txt", "alpha");
  Sandbox s2;
  s2.Write("a.txt", "ALPHA");

  const char* u1 = nullptr;
  const char* u2 = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s1.root().c_str(), &u1));
  std::string copy1(u1);
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s2.root().c_str(), &u2));
  CHECK(copy1 != std::string(u2));
  return true;
}

bool test_directory_hash_empty_dirs_ignored() {
  Sandbox s1;
  s1.Write("a.txt", "alpha");
  Sandbox s2;
  s2.Write("a.txt", "alpha");
  fs::create_directories(s2.root() / "empty_subdir");

  const char* u1 = nullptr;
  const char* u2 = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s1.root().c_str(), &u1));
  std::string copy1(u1);
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s2.root().c_str(), &u2));
  CHECK(copy1 == std::string(u2));
  return true;
}

bool test_directory_hash_rejects_symlink() {
  Sandbox s;
  s.Write("a.txt", "alpha");
  std::error_code ec;
  fs::create_symlink("a.txt", s.root() / "a_link.txt", ec);
  // If symlink creation isn't supported on this filesystem, skip the test
  // (treat as pass — the rejection is the behavior under test).
  if (ec) {
    std::printf("[SKIP] %s (symlink unsupported)\n", g_current);
    return true;
  }
  const char* uri = nullptr;
  ModelPackageStatus* st = ModelPackage_ComputeDirectoryHash(s.root().c_str(), &uri);
  CHECK(st != nullptr);
  CHECK(ModelPackageStatus_Code(st) == MODEL_PACKAGE_ERR_SCHEMA);
  ModelPackageStatus_Release(st);
  return true;
}

bool test_directory_hash_known_value_single_file() {
  // Known-answer check: the directory URI hashes a manifest of "<file_hex>  <name>\n"
  // lines, so compute the expected value the same way and compare.
  Sandbox s;
  s.Write("a.txt", "alpha");

  std::string file_hex = Sha256::HashStringHex("alpha");
  std::string manifest = file_hex + "  a.txt\n";
  std::string expected = "sha256:" + Sha256::HashStringHex(manifest);

  const char* uri = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s.root().c_str(), &uri));
  CHECK(std::string(uri) == expected);
  return true;
}

bool test_directory_hash_sorted_order_independent_of_walk() {
  // Whether the OS walks "b.txt" before "a.txt" must not matter.
  Sandbox s;
  s.Write("a.txt", "alpha");
  s.Write("b.txt", "beta");
  s.Write("c.txt", "gamma");

  // Compute expected manifest manually (sorted).
  std::string hex_a = Sha256::HashStringHex("alpha");
  std::string hex_b = Sha256::HashStringHex("beta");
  std::string hex_c = Sha256::HashStringHex("gamma");
  std::string manifest = hex_a + "  a.txt\n" +
                         hex_b + "  b.txt\n" +
                         hex_c + "  c.txt\n";
  std::string expected = "sha256:" + Sha256::HashStringHex(manifest);

  const char* uri = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s.root().c_str(), &uri));
  CHECK(std::string(uri) == expected);
  return true;
}

bool test_directory_hash_uses_forward_slash() {
  Sandbox s;
  s.Write("dir/sub/c.txt", "x");

  std::string file_hex = Sha256::HashStringHex("x");
  // Path must be POSIX style in the manifest (forward slashes).
  std::string manifest = file_hex + "  dir/sub/c.txt\n";
  std::string expected = "sha256:" + Sha256::HashStringHex(manifest);

  const char* uri = nullptr;
  CHECK_OK(ModelPackage_ComputeDirectoryHash(s.root().c_str(), &uri));
  CHECK(std::string(uri) == expected);
  return true;
}

bool test_missing_directory_errors() {
  const char* uri = nullptr;
  ModelPackageStatus* s = ModelPackage_ComputeDirectoryHash("/tmp/does_not_exist_xyzzy_zzz", &uri);
  CHECK(s != nullptr);
  CHECK(ModelPackageStatus_Code(s) == MODEL_PACKAGE_ERR_NOT_FOUND);
  ModelPackageStatus_Release(s);
  return true;
}

struct Test {
  const char* name;
  bool (*fn)();
};

const Test kTests[] = {
    {"sha256_known_vectors", test_sha256_known_vectors},
    {"sha256_incremental_matches_oneshot", test_sha256_incremental_matches_oneshot},
    {"directory_hash_basic", test_directory_hash_basic},
    {"directory_hash_reproducible", test_directory_hash_reproducible},
    {"directory_hash_name_change_differs", test_directory_hash_name_change_differs},
    {"directory_hash_swapped_names_differ", test_directory_hash_swapped_names_differ},
    {"directory_hash_content_change_differs", test_directory_hash_content_change_differs},
    {"directory_hash_empty_dirs_ignored", test_directory_hash_empty_dirs_ignored},
    {"directory_hash_rejects_symlink", test_directory_hash_rejects_symlink},
    {"directory_hash_known_value_single_file", test_directory_hash_known_value_single_file},
    {"directory_hash_sorted_order_independent_of_walk", test_directory_hash_sorted_order_independent_of_walk},
    {"directory_hash_uses_forward_slash", test_directory_hash_uses_forward_slash},
    {"missing_directory_errors", test_missing_directory_errors},
};

}  // namespace

int main() {
  for (const auto& t : kTests) {
    g_current = t.name;
    bool ok = t.fn();
    if (ok) {
      std::printf("[PASS] %s\n", t.name);
      g_passed++;
    } else {
      g_failed++;
    }
  }
  std::printf("\n=== %d passed, %d failed ===\n", g_passed, g_failed);
  return g_failed == 0 ? 0 : 1;
}

from pathlib import Path
import subprocess
import tempfile
import unittest

from build_bind_group_inline_key import HASH, HELPERS, ORIGINAL_HASH, ORIGINAL_KEY, transform


class BindGroupInlineKeyTests(unittest.TestCase):
    def test_existing_lifetime_eviction_and_byte_bounds_remain(self):
        source = ("#include <vector>\nstruct WebGpuContextFactory::BindGroupCache {\n" + ORIGINAL_HASH +
            "\nstd::unordered_map<std::vector<uint64_t>, wgpu::BindGroup, Hash> entries;\n};\n" +
            ORIGINAL_KEY + "\nauto found = cache.entries.find(key);\n" +
            "  cache.entries.emplace(std::move(key), result);\n" +
            "constexpr uint64_t kMaxEntryBytes = 1024 * 1024;\n"
            "constexpr uint64_t kMaxCacheBytes = 256 * 1024 * 1024;\ncache.Clear();")
        result = transform(source)
        for text in ("kMaxEntryBytes = 1024 * 1024", "kMaxCacheBytes = 256 * 1024 * 1024",
                     "cache.Clear()", "key.assign(lookup_key.begin(), lookup_key.end())"):
            self.assertIn(text, result)

    def test_actual_key_hash_equality_allocations_and_owned_lifetime(self):
        prelude = r'''
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <span>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>
std::atomic<int> allocations{0};
void* operator new(std::size_t n) {
  ++allocations;
  if (void* p = std::malloc(n)) return p;
  throw std::bad_alloc();
}
void operator delete(void* p) noexcept { std::free(p); }
void operator delete(void* p, std::size_t) noexcept { std::free(p); }
#define ORT_ENFORCE(condition, message) do { if (!(condition)) throw std::runtime_error(message); } while (false)
template <typename T> T ParseEnvironmentVariableWithDefault(const std::string& name, T fallback) {
  const char* text = std::getenv(name.c_str());
  return text ? static_cast<T>(std::stoi(text)) : fallback;
}
struct Entry { uint32_t binding; void* buffer; uint64_t offset; uint64_t size; };
struct WGPUBindGroupDescriptor { void* layout; size_t entryCount; const Entry* entries; };
'''
        driver = r'''
int main(int argc, char** argv) {
  if (argc != 2) return 1;
  setenv("ORT_WEBGPU_BIND_GROUP_INLINE_KEY", argv[1], 1);
  int mode = std::stoi(argv[1]);
  if (mode == 2) {
    try { BindGroupInlineKeyEnabled(); } catch (const std::runtime_error&) { return 0; }
    return 2;
  }
  bool enabled = BindGroupInlineKeyEnabled();
  std::vector<Entry> entries(33);
  for (size_t i = 0; i < entries.size(); ++i)
    entries[i] = {static_cast<uint32_t>(i), reinterpret_cast<void*>(0x1000 + i * 64), i * 4, i + 1};
  for (size_t count : {size_t{0}, size_t{1}, size_t{16}, size_t{32}, size_t{33}}) {
    WGPUBindGroupDescriptor descriptor{reinterpret_cast<void*>(0x2000), count, entries.data()};
    std::array<uint64_t, 129> local;
    std::vector<uint64_t> owned, original_owned;
    auto expected = BindGroupLookupKey(descriptor, false, local, original_owned);
    int before = allocations;
    auto key = BindGroupLookupKey(descriptor, enabled, local, owned);
    int added = allocations - before;
    if (added != (enabled && count <= 32 ? 0 : 1)) return 3;
    if (!Equal{}(key, expected) || Hash{}(key) != OriginalHash{}(original_owned)) return 4;
    std::unordered_map<std::vector<uint64_t>, int, Hash, Equal> cache;
    cache.emplace(std::vector<uint64_t>(key.begin(), key.end()), 7);
    before = allocations;
    if (cache.find(key) == cache.end() || allocations != before) return 5;
    std::vector<uint64_t> distinct(key.begin(), key.end());
    for (size_t index = 0; index < distinct.size(); ++index) {
      ++distinct[index];
      if (cache.find(std::span<const uint64_t>(distinct)) != cache.end()) return 6;
      --distinct[index];
    }
    local.fill(0);
    owned.clear();
    if (cache.find(std::span<const uint64_t>(original_owned)) == cache.end()) return 7;
  }
  return 0;
}
'''
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "key.cc"
            source.write_text(prelude + HELPERS + ORIGINAL_HASH.replace("struct Hash", "struct OriginalHash") + HASH + driver)
            executable = Path(directory) / "key"
            subprocess.run(["c++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(source), "-o", str(executable)], check=True)
            for mode in ("0", "1", "2"):
                subprocess.run([str(executable), mode], check=True)


if __name__ == "__main__":
    unittest.main()

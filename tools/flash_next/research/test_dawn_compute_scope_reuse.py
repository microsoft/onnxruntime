from pathlib import Path
import subprocess
import tempfile
import unittest

from build_dawn_compute_scope_reuse import WRAPPER, transform


SOURCE = Path("/home/sayanshaw/repos/onnxruntime/build/Linux/Release/_deps/dawn-src/src/dawn/native/ComputePassEncoder.cpp")


class ScopeReuseTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = SOURCE.read_text()
        cls.header = SOURCE.with_suffix(".h").read_text()
        cls.changed, cls.changed_header = transform(cls.original, cls.header)

    def test_only_scope_function_changes_after_private_selector_preamble(self):
        restored = self.changed.replace(WRAPPER, "", 1).replace(
            "void ComputePassEncoder::AddDispatchSyncScopeInternal(SyncScopeUsageTracker& scope) {",
            "void ComputePassEncoder::AddDispatchSyncScope(SyncScopeUsageTracker scope) {", 1)
        start = restored.index("#ifndef ORT_WEBGPU_DAWN_REUSE_COMPUTE_SCOPE\n")
        end = restored.index("namespace dawn::native {\n", start)
        restored = restored[:start] + restored[end:]
        self.assertEqual(restored, self.original)

    def test_header_has_no_layout_change(self):
        restored = self.changed_header.replace(
            "    void AddDispatchSyncScope();\n"
            "    void AddDispatchSyncScope(SyncScopeUsageTracker scope);\n"
            "    void AddDispatchSyncScopeInternal(SyncScopeUsageTracker& scope);",
            "    void AddDispatchSyncScope(SyncScopeUsageTracker scope = {});", 1)
        self.assertEqual(restored, self.header)

    def test_explicit_indirect_scope_unchanged(self):
        self.assertEqual(self.changed.count("AddDispatchSyncScope(std::move(scope));"),
                         self.original.count("AddDispatchSyncScope(std::move(scope));"))
        self.assertIn("AddDispatchSyncScopeInternal(scope);", WRAPPER)
        self.assertIn("if (inUse)", WRAPPER)
        self.assertIn("~ResetUse() { flag = false; }", WRAPPER)

    def test_each_reused_scope_is_consumed_and_cleared(self):
        tracker = SOURCE.with_name("PassResourceUsageTracker.cpp").read_text()
        for field in ("mBufferSyncInfos", "mTextureSyncInfos", "mExternalTextureUsages"):
            self.assertIn(field + ".clear();", tracker)
        self.assertIn("mUsedResourceTable = nullptr;", tracker)
        self.assertIn("mUsageTracker.AddDispatch(scope.AcquireSyncScopeUsage());", self.changed)

    def test_actual_wrapper_reentrancy_indirect_and_thread_isolation(self):
        stub = r"""
#include <cstdlib>
#include <functional>
#include <map>
#include <thread>
#include <utility>
#include <vector>
struct SyncScopeUsageTracker {
    std::map<int, int> values;
};
struct ComputePassEncoder {
    int key = 1;
    std::function<void()> callback;
    std::vector<std::map<int, int>> recorded;
    void AddDispatchSyncScope();
    void AddDispatchSyncScope(SyncScopeUsageTracker scope);
    void AddDispatchSyncScopeInternal(SyncScopeUsageTracker& scope) {
        scope.values[key] |= 1;
        if (callback) {
            auto nested = std::exchange(callback, nullptr);
            nested();
        }
        recorded.push_back(scope.values);
        scope.values.clear();
    }
};
"""
        main = r"""
void check(bool valid) { if (!valid) std::abort(); }
int main() {
    ComputePassEncoder encoder;
    encoder.callback = [&] {
        encoder.key = 2;
        encoder.AddDispatchSyncScope();
        encoder.key = 1;
    };
    encoder.AddDispatchSyncScope();
    check(encoder.recorded.size() == 2);
    check(encoder.recorded[0] == std::map<int, int>{{2, 1}});
    check(encoder.recorded[1] == std::map<int, int>{{1, 1}});
    SyncScopeUsageTracker indirect;
    indirect.values[8] = 4;
    encoder.AddDispatchSyncScope(std::move(indirect));
    check(encoder.recorded.back() == std::map<int, int>{{1, 1}, {8, 4}});
    std::vector<std::thread> threads;
    for (int i = 0; i < 8; ++i) {
        threads.emplace_back([i] {
            ComputePassEncoder local;
            for (int step = 0; step < 200; ++step) {
                local.key = 1000 * (i + 1) + step;
                local.AddDispatchSyncScope();
                check(local.recorded.back() == std::map<int, int>{{local.key, 1}});
            }
        });
    }
    for (auto& thread : threads) thread.join();
}
"""
        with tempfile.TemporaryDirectory(prefix="scope-wrapper-test-", dir=Path(__file__).parent) as tmp:
            source = Path(tmp) / "scope.cpp"
            source.write_text(stub + WRAPPER + main)
            for flag in (0, 1):
                binary = Path(tmp) / f"scope{flag}"
                subprocess.run(["c++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                                "-pthread", f"-DORT_WEBGPU_DAWN_REUSE_COMPUTE_SCOPE={flag}",
                                str(source), "-o", str(binary)], check=True)
                subprocess.run([str(binary)], check=True, timeout=10)


if __name__ == "__main__":
    unittest.main()

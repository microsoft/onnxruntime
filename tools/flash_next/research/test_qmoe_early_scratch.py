from pathlib import Path
import subprocess
import tempfile
import unittest

from build_qmoe_early_scratch import PREALLOCATE, SELECTOR, transform


SOURCE = Path("/home/sayanshaw/repos/qwen38-int4-fc2-private-a/build-v1/qmoe.cc")


class EarlyScratchTests(unittest.TestCase):
    def test_only_preallocation_and_private_selector_change(self):
        original = SOURCE.read_text()
        changed = transform(original)
        self.assertEqual(changed.replace(SELECTOR, "", 1).replace(PREALLOCATE, "", 1), original)
        self.assertLess(changed.index(PREALLOCATE), changed.index("// Step 1: Gate"))
        self.assertIn("static const int early_scratch", SELECTOR)
        self.assertIn("value == 0 || value == 1", SELECTOR)

    def test_actual_cpp_preserves_allocations_and_moves_clears_before_gate(self):
        changed = transform(SOURCE.read_text())
        start = changed.index("    std::array<std::optional<Tensor>, 5> local_scratch;")
        end = changed.index("    // Step 1: Gate", start)
        actual = SELECTOR + changed[start:end]
        actual = actual.replace(
            "tensor.emplace(context.CreateGPUTensor(type, shape));",
            "context.requested_index = static_cast<int>(index);\n"
            "        tensor.emplace(context.CreateGPUTensor(type, shape));")
        stub = r"""
#include <array>
#include <cstdint>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <string>
#include <vector>
#define ORT_ENFORCE(ok, message) do { if (!(ok)) std::abort(); } while (false)
int selected = 0;
bool first = false, second = false;
template<class T> T ParseEnvironmentVariableWithDefault(const char* name, T) {
    std::string key(name);
    if (key == "ORT_WEBGPU_QMOE_EARLY_SCRATCH") return selected;
    return key == "ORT_WEBGPU_QMOE_FC1_ACTIVATION" ? first : second;
}
using MLDataType = int;
struct TensorShape {
    std::vector<int64_t> dims;
    TensorShape(std::initializer_list<int64_t> values) : dims(values) {}
    bool operator==(const TensorShape&) const = default;
};
struct Tensor {
    int dtype;
    TensorShape shape;
    int DataType() const { return dtype; }
    const TensorShape& Shape() const { return shape; }
};
struct Context {
    std::vector<int> allocated;
    std::vector<int> clear_phase;
    int requested_index = 0, phase = 0;
    Tensor CreateGPUTensor(int dtype, const TensorShape& shape) {
        allocated.push_back(requested_index);
        clear_phase.push_back(phase);
        return {dtype, shape};
    }
};
bool CanFuseFc1Activation(Context&, bool enabled) { return enabled; }
bool CanFuseFc2FinalMix(Context&, bool enabled) { return enabled; }
void run(int rows, bool replay) {
    Context context;
    struct { int num_rows; int inter_size = 640; } moe_params{rows};
    std::array<std::optional<Tensor>, 5> saved;
    auto* scratch = replay ? &saved : nullptr;
    constexpr int dtype = 1, dtype_uint32 = 2, num_experts = 512;
    constexpr int fc1_output_size = 1280, N_fc2 = 2560, k_ = 10;
"""
        end = r"""
    (void)router_values;
    (void)indirect_experts;
    context.phase = 1;
    temporary(2, dtype, TensorShape({static_cast<int64_t>(routed_rows), fc1_output_size}));
    temporary(3, dtype, TensorShape({static_cast<int64_t>(routed_rows), moe_params.inter_size}));
    context.phase = 2;
    temporary(4, dtype, TensorShape({static_cast<int64_t>(routed_rows), N_fc2}));
    if (context.allocated != std::vector<int>{0, 1, 2, 3, 4}) std::abort();
    bool early = selected == 1 && rows == 1 && !replay && first && second;
    auto expected = early ? std::vector<int>{0, 0, 0, 0, 0} : std::vector<int>{0, 0, 1, 1, 2};
    if (context.clear_phase != expected) std::abort();
}
int main(int argc, char** argv) {
    if (argc != 2) return 1;
    selected = std::atoi(argv[1]);
    for (int rows : {1, 2}) for (bool replay : {false, true})
        for (bool f1 : {false, true}) for (bool f2 : {false, true}) {
            first = f1; second = f2; run(rows, replay);
        }
}
"""
        with tempfile.TemporaryDirectory(prefix="early-scratch-test-", dir=Path(__file__).parent) as tmp:
            source = Path(tmp) / "scratch.cpp"
            binary = Path(tmp) / "scratch"
            source.write_text(stub + actual + end)
            subprocess.run(["c++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                            str(source), "-o", str(binary)], check=True)
            for flag in (0, 1):
                subprocess.run([str(binary), str(flag)], check=True)


if __name__ == "__main__":
    unittest.main()

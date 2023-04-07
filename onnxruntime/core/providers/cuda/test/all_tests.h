// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#ifndef NDEBUG
namespace onnxruntime {
namespace cuda {
namespace test {

// Test header provides function declarations in EP-side bridge.
bool TestDeferredRelease();
bool TestDeferredReleaseWithoutArena();
bool TestBeamSearchTopK();
bool TestGreedySearchTopOne();

}  // namespace test
}  // namespace cuda
}  // namespace onnxruntime
#endif


// Custom tests are kept in a separate include to avoid regenerating the main file.

template <typename T>
class GridSampleCustomTest : public ::testing::Test {
};

using GridSampleCustomTestTypes = ::testing::Types<float, MLFloat16>;
TYPED_TEST_SUITE(GridSampleCustomTest, GridSampleCustomTestTypes);

TYPED_TEST(GridSampleCustomTest, test_grid_sample_20_4D_linear_zeros_mixed_bounds_right_bottom) {
  // Crafts grid points that mix fully in-bounds sampling with cases where either the right, bottom,
  // or both neighbors fall outside the source image so zero padding must be applied. This ensures
  // the optimized bilinear fast path matches the generic implementation for boundary handling.
  OpTester test("GridSample", 20);
  std::string mode = "linear";
  std::string padding_mode = "zeros";
  int64_t align_corners = 0;
  std::initializer_list<int64_t> X_shape{1, 1, 2, 2};
  std::initializer_list<TypeParam> X_data{TypeParam(1.0f), TypeParam(2.0f), TypeParam(3.0f), TypeParam(4.0f)};
  std::initializer_list<int64_t> Grid_shape{1, 2, 2, 2};
  // (nx, ny) pairs: center (in-bounds), right edge (x out), bottom edge (y out), corner (both out)
  std::initializer_list<TypeParam> Grid_data{
      TypeParam(0.0f), TypeParam(0.0f),   // center (all neighbors in bounds)
      TypeParam(0.9f), TypeParam(0.0f),   // near right edge (right neighbors out of bounds)
      TypeParam(0.0f), TypeParam(0.9f),   // near bottom edge (bottom neighbors out)
      TypeParam(0.9f), TypeParam(0.9f)};  // near bottom-right corner (both right and bottom neighbors out)
  std::initializer_list<int64_t> Y_shape{1, 1, 2, 2};
  std::initializer_list<TypeParam> Y_data{
      TypeParam(2.5f),    // all neighbors in bounds
      TypeParam(1.8f),    // right neighbors partially out-of-bounds
      TypeParam(2.1f),    // bottom neighbors partially out-of-bounds
      TypeParam(1.44f)};  // both right and bottom neighbors out-of-bounds
  test.AddInput<TypeParam>("X", X_shape, X_data);
  test.AddInput<TypeParam>("Grid", Grid_shape, Grid_data);
  test.AddAttribute("mode", mode);
  test.AddAttribute("padding_mode", padding_mode);
  test.AddAttribute("align_corners", align_corners);
  test.AddOutput<TypeParam>("Y", Y_shape, Y_data);
  RunTests(test, GetExecutionProviders(20));
}

TYPED_TEST(GridSampleCustomTest, test_grid_sample_20_4D_linear_zeros_mixed_bounds_left_top) {
  // Similar to test_grid_sample_20_4D_linear_zeros_mixed_bounds_right_bottom but focuses on left/top boundary cases,
  // where the left and/or top neighbors fall outside the source image and zero padding must be applied.
  // This ensures the optimized bilinear fast path correctly handles left/top boundary conditions.
  OpTester test("GridSample", 20);
  std::string mode = "linear";
  std::string padding_mode = "zeros";
  int64_t align_corners = 0;
  std::initializer_list<int64_t> X_shape{1, 1, 2, 2};
  std::initializer_list<TypeParam> X_data{TypeParam(1.0f), TypeParam(2.0f), TypeParam(3.0f), TypeParam(4.0f)};
  std::initializer_list<int64_t> Grid_shape{1, 2, 2, 2};
  // (nx, ny) pairs: center (in-bounds), left edge (x out), top edge (y out), corner (both out)
  std::initializer_list<TypeParam> Grid_data{
      TypeParam(0.0f), TypeParam(0.0f),     // center (all neighbors in bounds)
      TypeParam(-0.9f), TypeParam(0.0f),    // near left edge (left neighbors out of bounds)
      TypeParam(0.0f), TypeParam(-0.9f),    // near top edge (top neighbors out of bounds)
      TypeParam(-0.9f), TypeParam(-0.9f)};  // near top-left corner (both left and top neighbors out of bounds)
  std::initializer_list<int64_t> Y_shape{1, 1, 2, 2};
  std::initializer_list<TypeParam> Y_data{
      TypeParam(2.5f),    // all neighbors in bounds
      TypeParam(1.2f),    // left neighbors partially out-of-bounds
      TypeParam(0.9f),    // top neighbors partially out-of-bounds
      TypeParam(0.36f)};  // both left and top neighbors out-of-bounds
  test.AddInput<TypeParam>("X", X_shape, X_data);
  test.AddInput<TypeParam>("Grid", Grid_shape, Grid_data);
  test.AddAttribute("mode", mode);
  test.AddAttribute("padding_mode", padding_mode);
  test.AddAttribute("align_corners", align_corners);
  test.AddOutput<TypeParam>("Y", Y_shape, Y_data);
  RunTests(test, GetExecutionProviders(20));
}


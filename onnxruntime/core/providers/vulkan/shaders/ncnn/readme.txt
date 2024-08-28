Testing Kompute with shader used by NCNN for our test MatMul model

The preamble is copied from the string assembled in gpu.cpp:compile_spirv_module in NCNN.
The shader was chosen on the NCNN selection logic in InnerProduct_vulkan.cpp for the test model.

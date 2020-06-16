//
//  Use this file to import your target's public headers that you would like to expose to Swift.
//

#include "OnnxInterop.hpp"
typedef void *mnist;


mnist *mnist_new ();
float *mnist_get_input_image (mnist *_mnist, size_t *out);
float *mnist_get_results (mnist *_mnist, size_t *out);
long mnist_run (mnist *_mnist);

//
//  OnnxInterop.hpp
//  SwiftMnist
//
//  Created by Miguel de Icaza on 6/1/20.
//  Copyright © 2020 Miguel de Icaza. All rights reserved.
//

#ifndef OnnxInterop_hpp
#define OnnxInterop_hpp

#include <stdio.h>

typedef void *mnist;

mnist *mnist_new ();
float *mnist_get_input_image (mnist *_mnist, size_t *out);
float *mnist_get_results (mnist *_mnist, size_t *out);
long mnist_run (mnist *_mnist);

#endif /* OnnxInterop_hpp */

// /*++

// Copyright (c) Microsoft Corporation. All rights reserved.

// Licensed under the MIT License.

// Module Name:

//     test_sqlutgemm.h

// Abstract:

//     Tests for MLAS T-MAC quantized GEMM.

// --*/

// #include "test_util.h"
// #include "mlas_q4.h"
// #include "mlas_qnbit.h"


// static size_t MlasQLUTGemmTestAllShortExecuteTests() {
//   size_t tests_registered = 0;

//   tests_registered += MlasQLUTGemmShortExecuteTest<2,16>::RegisterShortExecuteTests();
//   tests_registered += MlasQLUTGemmShortExecuteTest<2,32>::RegisterShortExecuteTests();
//   tests_registered += MlasQLUTGemmShortExecuteTest<2,64>::RegisterShortExecuteTests();
//   tests_registered += MlasQLUTGemmShortExecuteTest<2,128>::RegisterShortExecuteTests();
//   tests_registered += MlasQLUTGemmShortExecuteTest<2,256>::RegisterShortExecuteTests();

//   tests_registered += MlasQLUTGemmShortExecuteTest<4,16>::RegisterShortExecuteTests();
//   tests_registered += MlasQLUTGemmShortExecuteTest<4,32>::RegisterShortExecuteTests();
//   tests_registered += MlasQLUTGemmShortExecuteTest<4,64>::RegisterShortExecuteTests();
//   tests_registered += MlasQLUTGemmShortExecuteTest<4,128>::RegisterShortExecuteTests();
//   tests_registered += MlasQLUTGemmShortExecuteTest<4,256>::RegisterShortExecuteTests();

//   return tests_registered;
// }

// static UNUSED_VARIABLE bool added_to_main = AddTestRegister([](bool is_short_execute) {
//   if (is_short_execute) {
//     return MlasQLUTGemmTestAllShortExecuteTests();
//   }
//   return 0;
// });

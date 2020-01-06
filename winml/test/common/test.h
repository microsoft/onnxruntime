#define BUILD_GOOGLE_TEST
#ifdef BUILD_GOOGLE_TEST
#include "googleTestMacros.h"
#elif BUILD_TAEF_TEST
#include "taefTestMacros.h"
#endif
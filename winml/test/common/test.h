using VoidTest = void (*)();
using SetupTest = VoidTest;

#ifdef BUILD_GOOGLE_TEST
#include "googleTestMacros.h"
#else
#ifdef BUILD_TAEF_TEST
#include "taefTestMacros.h"
#endif
#endif
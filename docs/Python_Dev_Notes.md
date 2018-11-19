# Python Dev Notes

Each Python version uses a specific compiler version. In most cases, you should use the same compiler version for building python extensions.

## Which Microsoft Visual C++ compiler to use with a specific Python version ?

| Visual C++  | CPython                 |
|-------------|:-----------------------:|
|2015, 2017   | 3.7                     |
|2015         | 3.5,3.6                 |
|2010         | 3.3,3.4                 |
|2008         | 2.6, 2.7, 3.0, 3.1, 3.2 |

Currently, ONNXRuntime only supports Visual C++ 2017. Therefore, Python 3.7 seems to be the best choice.

CPython 3.7 is distributed with a VC++ 2017 runtime. Unlike the earlier VC++ version, VC++ 2017 Runtime is binary backward compatible with VC++ 2015. Which means you could build your application with VC++ 2015 then run it with VC++ 2017 runtime.

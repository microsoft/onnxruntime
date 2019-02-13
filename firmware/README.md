Example projects
----------------

The examples directory contains the following projects:

- loopback
  Schema and source code of an example BrainSlice firmware.
- emulator
  Project to build the firmware as x86 Dll which can be run in the emulator.
- firmware
  Project to build the firmware as Nios program which can be run in the
  emulator or on Fpga.
- client
  Project to build a Dll with C APIs for the firmware functions.
- python
  Project to build Python extension module with APIs for the firmware
  functions.

The `emulator`, `firmware`, `client` and `python` projects are not firmware
specific. They can be easily reused to build any firmware by changing the
`firmware.props` configuration file.

See `docs\BrainSlice-API-Design.md` for more information about firmware APIs.

Building examples
-----------------

The example projects require the following properties to find location of the
dependencies:

    MSBuild property            Dependency
    --------------------------------------
    PkgBond_Fpga                Bond.Fpga
    PkgBond_Fpga_Compiler       Bond.Fpga.Compiler
    PkgBond_Cpp                 Bond.Cpp
    Pkgboost                    boost
    PkgCatapult_HaaS_Rtl        Catapult.HaaS.Rtl
    BrainSliceNiosEds           NIOS tools (e.g. nios2eds in Quartus package)
    pybind11                    https://github.com/pybind/pybind11
    python                      Python (e.g. Anaconda package)
    python_version              27 or 36

If not already defined by the build environment these properties can be
provided in one of the following ways:

- Specified in msbuild command line arguments using /property flag(s)
- Defined as environment variables
- Defined within PropertyGroup in a .props file pointed to by `EnvironmentConfig`
  environment variable or property.

Running examples:
-----------------

The example firmware can be executed in emulator or on FPGA using the Python
script:

    python.exe examples\loopback\test_loopback.py

The script depends on Python extension modules included in `lib\native\python36`
subdirectory of the DevKit and on the extension module for the example firmware
built by `examples\python`. Python interpreter uses `PYTHONPATH` environment
variable to find extensions and it should be configured to point to correct
directories before running the script, e.g.:

    set PYTHONPATH=lib\native\python36;examples\python\objd\amd64

Note that the debug/release build configurations must be consistent between
binaries from DevKit and those built locally, e.g. when using debug build of
firmware Python APIs use debug flavour of the DevKit.

The test can be configured via global variables in the script which specify
paths to compiled firmware files and whether the firmware should execute on FPGA
device running BrainSlice bitstream or in emulator. By default Nios build of the
firmware is executed in emulator.

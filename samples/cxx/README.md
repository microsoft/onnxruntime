# ONNX Runtime C++ Sample

A minimal C++ program demonstrating basic ONNX Runtime inference. It loads an ONNX model that adds two float tensors (`C = A + B`), runs inference, and verifies the result.

## Prerequisites

- CMake 3.28 or later
- C++20 compatible compiler (e.g., Visual Studio 2022)
- An ONNX Runtime release package (download from [GitHub releases](https://github.com/microsoft/onnxruntime/releases))
- For model generation:
  - Python with the `onnx` package

## Directory Structure

```
samples/cxx/
├── CMakeLists.txt         # Build configuration
├── main.cc                # Sample program source
├── add_model.onnx         # ONNX model (C = A + B)
├── generate_model.py      # Script to generate the ONNX model
└── README.md              # This file
```

## Steps

### 1. Extract the ONNX Runtime package

Download and extract an ONNX Runtime release archive. For example:

```
tar -xf onnxruntime-win-x64-1.25.0.zip
```

This creates a directory like `onnxruntime-win-x64-1.25.0/` containing `include/` and `lib/` subdirectories.

### 2. [Optional] Generate the ONNX model

```
cd samples/cxx
pip install onnx
python generate_model.py
```

This creates `add_model.onnx` in the current directory.

### 3. Configure and build

From the `samples/cxx` directory:

**Windows:**
```
cmake -S . -B build ^
    -DORT_HEADER_DIR:PATH=path\to\onnxruntime-win-x64-1.25.0\include ^
    -DORT_LIBRARY_DIR:PATH=path\to\onnxruntime-win-x64-1.25.0\lib
cmake --build build --config Release
```

**Linux / macOS:**
```
cmake -S . -B build \
    -DORT_HEADER_DIR:PATH=path/to/onnxruntime-linux-x64-1.25.0/include \
    -DORT_LIBRARY_DIR:PATH=path/to/onnxruntime-linux-x64-1.25.0/lib
cmake --build build --config Release
```

Adjust the paths to match your extracted package name and location.

The build automatically copies the ONNX Runtime shared libraries next to the executable.

#### CMake Variables

| Variable | Description |
|---|---|
| `ORT_HEADER_DIR` | Path to the ONNX Runtime `include` directory |
| `ORT_LIBRARY_DIR` | Path to the ONNX Runtime `lib` directory |

### 4. Run

**Windows:**
```
build\Release\onnxruntime_sample_program.exe
```

**Linux / macOS:**
```
./build/onnxruntime_sample_program
```

You can also pass a model path as an argument:
```
onnxruntime_sample_program path/to/add_model.onnx
```

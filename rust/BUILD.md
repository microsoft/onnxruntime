# Building and testing the Rust bindings

These instructions require cargo and rustc.
To get these follow the instructions at [https://rustup.rs](https://rustup.rs)
The instructions compile the onnxruntime along with the bindings,
so require `cmake`, a python 3 interpreter, clang (needed to parse the C headers to generate the Rust bindings),
and the platform compiler to compile onnxruntime.

## Local setup of onnxruntime repo

```sh
    git clone https://github.com/microsoft/onnxruntime
    cd onnxruntime
    git submodule update --init --recursive
```

## cargo build both crates

from the root of onnxruntime repo

```sh
    CARGO_TARGET_DIR=build/rust cargo build --manifest-path rust/Cargo.toml
```

The CARGO_TARGET_DIR environment variable puts the build artifacts in `onnxruntime/build/rust`
instead of `onnxruntime/rust/target`.

## cargo test both crates

```sh
    CARGO_TARGET_DIR=build/rust cargo test --manifest-path rust/Cargo.toml --features model-fetching
```

## cargo test with sanitizer support

**If you are using a nightly Rust compiler and are on one the platforms listed in [Rust sanitizer support](https://doc.rust-lang.org/beta/unstable-book/compiler-flags/sanitizer.html).**

where `$SAN` is one of `address`, `thread`, `memory` or `leak`

```sh
    RUSTFLAGS="-Zsanitizer=$SAN" CARGO_TARGET_DIR=build/rust cargo test --manifest-path rust/Cargo.toml --features model-fetching --target <your target for example x86_64-unknown-linux-gnu> -Z build-std -- --test-threads=1
```

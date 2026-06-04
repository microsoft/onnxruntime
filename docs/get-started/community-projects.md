---
title: Julia, Ruby and Rust APIs
description: Get started with APIs for Julia, Ruby and Rust
parent: Get Started
nav_order: 8
has_toc: false
---

# Julia and Ruby APIs

* [Julia](https://github.com/jw3126/ONNXRunTime.jl) (external project)
* [Ruby](https://github.com/ankane/onnxruntime) (external project)

# Rust API

ONNX Runtime provides an official Rust API. The bindings are available via the [`ort` crate](https://crates.io/crates/ort), which is maintained by [pykeio](https://github.com/pykeio/ort) and integrates with the official ONNX Runtime libraries.

```toml
[dependencies]
ort = "2"
```

See the [ort documentation](https://ort.pyke.io/) for usage examples and a full API reference.

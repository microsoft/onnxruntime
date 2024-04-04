#![allow(dead_code)]

use std::{
    borrow::Cow,
    env, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    str::FromStr,
    process
};

// use cmake::build;

use anyhow::{anyhow, Context, Result};

/// ONNX Runtime version
///
/// WARNING: If version is changed, bindings for all platforms will have to be re-generated.
///          To do so, run this:
///              cargo build --package onnxruntime-sys --features generate-bindings
const ORT_VERSION: &str = include_str!("./vendor/onnxruntime-src/VERSION_NUMBER");

/// Base Url from which to download pre-built releases/
const ORT_RELEASE_BASE_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download";

/// Environment variable selecting which strategy to use for finding the library
/// Possibilities:
/// * "download": Download a pre-built library. This is the default if `ORT_STRATEGY` is not set.
/// * "system": Use installed library. Use `ORT_LIB_LOCATION` to point to proper location.
/// * "compile": Download source and compile (TODO).
const ORT_RUST_ENV_STRATEGY: &str = "ORT_RUST_STRATEGY";

/// Name of environment variable that, if present, contains the location of a pre-built library.
/// Only used if `ORT_STRATEGY=system`.
const ORT_RUST_ENV_SYSTEM_LIB_LOCATION: &str = "ORT_RUST_LIB_LOCATION";
/// Name of environment variable that, if present, controls whether to use CUDA or not.
const ORT_RUST_ENV_GPU: &str = "ORT_RUST_USE_CUDA";

/// Subdirectory (of the 'target' directory) into which to extract the prebuilt library.
const ORT_PREBUILT_EXTRACT_DIR: &str = "onnxruntime";

fn main() -> Result<()> {
    let libort_install_dir = prepare_libort_dir().context("preparing libort directory")?;

    let include_dir = libort_install_dir.join("include");
    let lib_dir = libort_install_dir.join("lib");

    println!("Include directory: {:?}", include_dir);
    println!("Lib directory: {:?}", lib_dir);

    // Tell cargo to tell rustc to link onnxruntime shared library.
    // println!("cargo:rustc-link-lib=onnxruntime");
    // std::process::exit(-1);

    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    for lib in &["common", "flatbuffers", "framework", "graph", "mlas", "optimizer", "providers", "session", "util"] {
		println!("{0}", lib_dir.display());
		let lib_path = lib_dir.join(if cfg!(target_os = "windows") {
			format!("onnxruntime_{lib}.lib")
		} else {
			format!("libonnxruntime_{lib}.a")
		});
		// sanity check, just make sure the library exists before we try to link to it
		if lib_path.exists() {
			println!("cargo:rustc-link-lib=static=onnxruntime_{lib}");
		} else {
			panic!("[ort] unable to find ONNX Runtime library: {}", lib_path.display());
		}
	}

    println!("cargo:rerun-if-env-changed={}", ORT_RUST_ENV_STRATEGY);
    println!("cargo:rerun-if-env-changed={}", ORT_RUST_ENV_GPU);
    println!(
        "cargo:rerun-if-env-changed={}",
        ORT_RUST_ENV_SYSTEM_LIB_LOCATION
    );

    generate_bindings(&include_dir);
    Ok(())
}

#[cfg(not(feature = "generate-bindings"))]
fn generate_bindings(_include_dir: &Path) {
    println!("Bindings not generated automatically, using committed files instead.");
    println!("Enable with the 'generate-bindings' cargo feature.");

    // NOTE: If bindings could not be be generated for Apple Sillicon M1, please uncomment the following
    // let os = env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    // let arch = env::var("CARGO_CFG_TARGET_ARCH").expect("Unable to get TARGET_ARCH");
    // if os == "macos" && arch == "aarch64" {
    //     panic!(
    //         "OnnxRuntime {} bindings for Apple M1 are not available",
    //         ORT_VERSION
    //     );
    // }
}

#[cfg(feature = "generate-bindings")]
fn generate_bindings(include_dir: &Path) {
    let clang_args = &[
        format!("-I{}", include_dir.display()),
        format!(
            "-I{}",
            include_dir
                .join("onnxruntime")
                .join("core")
                .join("session")
                .display()
        ),
    ];

    let path = include_dir.join("onnxruntime").join("onnxruntime_c_api.h");

    // The bindgen::Builder is the main entry point
    // to bindgen, and lets you build up options for
    // the resulting bindings.
    let bindings = bindgen::Builder::default()
        // The input header we would like to generate
        // bindings for.
        .header(path.to_string_lossy().to_string())
        // The current working directory is 'onnxruntime-sys'
        .clang_args(clang_args)
        // Tell cargo to invalidate the built crate whenever any of the
        // included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        .dynamic_library_name("onnxruntime")
        .allowlist_type("Ort.*")
        .allowlist_type("Onnx.*")
        .allowlist_type("ONNX.*")
        .allowlist_function("Ort.*")
        .allowlist_var("ORT.*")
        // Set `size_t` to be translated to `usize` for win32 compatibility.
        .size_t_is_usize(true)
        // Format using rustfmt
        .rustfmt_bindings(true)
        .rustified_enum(".*")
        // Finish the builder and generate the bindings.
        .generate()
        // Unwrap the Result and panic on failure.
        .expect("Unable to generate bindings");

    //let generated_file = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    // Write the bindings to (source controlled) src/generated/<os>/<arch>/bindings.rs
    let generated_file = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
        .join("src")
        .join("generated")
        .join(env::var("CARGO_CFG_TARGET_OS").unwrap())
        .join(env::var("CARGO_CFG_TARGET_ARCH").unwrap())
        .join("bindings.rs");
    println!("cargo:rerun-if-changed={:?}", generated_file);
    bindings
        .write_to_file(generated_file)
        .expect("Couldn't write bindings!");
}

fn download<P>(source_url: &str, target_file: P)
where
    P: AsRef<Path>,
{
    let resp = ureq::get(source_url)
        .timeout(std::time::Duration::from_secs(300))
        .call()
        .unwrap_or_else(|err| panic!("ERROR: Failed to download {}: {:?}", source_url, err));

    let len = resp
        .header("Content-Length")
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap();
    let mut reader = resp.into_reader();
    // FIXME: Save directly to the file
    let mut buffer = vec![];
    let read_len = reader.read_to_end(&mut buffer).unwrap();
    assert_eq!(buffer.len(), len);
    assert_eq!(buffer.len(), read_len);

    let f = fs::File::create(&target_file).unwrap();
    let mut writer = io::BufWriter::new(f);
    writer.write_all(&buffer).unwrap();
}

fn extract_archive(filename: &Path, output: &Path) {
    match filename.extension().map(std::ffi::OsStr::to_str) {
        Some(Some("zip")) => extract_zip(filename, output),
        Some(Some("tgz")) => extract_tgz(filename, output),
        _ => unimplemented!(),
    }
}

fn extract_tgz(filename: &Path, output: &Path) {
    let file = fs::File::open(filename).unwrap();
    let buf = io::BufReader::new(file);
    let tar = flate2::read::GzDecoder::new(buf);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(output).unwrap();
}

fn extract_zip(filename: &Path, outpath: &Path) {
    let file = fs::File::open(filename).unwrap();
    let buf = io::BufReader::new(file);
    let mut archive = zip::ZipArchive::new(buf).unwrap();
    for i in 0..archive.len() {
        let mut file = archive.by_index(i).unwrap();
        #[allow(deprecated)]
        let outpath = outpath.join(file.sanitized_name());
        if !file.name().ends_with('/') {
            println!(
                "File {} extracted to \"{}\" ({} bytes)",
                i,
                outpath.as_path().display(),
                file.size()
            );
            if let Some(p) = outpath.parent() {
                if !p.exists() {
                    fs::create_dir_all(p).unwrap();
                }
            }
            let mut outfile = fs::File::create(&outpath).unwrap();
            io::copy(&mut file, &mut outfile).unwrap();
        }
    }
}

trait OnnxPrebuiltArchive {
    fn as_onnx_str(&self) -> Cow<str>;
}

#[derive(Debug)]
enum Architecture {
    X86,
    X86_64,
    Arm,
    Arm64,
}

impl FromStr for Architecture {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "x86" => Ok(Architecture::X86),
            "x86_64" => Ok(Architecture::X86_64),
            "arm" => Ok(Architecture::Arm),
            "aarch64" => Ok(Architecture::Arm64),
            _ => Err(anyhow!("Unsupported architecture: {s}")),
        }
    }
}

impl OnnxPrebuiltArchive for Architecture {
    fn as_onnx_str(&self) -> Cow<str> {
        match self {
            Architecture::X86 => Cow::from("x86"),
            Architecture::X86_64 => Cow::from("x64"),
            Architecture::Arm => Cow::from("arm"),
            Architecture::Arm64 => Cow::from("arm64"),
        }
    }
}

#[derive(Debug)]
#[allow(clippy::enum_variant_names)]
enum Os {
    Windows,
    Linux,
    MacOs,
    IOs,
}

impl Os {
    fn archive_extension(&self) -> &'static str {
        match self {
            Os::Windows => "zip",
            Os::Linux | Os::MacOs | Os::IOs => "tgz",
        }
    }
}

impl FromStr for Os {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "windows" => Ok(Os::Windows),
            "macos" => Ok(Os::MacOs),
            "linux" => Ok(Os::Linux),
            "ios" => Ok(Os::IOs),
            _ => Err(format!("Unsupported OS: {}", s)),
        }
    }
}

// Check is simulator
#[derive(Debug)]
#[allow(clippy::enum_variant_names)]
enum Simulator {
    Yes,
    No,
}

impl FromStr for Simulator {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "aarch64-apple-ios-sim" => Ok(Simulator::Yes),
            "x86_64-apple-ios" => Ok(Simulator::Yes),
            "aarch64-apple-ios" | "aarch64-apple-ios-macabi" | "x86_64-apple-ios-macabi" | "x86_64-apple-darwin" | "aarch64-apple-darwin" => Ok(Simulator::No),
            _ => Err(format!("Unsupported OS: {}", s)),
        }
    }
}

impl OnnxPrebuiltArchive for Os {
    fn as_onnx_str(&self) -> Cow<str> {
        match self {
            Os::Windows => Cow::from("win"),
            Os::Linux => Cow::from("linux"),
            Os::MacOs => Cow::from("osx"),
            Os::IOs => Cow::from("ios"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum ABI {
    Macabi,
    None,
}

impl FromStr for ABI {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "macabi" => Ok(ABI::Macabi),
            _ => Ok(ABI::None),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Accelerator {
    Cpu,
    Cuda,
}

impl FromStr for Accelerator {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        match s.to_lowercase().as_str() {
            "1" | "yes" | "true" | "on" => Ok(Accelerator::Cuda),
            _ => Ok(Accelerator::Cpu),
        }
    }
}

impl OnnxPrebuiltArchive for Accelerator {
    fn as_onnx_str(&self) -> Cow<str> {
        match self {
            Accelerator::Cpu => Cow::from(""),
            Accelerator::Cuda => Cow::from("gpu"),
        }
    }
}

#[derive(Debug)]
struct Triplet {
    os: Os,
    arch: Architecture,
    accelerator: Accelerator,
}

impl OnnxPrebuiltArchive for Triplet {
    fn as_onnx_str(&self) -> Cow<str> {
        match (&self.os, &self.arch, &self.accelerator) {
            // onnxruntime-win-x86-1.11.1.zip
            // onnxruntime-win-x64-1.11.1.zip
            // onnxruntime-win-arm-1.11.1.zip
            // onnxruntime-win-arm64-1.11.1.zip
            // onnxruntime-linux-x64-1.11.1.tgz
            // onnxruntime-osx-x86_64-1.11.1.tgz
            // onnxruntime-osx-arm64-1.11.1.tgz
            (
                Os::Windows,
                Architecture::X86 | Architecture::X86_64 | Architecture::Arm | Architecture::Arm64,
                Accelerator::Cpu,
            )
            | (Os::MacOs, Architecture::Arm64, Accelerator::Cpu)
            | (Os::Linux, Architecture::X86_64, Accelerator::Cpu) => Cow::from(format!(
                "{}-{}",
                self.os.as_onnx_str(),
                self.arch.as_onnx_str()
            )),
            (Os::MacOs, Architecture::X86_64, Accelerator::Cpu) => Cow::from(format!(
                "{}-x86_{}",
                self.os.as_onnx_str(),
                self.arch.as_onnx_str().trim_start_matches('x')
            )),
            // onnxruntime-win-x64-gpu-1.11.1.zip
            // onnxruntime-linux-x64-gpu-1.11.1.tgz
            (Os::Linux | Os::Windows, Architecture::X86_64, Accelerator::Cuda) => {
                Cow::from(format!(
                    "{}-{}-{}",
                    self.os.as_onnx_str(),
                    self.arch.as_onnx_str(),
                    self.accelerator.as_onnx_str(),
                ))
            }
            _ => {
                panic!(
                    "Unsupported prebuilt triplet: {:?}, {:?}, {:?}. Please use {}=system and {}=/path/to/onnxruntime",
                    self.os, self.arch, self.accelerator, ORT_RUST_ENV_STRATEGY, ORT_RUST_ENV_SYSTEM_LIB_LOCATION
                );
            }
        }
    }
}

fn prebuilt_archive_url() -> (PathBuf, String) {
    let triplet = Triplet {
        os: env::var("CARGO_CFG_TARGET_OS")
            .expect("Unable to get TARGET_OS")
            .parse()
            .unwrap(),
        arch: env::var("CARGO_CFG_TARGET_ARCH")
            .expect("Unable to get TARGET_ARCH")
            .parse()
            .unwrap(),
        accelerator: env::var(ORT_RUST_ENV_GPU)
            .unwrap_or_default()
            .parse()
            .unwrap(),
    };

    let prebuilt_archive = format!(
        "onnxruntime-{}-{}.{}",
        triplet.as_onnx_str(),
        ORT_VERSION,
        triplet.os.archive_extension()
    );
    let prebuilt_url = format!(
        "{}/v{}/{}",
        ORT_RELEASE_BASE_URL, ORT_VERSION, prebuilt_archive
    );

    (PathBuf::from(prebuilt_archive), prebuilt_url)
}

fn prepare_libort_dir_prebuilt() -> PathBuf {
    let (prebuilt_archive, prebuilt_url) = prebuilt_archive_url();

    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
    let extract_dir = out_dir.join(ORT_PREBUILT_EXTRACT_DIR);
    let downloaded_file = out_dir.join(&prebuilt_archive);

    println!("cargo:rerun-if-changed={}", downloaded_file.display());

    if !downloaded_file.exists() {
        println!("Creating directory {:?}", out_dir);
        fs::create_dir_all(&out_dir).unwrap();

        println!(
            "Downloading {} into {}",
            prebuilt_url,
            downloaded_file.display()
        );
        download(&prebuilt_url, &downloaded_file);
    }

    if !extract_dir.exists() {
        println!("Extracting to {}...", extract_dir.display());
        extract_archive(&downloaded_file, &extract_dir);
    }

    extract_dir.join(prebuilt_archive.file_stem().unwrap())
}

fn prepare_libort_dir() -> Result<PathBuf> {
    let strategy = env::var(ORT_RUST_ENV_STRATEGY);
    println!(
        "strategy: {:?}",
        strategy.as_ref().map_or_else(|_| "unknown", String::as_str)
    );
    match strategy.as_ref().map(String::as_str) {
        Ok("download") => Ok(prepare_libort_dir_prebuilt()),
        Ok("system") => {
            let location = env::var(ORT_RUST_ENV_SYSTEM_LIB_LOCATION).context(format!(
                "Could not get value of environment variable {:?}",
                ORT_RUST_ENV_SYSTEM_LIB_LOCATION
            ))?;
            Ok(PathBuf::from(location))
        }
        Ok("compile") | Err(_) => prepare_libort_dir_compiled(),
        _ => Err(anyhow!("Unknown value for {:?}", ORT_RUST_ENV_STRATEGY)),
    }
}

// Prepare cmake config
fn prepare_cmake_config(mut config: cmake::Config) -> cmake::Config {
    print!("{:?}", env::var("CARGO_CFG_TARGET_ABI"));
    for (key, value) in env::vars() {
        println!("{key}: {value}");
    }
    let target_os: Os = env::var("CARGO_CFG_TARGET_OS")
                    .expect("Unable to get TARGET_OS")
                    .parse()
                    .unwrap();

    let target_abi: ABI = match env::var("CARGO_CFG_TARGET_ABI") {
        Ok(val) => val.parse().unwrap(),
        Err(_e) => ABI::None,
    };
    let target_simulator: Simulator = env::var("TARGET")
                                    .expect("Unable to get TARGET")
                                    .parse()
                                    .unwrap();

    let target_arch: Architecture = env::var("CARGO_CFG_TARGET_ARCH")
                                        .expect("Unable to get TARGET_ARCH")
                                        .parse()
                                        .unwrap();

    let triplet = Triplet {
        os: env::var("CARGO_CFG_TARGET_OS")
            .expect("Unable to get TARGET_OS")
            .parse()
            .unwrap(),
        arch: env::var("CARGO_CFG_TARGET_ARCH")
            .expect("Unable to get TARGET_ARCH")
            .parse()
            .unwrap(),
        accelerator: env::var(ORT_RUST_ENV_GPU)
            .unwrap_or_default()
            .parse()
            .unwrap(),
    };
    print!("Paco: prebuilt triplet: {:?}, {:?}, {:?}. Please use {}=system and {}=/path/to/onnxruntime",
    triplet.os, triplet.arch, triplet.accelerator, ORT_RUST_ENV_STRATEGY, ORT_RUST_ENV_SYSTEM_LIB_LOCATION);
    //aarch64-apple-ios-sim
    match (target_os, target_abi, target_simulator) {
        (Os::Windows, ABI::Macabi | ABI::None, _) => {

        },
        (Os::MacOs, ABI::Macabi | ABI::None, _) => {
            config.define("onnxruntime_BUILD_SHARED_LIB", "ON");
            config.define("CMAKE_SYSTEM_NAME", "Darwin");
            config.define("onnxruntime_BUILD_SHARED_LIB", "ON");
            config.define("PYTHON_EXECUTABLE", "/usr/bin/python3");
            config.define("CMAKE_OSX_SYSROOT", "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX13.0.sdk");
            config.define("CMAKE_THREAD_LIBS_INIT", "-lpthread");
            config.define("CMAKE_HAVE_THREADS_LIBRARY", "1");
            config.define("CMAKE_USE_WIN32_THREADS_INIT", "0");
            config.define("CMAKE_USE_PTHREADS_INIT", "1");
            config.define("THREADS_PREFER_PTHREAD_FLAG", "ON");
            config.define("onnxruntime_BUILD_UNIT_TESTS", "OFF");
            config.define("onnxruntime_BUILD_APPLE_FRAMEWORK", "ON");
            config.define("onnxruntime_ENABLE_BITCODE", "OFF");

            match target_arch {
                Architecture::X86 | Architecture::X86_64 => {
                    config.define("CMAKE_OSX_ARCHITECTURES", "x86_64");
                },
                Architecture::Arm | Architecture::Arm64 => {
                    config.define("CMAKE_OSX_ARCHITECTURES", "arm64");
                }
            }

            config.cxxflag("-I/opt/homebrew/opt/flatbuffers/include -I/Users/goodnotesci/goodnotes/gn_onnx/third_party/protobuf-21.12/src");

        },
        (Os::Linux, ABI::Macabi | ABI::None, _) => {

        },
        (Os::IOs, ABI::None, Simulator::Yes) => {
            println!("Running IOS + No ABI");
            config.define("CMAKE_SYSTEM_NAME", "ios");
            config.define("CMAKE_TOOLCHAIN_FILE", "../cmake/onnxruntime_ios.toolchain.cmake");
            //config.define("onnxruntime_BUILD_SHARED_LIB", "ON");
            config.define("PYTHON_EXECUTABLE", "/usr/bin/python3");
            match target_arch {
                Architecture::X86 | Architecture::X86_64 => {
                    config.define("CMAKE_OSX_ARCHITECTURES", "x86_64");
                },
                Architecture::Arm | Architecture::Arm64 => {
                    config.define("CMAKE_OSX_ARCHITECTURES", "arm64");
                }
            }

            config.define("CMAKE_VERBOSE_MAKEFILE", "ON");
            config.define("CMAKE_EXPORT_COMPILE_COMMANDS", "ON");
            //config.env("IPHONEOS_DEPLOYMENT_TARGET", "13.0");
            env::set_var("IPHONEOS_DEPLOYMENT_TARGET", "13.0");

            config.define("CMAKE_OSX_DEPLOYMENT_TARGET", "13.0");
            //config.define("CMAKE_OSX_SYSROOT", "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk");
            config.define("CMAKE_OSX_SYSROOT", "iphonesimulator");
            config.define("CMAKE_THREAD_LIBS_INIT", "-lpthread");
            config.define("CMAKE_HAVE_THREADS_LIBRARY", "1");
            config.define("CMAKE_USE_WIN32_THREADS_INIT", "0");
            config.define("CMAKE_USE_PTHREADS_INIT", "1");
            config.define("THREADS_PREFER_PTHREAD_FLAG", "ON");
            config.define("onnxruntime_BUILD_UNIT_TESTS", "OFF");
            config.define("onnxruntime_BUILD_APPLE_FRAMEWORK", "ON");
            config.define("onnxruntime_ENABLE_BITCODE", "ON");
            config.cxxflag("-I/opt/homebrew/opt/flatbuffers/include -I/Users/goodnotesci/goodnotes/gn_onnx/third_party/protobuf-21.12/src --target=arm64-apple-ios14.0-simulator -D__thread= -mios-simulator-version-min=13.0");
            //config.define("__thread", ""); //wrong
            //add_definitions(-Dfoo=5)

            // Use local protobuf as the protoc compiled via cmake crashes immeidately.
            config.define("ONNX_CUSTOM_PROTOC_EXECUTABLE", "/usr/local/bin/protoc-3.21.12.0");
            config.define("PROTOBUF_PROTOC_EXECUTABLE", "/usr/local/bin/protoc-3.21.12.0");
            config.define("onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS", "OFF");
            config.define("protobuf_BUILD_PROTOC_BINARIES", "OFF");
            config.define("onnxruntime_BUILD_SHARED_LIB", "OFF");
        },
        (Os::IOs, ABI::None, Simulator::No) => {
            println!("Running IOS + No ABI");
            config.define("CMAKE_SYSTEM_NAME", "iOS");
            config.define("CMAKE_TOOLCHAIN_FILE", "../cmake/onnxruntime_ios.toolchain.cmake");
            //config.define("onnxruntime_BUILD_SHARED_LIB", "ON");
            config.define("PYTHON_EXECUTABLE", "/usr/bin/python3");

            config.define("CMAKE_OSX_DEPLOYMENT_TARGET", "11.0");
            config.define("CMAKE_OSX_SYSROOT", "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk");
            config.define("CMAKE_THREAD_LIBS_INIT", "-lpthread");
            config.define("CMAKE_HAVE_THREADS_LIBRARY", "1");
            config.define("CMAKE_USE_WIN32_THREADS_INIT", "0");
            config.define("CMAKE_USE_PTHREADS_INIT", "1");
            config.define("THREADS_PREFER_PTHREAD_FLAG", "ON");
            config.define("onnxruntime_BUILD_UNIT_TESTS", "OFF");
            config.define("onnxruntime_BUILD_APPLE_FRAMEWORK", "ON");
            config.define("onnxruntime_ENABLE_BITCODE", "ON");
            config.cxxflag("-I/opt/homebrew/opt/flatbuffers/include -I/Users/goodnotesci/goodnotes/gn_onnx/third_party/protobuf-21.12/src");

            // Use local protobuf as the protoc compiled via cmake crashes immeidately.
            config.define("ONNX_CUSTOM_PROTOC_EXECUTABLE", "/usr/local/bin/protoc-3.21.12.0");
            config.define("PROTOBUF_PROTOC_EXECUTABLE", "/usr/local/bin/protoc-3.21.12.0");
            config.define("onnxruntime_ENABLE_EXTERNAL_CUSTOM_OP_SCHEMAS", "OFF");
            config.define("protobuf_BUILD_PROTOC_BINARIES", "OFF");
            config.define("onnxruntime_BUILD_SHARED_LIB", "OFF");
        },
        (Os::IOs, ABI::Macabi, _) => {
            // rustup default nightly-2022-08-03 && cargo build --target aarch64-apple-ios-macabi --verbose
            // rustup default nightly-2022-08-03 && cargo build -Z build-std=panic_abort,std --target aarch64-apple-ios-macabi
            // rustup default nightly-2022-08-03 && cargo build -Z build-std=panic_abort,std --target x86_64-apple-ios-macabi
            println!("Running IOS + MacABI");
            config.define("CMAKE_SYSTEM_NAME", "Darwin"); //for Catalyst
            config.define("onnxruntime_BUILD_SHARED_LIB", "ON");
            config.define("PYTHON_EXECUTABLE", "/usr/bin/python3");
            config.define("CMAKE_OSX_SYSROOT", "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX13.0.sdk");
            config.define("CMAKE_THREAD_LIBS_INIT", "-lpthread");
            config.define("CMAKE_HAVE_THREADS_LIBRARY", "1");
            config.define("CMAKE_USE_WIN32_THREADS_INIT", "0");
            config.define("CMAKE_USE_PTHREADS_INIT", "1");
            config.define("THREADS_PREFER_PTHREAD_FLAG", "ON");
            config.define("onnxruntime_BUILD_UNIT_TESTS", "OFF");
            config.define("onnxruntime_BUILD_APPLE_FRAMEWORK", "ON");
            config.define("onnxruntime_ENABLE_BITCODE", "OFF");

            match target_arch {
                Architecture::X86 | Architecture::X86_64 => {
                    config.define("CMAKE_OSX_ARCHITECTURES", "x86_64");
                },
                Architecture::Arm | Architecture::Arm64 => {
                    config.define("CMAKE_OSX_ARCHITECTURES", "arm64");
                }
            }

            config.cxxflag("-I/opt/homebrew/opt/flatbuffers/include -I/Users/goodnotesci/goodnotes/gn_onnx/third_party/protobuf-21.12/src");
        },
    }
    //std::process::exit(-1);
    config
}


//
fn prepare_libort_dir_compiled() -> PathBuf {
    let mut config = cmake::Config::new("../../cmake");
    //config.showCXXFlag();
    //std::process::exit(-1);
    //println!("Rust Paco: {}", config.get_profile());
    /*
    //config.define("CMAKE_SYSTEM_NAME", "iOS"); //for iOS only
    config.define("CMAKE_SYSTEM_NAME", "Darwin"); //for Catalyst
    config.define("onnxruntime_BUILD_SHARED_LIB", "ON");
    //config.define("ONNX_CUSTOM_PROTOC_EXECUTABLE", "/usr/local/bin/protoc-3.21.12.0");
    //config.define("PROTOBUF_PROTOC_EXECUTABLE", "/usr/local/bin/protoc-3.21.12.0");
    //config.define("PROTOBUF_INCLUDE_DIR", "/Users/goodnotesci/goodnotes/gn_onnx/third_party/protobuf-21.12/src");

    //config.define("protobuf_BUILD_PROTOC_BINARIES", "OFF");
    //config.define("CMAKE_TOOLCHAIN_FILE", "../cmake/onnxruntime_ios.toolchain.cmake"); //for iOS

    config.define("onnxruntime_BUILD_SHARED_LIB", "ON");
    config.define("PYTHON_EXECUTABLE", "/usr/bin/python3");
    //config.define("CMAKE_OSX_DEPLOYMENT_TARGET", "11.0"); //for iOS

    //config.define("CMAKE_OSX_SYSROOT", "/Applications/Xcode.app/Contents/Developer/Platforms/iPhoneOS.platform/Developer/SDKs/iPhoneOS.sdk"); //for iOS, Catalyst uses MacOSX sdk
    config.define("CMAKE_OSX_SYSROOT", "/Applications/Xcode.app/Contents/Developer/Platforms/MacOSX.platform/Developer/SDKs/MacOSX13.0.sdk"); //for Catalyst, iOS uses iPhoneOS sdk

    config.define("CMAKE_THREAD_LIBS_INIT", "-lpthread");
    config.define("CMAKE_HAVE_THREADS_LIBRARY", "1");
    config.define("CMAKE_USE_WIN32_THREADS_INIT", "0");
    config.define("CMAKE_USE_PTHREADS_INIT", "1");
    config.define("THREADS_PREFER_PTHREAD_FLAG", "ON");
    config.define("onnxruntime_BUILD_UNIT_TESTS", "OFF");
    config.define("onnxruntime_BUILD_APPLE_FRAMEWORK", "ON");

    //config.define("onnxruntime_BUILD_SHARED_LIB", "OFF"); //for Catalyst??

    //config.define("onnxruntime_ENABLE_BITCODE", "ON"); //for iOS
    config.define("onnxruntime_ENABLE_BITCODE", "OFF"); //for Catalyst

    // config.define("INCLUDE_DIRECTORIES", "/opt/homebrew/opt/flatbuffers/include");
    config.cxxflag("-I/opt/homebrew/opt/flatbuffers/include -I/Users/goodnotesci/goodnotes/gn_onnx/third_party/protobuf-21.12/src");
    */
    config = prepare_cmake_config(config);

    if let Ok(Accelerator::Cuda) = env::var(ORT_RUST_ENV_GPU).unwrap_or_default().parse() {
        config.define("onnxruntime_USE_CUDA", "ON");
    };

    Ok(config.build())
}

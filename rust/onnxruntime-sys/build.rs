#![allow(dead_code)]

use std::{
    borrow::Cow,
    env, fs,
    io::{self, Read, Write},
    path::{Path, PathBuf},
    str::FromStr,
};

/// ONNX Runtime version
///
/// WARNING: If version is changed, bindings for all platforms will have to be re-generated.
///          To do so, run this:
///              cargo build --package onnxruntime-sys --features generate-bindings
const ORT_VERSION: &str = include_str!("../../VERSION_NUMBER");

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

fn main() {
    let libort_install_dir = prepare_libort_dir();

    let include_dir = libort_install_dir.join("include");
    let lib_dir = libort_install_dir.join("lib");

    println!("Include directory: {:?}", include_dir);
    println!("Lib directory: {:?}", lib_dir);

    // Tell cargo to tell rustc to link onnxruntime shared library.
    println!("cargo:rustc-link-lib=onnxruntime");
    println!("cargo:rustc-link-search=native={}", lib_dir.display());

    println!("cargo:rerun-if-env-changed={}", ORT_RUST_ENV_STRATEGY);
    println!("cargo:rerun-if-env-changed={}", ORT_RUST_ENV_GPU);
    println!(
        "cargo:rerun-if-env-changed={}",
        ORT_RUST_ENV_SYSTEM_LIB_LOCATION
    );

    generate_bindings(&include_dir);
}

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

    let path = include_dir
        .join("onnxruntime")
        .join("onnxruntime_c_api.h");

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

    let generated_file = PathBuf::from(env::var("OUT_DIR").unwrap()).join("bindings.rs");
    bindings
        .write_to_file(&generated_file)
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
    let file = fs::File::open(&filename).unwrap();
    let buf = io::BufReader::new(file);
    let tar = flate2::read::GzDecoder::new(buf);
    let mut archive = tar::Archive::new(tar);
    archive.unpack(output).unwrap();
}

fn extract_zip(filename: &Path, outpath: &Path) {
    let file = fs::File::open(&filename).unwrap();
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
                    fs::create_dir_all(&p).unwrap();
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
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "x86" => Ok(Architecture::X86),
            "x86_64" => Ok(Architecture::X86_64),
            "arm" => Ok(Architecture::Arm),
            "aarch64" => Ok(Architecture::Arm64),
            _ => Err(format!("Unsupported architecture: {}", s)),
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
}

impl Os {
    fn archive_extension(&self) -> &'static str {
        match self {
            Os::Windows => "zip",
            Os::Linux => "tgz",
            Os::MacOs => "tgz",
        }
    }
}

impl FromStr for Os {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "windows" => Ok(Os::Windows),
            "macos" => Ok(Os::MacOs),
            "linux" => Ok(Os::Linux),
            _ => Err(format!("Unsupported os: {}", s)),
        }
    }
}

impl OnnxPrebuiltArchive for Os {
    fn as_onnx_str(&self) -> Cow<str> {
        match self {
            Os::Windows => Cow::from("win"),
            Os::Linux => Cow::from("linux"),
            Os::MacOs => Cow::from("osx"),
        }
    }
}

#[derive(Debug, PartialEq, Eq)]
enum Accelerator {
    Cpu,
    Cuda,
}

impl FromStr for Accelerator {
    type Err = String;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
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

fn prepare_libort_dir() -> PathBuf {
    let strategy = env::var(ORT_RUST_ENV_STRATEGY);
    println!(
        "strategy: {:?}",
        strategy.as_ref().map_or_else(|_| "unknown", String::as_str)
    );
    match strategy.as_ref().map(String::as_str) {
        Ok("download") => prepare_libort_dir_prebuilt(),
        Ok("system") => PathBuf::from(match env::var(ORT_RUST_ENV_SYSTEM_LIB_LOCATION) {
            Ok(p) => p,
            Err(e) => {
                panic!(
                    "Could not get value of environment variable {:?}: {:?}",
                    ORT_RUST_ENV_SYSTEM_LIB_LOCATION, e
                );
            }
        }),
        Ok("compile") | Err(_) => prepare_libort_dir_compiled(),
        _ => panic!("Unknown value for {:?}", ORT_RUST_ENV_STRATEGY),
    }
}

fn prepare_libort_dir_compiled() -> PathBuf {
    let mut config = cmake::Config::new("../../cmake");

    config.define("onnxruntime_BUILD_SHARED_LIB", "ON");

    if env::var(ORT_RUST_ENV_GPU).unwrap_or_default().parse() == Ok(Accelerator::Cuda) {
        config.define("onnxruntime_USE_CUDA", "ON");
    }

    config.build()
}

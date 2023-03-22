#![allow(unused)]

use std::{
	borrow::Cow,
	env, fs,
	io::{self, Read, Write},
	path::{Path, PathBuf},
	process::Stdio,
	str::FromStr
};

const ORT_VERSION: &str = "1.13.1";
const ORT_RELEASE_BASE_URL: &str = "https://github.com/microsoft/onnxruntime/releases/download";
const ORT_ENV_STRATEGY: &str = "ORT_STRATEGY";
const ORT_ENV_SYSTEM_LIB_LOCATION: &str = "ORT_LIB_LOCATION";
const ORT_ENV_CMAKE_TOOLCHAIN: &str = "ORT_CMAKE_TOOLCHAIN";
const ORT_ENV_CMAKE_PROGRAM: &str = "ORT_CMAKE_PROGRAM";
const ORT_ENV_PYTHON_PROGRAM: &str = "ORT_PYTHON_PROGRAM";
const ORT_EXTRACT_DIR: &str = "onnxruntime";
const ORT_GIT_DIR: &str = "onnxruntime";
const ORT_GIT_REPO: &str = "https://github.com/microsoft/onnxruntime";
const PROTOBUF_EXTRACT_DIR: &str = "protobuf";
const PROTOBUF_VERSION: &str = "3.18.1";
const PROTOBUF_RELEASE_BASE_URL: &str = "https://github.com/protocolbuffers/protobuf/releases/download";

macro_rules! incompatible_providers {
	($($provider:ident),*) => {
		#[allow(unused_imports)]
		use casey::upper;
		$(
			if env::var(concat!("CARGO_FEATURE_", stringify!(upper!($provider)))).is_ok() {
				panic!(concat!("Provider not available for this strategy and/or target: ", stringify!($provider)));
			}
		)*
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
	Arm64
}

impl FromStr for Architecture {
	type Err = String;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s {
			"x86" => Ok(Architecture::X86),
			"x86_64" => Ok(Architecture::X86_64),
			"arm" => Ok(Architecture::Arm),
			"aarch64" => Ok(Architecture::Arm64),
			_ => Err(format!("Unsupported architecture: {s}"))
		}
	}
}

impl OnnxPrebuiltArchive for Architecture {
	fn as_onnx_str(&self) -> Cow<str> {
		match self {
			Architecture::X86 => "x86".into(),
			Architecture::X86_64 => "x64".into(),
			Architecture::Arm => "arm".into(),
			Architecture::Arm64 => "arm64".into()
		}
	}
}

#[derive(Debug)]
#[allow(clippy::enum_variant_names)]
enum Os {
	Windows,
	Linux,
	MacOS,
	IOS,
}

impl Os {
	fn archive_extension(&self) -> &'static str {
		match self {
			Os::Windows => "zip",
			Os::Linux => "tgz",
			Os::MacOS => "tgz",
			Os::IOS => "tgz" //Not important
		}
	}
}

impl FromStr for Os {
	type Err = String;

	fn from_str(s: &str) -> Result<Self, Self::Err> {
		match s {
			"windows" => Ok(Os::Windows),
			"linux" => Ok(Os::Linux),
			"macos" => Ok(Os::MacOS),
			"ios" => Ok(Os::IOS),
			_ => Err(format!("Unsupported OS: {s}"))
		}
	}
}

impl OnnxPrebuiltArchive for Os {
	fn as_onnx_str(&self) -> Cow<str> {
		match self {
			Os::Windows => "win".into(),
			Os::Linux => "linux".into(),
			Os::MacOS => "osx".into(),
			Os::IOS => "ios".into() //Not important
		}
	}
}

#[derive(Debug)]
enum Accelerator {
	None,
	Gpu
}

impl OnnxPrebuiltArchive for Accelerator {
	fn as_onnx_str(&self) -> Cow<str> {
		match self {
			Accelerator::None => "unaccelerated".into(),
			Accelerator::Gpu => "gpu".into()
		}
	}
}

#[derive(Debug)]
struct Triplet {
	os: Os,
	arch: Architecture,
	accelerator: Accelerator
}

impl OnnxPrebuiltArchive for Triplet {
	fn as_onnx_str(&self) -> Cow<str> {
		match (&self.os, &self.arch, &self.accelerator) {
			(Os::Windows, Architecture::X86, Accelerator::None)
			| (Os::Windows, Architecture::X86_64, Accelerator::None)
			| (Os::Windows, Architecture::Arm, Accelerator::None)
			| (Os::Windows, Architecture::Arm64, Accelerator::None)
			| (Os::Linux, Architecture::X86_64, Accelerator::None)
			| (Os::MacOS, Architecture::Arm64, Accelerator::None) => format!("{}-{}", self.os.as_onnx_str(), self.arch.as_onnx_str()).into(),
			// for some reason, arm64/Linux uses `aarch64` instead of `arm64`
			(Os::Linux, Architecture::Arm64, Accelerator::None) => format!("{}-{}", self.os.as_onnx_str(), "aarch64").into(),
			// for another odd reason, x64/macOS uses `x86_64` instead of `x64`
			(Os::MacOS, Architecture::X86_64, Accelerator::None) => format!("{}-{}", self.os.as_onnx_str(), "x86_64").into(),
			(Os::Windows, Architecture::X86_64, Accelerator::Gpu) | (Os::Linux, Architecture::X86_64, Accelerator::Gpu) => {
				format!("{}-{}-{}", self.os.as_onnx_str(), self.arch.as_onnx_str(), self.accelerator.as_onnx_str()).into()
			}
			_ => panic!(
				"Microsoft does not provide ONNX Runtime downloads for triplet: {}-{}-{}; you may have to use the `system` strategy instead",
				self.os.as_onnx_str(),
				self.arch.as_onnx_str(),
				self.accelerator.as_onnx_str()
			)
		}
	}
}

fn prebuilt_onnx_url() -> (PathBuf, String) {
	let accelerator = if cfg!(feature = "cuda") || cfg!(feature = "tensorrt") {
		Accelerator::Gpu
	} else {
		Accelerator::None
	};

	let triplet = Triplet {
		os: env::var("CARGO_CFG_TARGET_OS").expect("unable to get target OS").parse().unwrap(),
		arch: env::var("CARGO_CFG_TARGET_ARCH").expect("unable to get target arch").parse().unwrap(),
		accelerator
	};

	let prebuilt_archive = format!("onnxruntime-{}-{}.{}", triplet.as_onnx_str(), ORT_VERSION, triplet.os.archive_extension());
	let prebuilt_url = format!("{ORT_RELEASE_BASE_URL}/v{ORT_VERSION}/{prebuilt_archive}");

	(PathBuf::from(prebuilt_archive), prebuilt_url)
}

fn prebuilt_protoc_url() -> (PathBuf, String) {
	let host_platform = if cfg!(target_os = "windows") {
		std::string::String::from("win32")
	} else if cfg!(target_os = "macos") {
		format!(
			"osx-{}",
			if cfg!(target_arch = "x86_64") {
				"x86_64"
			} else if cfg!(target_arch = "x86") {
				"x86"
			} else {
				panic!("protoc does not have prebuilt binaries for darwin arm64 yet")
			}
		)
	} else {
		format!("linux-{}", if cfg!(target_arch = "x86_64") { "x86_64" } else { "x86_32" })
	};

	let prebuilt_archive = format!("protoc-{PROTOBUF_VERSION}-{host_platform}.zip");
	let prebuilt_url = format!("{PROTOBUF_RELEASE_BASE_URL}/v{PROTOBUF_VERSION}/{prebuilt_archive}");

	(PathBuf::from(prebuilt_archive), prebuilt_url)
}

fn download<P>(source_url: &str, target_file: P)
where
	P: AsRef<Path>
{
	let resp = ureq::get(source_url)
		.timeout(std::time::Duration::from_secs(300))
		.call()
		.unwrap_or_else(|err| panic!("[ort] failed to download {source_url}: {err:?}"));

	let len = resp.header("Content-Length").and_then(|s| s.parse::<usize>().ok()).unwrap();
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
	match filename.extension().map(|e| e.to_str()) {
		Some(Some("zip")) => extract_zip(filename, output),
		#[cfg(not(target_os = "windows"))]
		Some(Some("tgz")) => extract_tgz(filename, output),
		_ => unimplemented!()
	}
}

#[cfg(not(target_os = "windows"))]
fn extract_tgz(filename: &Path, output: &Path) {
	let file = fs::File::open(&filename).unwrap();
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
		let outpath = outpath.join(file.enclosed_name().unwrap());
		if !file.name().ends_with('/') {
			println!("File {} extracted to \"{}\" ({} bytes)", i, outpath.as_path().display(), file.size());
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

fn copy_libraries(lib_dir: &Path, out_dir: &Path) {
	// get the target directory - we need to place the dlls next to the executable so they can be properly loaded by windows
	let out_dir = out_dir.parent().unwrap().parent().unwrap().parent().unwrap();

	let lib_files = fs::read_dir(lib_dir).unwrap();
	for lib_file in lib_files.filter(|e| {
		e.as_ref().ok().map_or(false, |e| {
			e.file_type().map_or(false, |e| !e.is_dir())
				&& [".dll", ".so", ".dylib"]
					.into_iter()
					.any(|v| e.path().into_os_string().into_string().unwrap().contains(v))
		})
	}) {
		let lib_file = lib_file.unwrap();
		let lib_path = lib_file.path();
		let lib_name = lib_path.file_name().unwrap();
		let out_path = out_dir.join(lib_name);
		if !out_path.exists() {
			fs::copy(&lib_path, out_path).unwrap();
		}
	}
}

fn prepare_libort_dir() -> (PathBuf, bool) {
	let strategy = env::var(ORT_ENV_STRATEGY);
	println!("[ort] strategy: {:?}", strategy.as_ref().map(String::as_str).unwrap_or_else(|_| "unknown"));

	let target = env::var("TARGET").unwrap();
	let target_arch = env::var("CARGO_CFG_TARGET_ARCH").unwrap();
	if target_arch.eq_ignore_ascii_case("aarch64") {
		incompatible_providers![cuda, openvino, vitis_ai, tensorrt, migraphx, rocm];
	} else if target_arch.eq_ignore_ascii_case("x86_64") {
		incompatible_providers![vitis_ai, acl, armnn];
	} else {
		panic!("unsupported target architecture: {target_arch}");
	}

	if target.contains("macos") {
		incompatible_providers![cuda, openvino, tensorrt, directml, winml];
	} else if target.contains("windows") {
		incompatible_providers![coreml, vitis_ai, acl, armnn];
	} else {
		incompatible_providers![coreml, vitis_ai, directml, winml];
	}

	println!("cargo:rerun-if-env-changed={}", ORT_ENV_STRATEGY);

	match strategy.as_ref().map_or("download", String::as_str) {
		"download" => {
			if target.contains("macos") {
				incompatible_providers![cuda, onednn, openvino, openmp, vitis_ai, tvm, tensorrt, migraphx, directml, winml, acl, armnn, rocm];
			} else {
				incompatible_providers![onednn, coreml, openvino, openmp, vitis_ai, tvm, migraphx, directml, winml, acl, armnn, rocm];
			}

			let (prebuilt_archive, prebuilt_url) = prebuilt_onnx_url();

			let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
			let extract_dir = out_dir.join(ORT_EXTRACT_DIR);
			let downloaded_file = out_dir.join(&prebuilt_archive);

			println!("cargo:rerun-if-changed={}", downloaded_file.display());

			if !downloaded_file.exists() {
				fs::create_dir_all(&out_dir).unwrap();
				download(&prebuilt_url, &downloaded_file);
			}

			if !extract_dir.exists() {
				extract_archive(&downloaded_file, &extract_dir);
			}

			let lib_dir = extract_dir.join(prebuilt_archive.file_stem().unwrap());
			#[cfg(feature = "copy-dylibs")]
			{
				copy_libraries(&lib_dir.join("lib"), &out_dir);
			}

			(lib_dir, true)
		}
		"system" => {
			let lib_dir = PathBuf::from(env::var(ORT_ENV_SYSTEM_LIB_LOCATION).expect("[ort] system strategy requires ORT_LIB_LOCATION env var to be set"));
			#[cfg(feature = "copy-dylibs")]
			{
				copy_libraries(&lib_dir.join("lib"), &PathBuf::from(env::var("OUT_DIR").unwrap()));
			}

			let mut needs_link = true;
			if lib_dir.join("libonnxruntime_common.a").exists() {
				println!("cargo:rustc-link-search=native={}", lib_dir.display());

				let external_lib_dir = lib_dir.join("external");
				println!("cargo:rustc-link-search=native={}", external_lib_dir.join("protobuf").join("cmake").display());
				println!("cargo:rustc-link-lib=static=protobuf-lited");

				println!("cargo:rustc-link-search=native={}", external_lib_dir.join("onnx").display());
				println!("cargo:rustc-link-lib=static=onnx");
				println!("cargo:rustc-link-lib=static=onnx_proto");

				println!("cargo:rustc-link-search=native={}", external_lib_dir.join("nsync").display());
				println!("cargo:rustc-link-lib=static=nsync_cpp");

				println!("cargo:rustc-link-search=native={}", external_lib_dir.join("re2").display());
				println!("cargo:rustc-link-lib=static=re2");

				println!("cargo:rustc-link-search=native={}", external_lib_dir.join("abseil-cpp").join("absl").join("base").display());
				println!("cargo:rustc-link-lib=static=absl_base");
				println!("cargo:rustc-link-lib=static=absl_throw_delegate");
				println!("cargo:rustc-link-search=native={}", external_lib_dir.join("abseil-cpp").join("absl").join("hash").display());
				println!("cargo:rustc-link-lib=static=absl_hash");
				println!("cargo:rustc-link-lib=static=absl_low_level_hash");
				println!("cargo:rustc-link-search=native={}", external_lib_dir.join("abseil-cpp").join("absl").join("container").display());
				println!("cargo:rustc-link-lib=static=absl_raw_hash_set");

				if cfg!(target_os = "macos") {
					println!("cargo:rustc-link-lib=framework=Foundation");
				}

				println!("cargo:rustc-link-lib=onnxruntime_providers_shared");
				#[cfg(feature = "rocm")]
				println!("cargo:rustc-link-lib=onnxruntime_providers_rocm");

				needs_link = false;
			}

			(lib_dir, needs_link)
		}
		"compile" => {
			use std::process::Command;

			let target = env::var("TARGET").unwrap();
			let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

			let python = env::var("PYTHON").unwrap_or_else(|_| "python".to_string());

			Command::new("git")
				.args([
					"clone",
					"--depth",
					"1",
					"--single-branch",
					"--branch",
					&format!("v{ORT_VERSION}"),
					"--shallow-submodules",
					"--recursive",
					ORT_GIT_REPO,
					ORT_GIT_DIR
				])
				.current_dir(&out_dir)
				.stdout(Stdio::null())
				.stderr(Stdio::null())
				.status()
				.expect("failed to clone ORT repo");

			let root = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
			let _cmake_toolchain = env::var(ORT_ENV_CMAKE_TOOLCHAIN).map_or_else(
				|_| {
					if cfg!(target_os = "linux") && target.contains("aarch64") && target.contains("linux") {
						root.join("toolchains").join("default-aarch64-linux-gnu.cmake")
					} else if cfg!(target_os = "linux") && target.contains("aarch64") && target.contains("windows") {
						root.join("toolchains").join("default-aarch64-w64-mingw32.cmake")
					} else if cfg!(target_os = "linux") && target.contains("x86_64") && target.contains("windows") {
						root.join("toolchains").join("default-x86_64-w64-mingw32.cmake")
					} else {
						PathBuf::default()
					}
				},
				PathBuf::from
			);

			let mut command = Command::new(python);
			command
				.current_dir(&out_dir.join(ORT_GIT_DIR))
				.stdout(Stdio::null())
				.stderr(Stdio::inherit());

			// note: --parallel will probably break something... parallel build *while* doing another parallel build (cargo)?
			let mut build_args = vec!["tools/ci_build/build.py", "--build", "--update", "--parallel", "--skip_tests", "--skip_submodule_sync"];
			let config = if cfg!(debug_assertions) {
				"Debug"
			} else if cfg!(feature = "minimal-build") {
				"MinSizeRel"
			} else {
				"Release"
			};
			build_args.push("--config");
			build_args.push(config);

			if cfg!(feature = "minimal-build") {
				build_args.push("--disable_exceptions");
			}

			build_args.push("--disable_rtti");

			if target.contains("windows") {
				build_args.push("--disable_memleak_checker");
			}

			if !cfg!(feature = "compile-static") {
				build_args.push("--build_shared_lib");
			} else {
				build_args.push("--enable_msvc_static_runtime");
			}

			// onnxruntime will still build tests when --skip_tests is enabled, this filters out most of them
			// this "fixes" compilation on alpine: https://github.com/microsoft/onnxruntime/issues/9155
			// but causes other compilation errors: https://github.com/microsoft/onnxruntime/issues/7571
			// build_args.push("--cmake_extra_defines");
			// build_args.push("onnxruntime_BUILD_UNIT_TESTS=0");

			#[cfg(windows)]
			{
				use vswhom::VsFindResult;
				let vs_find_result = VsFindResult::search();
				match vs_find_result {
					Some(VsFindResult { vs_exe_path: Some(vs_exe_path), .. }) => {
						let vs_exe_path = vs_exe_path.to_string_lossy();
						// the one sane thing about visual studio is that the version numbers are somewhat predictable...
						if vs_exe_path.contains("14.1") {
							build_args.push("--cmake_generator=Visual Studio 15 2017");
						} else if vs_exe_path.contains("14.2") {
							build_args.push("--cmake_generator=Visual Studio 16 2019");
						} else if vs_exe_path.contains("14.3") {
							build_args.push("--cmake_generator=Visual Studio 17 2022");
						}
					}
					Some(VsFindResult { vs_exe_path: None, .. }) | None => panic!("[ort] unable to find Visual Studio installation")
				};
			}

			let current_line = line!();
			println!("defined on line: {current_line}");

			build_args.push("--build_dir=build");
			command.args(build_args);

			println!("{:?}", command);

			let code = command.status().expect("failed to run build script");
			assert!(code.success(), "failed to build ONNX Runtime");
			println!("defined on line: {current_line}");

			let lib_dir = out_dir.join(ORT_GIT_DIR).join("build").join(config);
			let lib_dir = if cfg!(target_os = "windows") { lib_dir.join(config) } else { lib_dir };
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

			let current_line2 = line!();
			println!("defined on line: {current_line2}");

			println!("cargo:rustc-link-search=native={}", lib_dir.display());

			let external_lib_dir = lib_dir.join("external");
			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("protobuf").join("cmake").display());
			println!("cargo:rustc-link-lib=static=protobuf-lited");

			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("onnx").display());
			println!("cargo:rustc-link-lib=static=onnx");
			println!("cargo:rustc-link-lib=static=onnx_proto");

			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("nsync").display());
			println!("cargo:rustc-link-lib=static=nsync_cpp");

			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("re2").display());
			println!("cargo:rustc-link-lib=static=re2");

			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("abseil-cpp").join("absl").join("base").display());
			println!("cargo:rustc-link-lib=static=absl_base");
			println!("cargo:rustc-link-lib=static=absl_throw_delegate");
			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("abseil-cpp").join("absl").join("hash").display());
			println!("cargo:rustc-link-lib=static=absl_hash");
			println!("cargo:rustc-link-lib=static=absl_low_level_hash");
			println!("cargo:rustc-link-search=native={}", external_lib_dir.join("abseil-cpp").join("absl").join("container").display());
			println!("cargo:rustc-link-lib=static=absl_raw_hash_set");

			if cfg!(target_os = "macos") {
				println!("cargo:rustc-link-lib=framework=Foundation");
			}

			println!("cargo:rustc-link-lib=onnxruntime_providers_shared");
			#[cfg(feature = "rocm")]
			println!("cargo:rustc-link-lib=onnxruntime_providers_rocm");

			(out_dir, false)
		}
		_ => panic!("[ort] unknown strategy: {} (valid options are `download` or `system`)", strategy.unwrap_or_else(|_| "unknown".to_string()))
	}
}

#[cfg(feature = "generate-bindings")]
fn generate_bindings(include_dir: &Path) {
	let clang_args = &[
		format!("-I{}", include_dir.display()),
		format!("-I{}", include_dir.join("onnxruntime").join("core").join("session").display())
	];

	println!("cargo:rerun-if-changed=src/wrapper.h");
	println!("Paco: hello");
	let bindings = bindgen::Builder::default()
        .header("src/wrapper.h")
        .clang_args(clang_args)
        // Tell cargo to invalidate the built crate whenever any of the included header files changed.
        .parse_callbacks(Box::new(bindgen::CargoCallbacks))
        // Set `size_t` to be translated to `usize` for win32 compatibility.
        .size_t_is_usize(env::var("CARGO_CFG_TARGET_ARCH").unwrap().contains("x86"))
        // Format using rustfmt
        .rustfmt_bindings(true)
        .rustified_enum("*")
        .generate()
        .expect("Unable to generate bindings");

	// Write the bindings to (source controlled) src/onnx/bindings/<os>/<arch>/bindings.rs
	let generated_file = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap())
		.join("src")
		.join("bindings")
		.join(env::var("CARGO_CFG_TARGET_OS").unwrap())
		.join(env::var("CARGO_CFG_TARGET_ARCH").unwrap())
		.join("bindings.rs");
	println!("cargo:rerun-if-changed={generated_file:?}");
	fs::create_dir_all(generated_file.parent().unwrap()).unwrap();
	bindings.write_to_file(&generated_file).expect("Couldn't write bindings!");
}

fn main() {
	println!("Paco: hello main 1");
	if std::env::var("DOCS_RS").is_err() {
		let (install_dir, needs_link) = prepare_libort_dir();
		println!("Paco: hello main 2");

		let include_dir = install_dir.join("include");
		println!("Paco: hello main 3");
		let lib_dir = install_dir.join("lib");

		println!("Paco: hello main 4");
		if needs_link {
			println!("cargo:rustc-link-lib=onnxruntime");
			println!("cargo:rustc-link-search=native={}", lib_dir.display());
		}

		println!("cargo:rerun-if-env-changed={}", ORT_ENV_STRATEGY);
		println!("cargo:rerun-if-env-changed={}", ORT_ENV_SYSTEM_LIB_LOCATION);

		#[cfg(feature = "generate-bindings")]
		generate_bindings(&include_dir);
	}
}

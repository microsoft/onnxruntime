Name:           onnxruntime
Version:        1.8.0
Release:        1%{?dist}
Summary:        onnxruntime

License:        MIT
URL:            https://onnx.ai
Source0:        onnxruntime.tar

BuildRequires:  gcc
BuildRequires:  gcc-c++
BuildRequires:  zlib-devel
BuildRequires:  make
BuildRequires:  git
BuildRequires:  python3-devel
BuildRequires:  python3-numpy
BuildRequires:  python3-setuptools
BuildRequires:  python3-wheel
BuildRequires:  bzip2
Requires:       libstdc++
Requires:       glibc

%description


%prep
%autosetup


%build
mkdir debug
cd debug
/opt/cmake/bin/cmake -Donnxruntime_DEV_MODE=OFF -DCMAKE_DEBUG_POSTFIX=_debug -DCMAKE_BUILD_TYPE=Debug -Deigen_SOURCE_PATH=/usr/include/eigen3 -Donnxruntime_USE_PREINSTALLED_EIGEN=ON -Donnxruntime_BUILD_SHARED_LIB=ON -DCMAKE_INSTALL_PREFIX=%{_prefix} ../cmake
%make_build
cd ..
mkdir release
cd release
/opt/cmake/bin/cmake -Donnxruntime_DEV_MODE=OFF -DCMAKE_BUILD_TYPE=RelWithDebInfo -Deigen_SOURCE_PATH=/usr/include/eigen3 -Donnxruntime_USE_PREINSTALLED_EIGEN=ON -Donnxruntime_BUILD_SHARED_LIB=ON -DCMAKE_INSTALL_PREFIX=%{_prefix} ../cmake
%make_build
cd ..

%install
rm -rf $RPM_BUILD_ROOT
cd debug
%make_install
cd ..
cd release
%make_install
cd ..

%files
%license LICENSE
%doc docs/*
%doc ThirdPartyNotices.txt
%{_bindir}/onnx_test_runner
%{_libdir}/libonnxruntime.so*
%{_libdir}/libonnxruntime_debug.so*
%{_includedir}/onnxruntime/*


%changelog
* Wed Oct 17 2018 ONNXRuntime Team
- Initial release

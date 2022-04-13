rm -rf build
python3 tools/ci_build/build.py --build_dir build --config RelWithDebInfo --parallel --use_hip --hip_home /opt/rocm --enable_training \
> build_log.txt
#export PATH=/usr/local/cuda-11.7/bin${PATH:+:${PATH}}
#export LD_LIBRARY_PATH=/usr/local/cuda-11.7/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}
#cd /work/tlwu/git/onnxruntime
#sh build_release.sh
#sh install_release.sh
cd /work/tlwu/git/onnxruntime/onnxruntime/python/tools/transformers/models/stable_diffusion
python optimize_pipeline.py -i /work/tlwu/sd_onnx/stable-diffusion-v2-1-fp32 -o /work/tlwu/sd_onnx/stable-diffusion-v2-1-fp16 -e --overwrite --float16
python benchmark.py -v 2.1 -p /work/tlwu/sd_onnx/stable-diffusion-v2-1-fp16 -b 1


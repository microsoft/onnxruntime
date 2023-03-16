python benchmark.py -e torch -b 1
sleep 15
python benchmark.py -e torch -b 4
sleep 15
python benchmark.py -e torch -b 8
sleep 15
python benchmark.py -e torch -b 16
sleep 15
python benchmark.py -e torch -b 32
sleep 15

python benchmark.py -e torch --enable_torch_compile -b 1 
sleep 15
python benchmark.py -e torch --enable_torch_compile -b 4
sleep 15
python benchmark.py -e torch --enable_torch_compile -b 8
sleep 15
python benchmark.py -e torch --enable_torch_compile -b 16
sleep 15
python benchmark.py -e torch --enable_torch_compile -b 32
sleep 15

python benchmark.py -p /nvme/users/tlwu/stable-diffusion-v1-5/ -b 1
sleep 15
python benchmark.py -p /nvme/users/tlwu/stable-diffusion-v1-5/ -b 4
sleep 15
python benchmark.py -p /nvme/users/tlwu/stable-diffusion-v1-5/ -b 8
sleep 15
python benchmark.py -p /nvme/users/tlwu/stable-diffusion-v1-5/ -b 16
sleep 15

python benchmark.py --enable_safety_checker -e torch -b 1
sleep 15
python benchmark.py --enable_safety_checker -e torch -b 4
sleep 15
python benchmark.py --enable_safety_checker -e torch -b 8
sleep 15
python benchmark.py --enable_safety_checker -e torch -b 16
sleep 15
python benchmark.py --enable_safety_checker -e torch -b 32
sleep 15

python benchmark.py --enable_safety_checker -e torch --enable_torch_compile -b 1 
sleep 15
python benchmark.py --enable_safety_checker -e torch --enable_torch_compile -b 4
sleep 15
python benchmark.py --enable_safety_checker -e torch --enable_torch_compile -b 8
sleep 15
python benchmark.py --enable_safety_checker -e torch --enable_torch_compile -b 16
sleep 15
python benchmark.py --enable_safety_checker -e torch --enable_torch_compile -b 32
sleep 15

python benchmark.py --enable_safety_checker -p /nvme/users/tlwu/stable-diffusion-v1-5/ -b 1
sleep 15
python benchmark.py --enable_safety_checker -p /nvme/users/tlwu/stable-diffusion-v1-5/ -b 4
sleep 15
python benchmark.py --enable_safety_checker -p /nvme/users/tlwu/stable-diffusion-v1-5/ -b 8
sleep 15
python benchmark.py --enable_safety_checker -p /nvme/users/tlwu/stable-diffusion-v1-5/ -b 16
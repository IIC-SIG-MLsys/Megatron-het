# Docker
## nvidia
```
docker pull pytorch/pytorch:2.4.0-cuda11.8-cudnn8-devel

docker run -it \
  --gpus '"device=0,1"' \
  --network host \
  --device /dev/infiniband/rdma_cm \
  --device /dev/infiniband/uverbs0 \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -v $(pwd):/workspace \
  -w /workspace \
  --name megatron-het \
  --cap-add IPC_LOCK \
  --cap-add SYS_NICE \
  --security-opt seccomp=unconfined \
  pytorch/pytorch:2.4.0-cuda11.8-cudnn8-devel \
  /bin/bash

docker stop megatron-het && docker rm megatron-het
```

Fake Data:
```
pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
// pip install megatron-core
```

Train
```
cd thirdparty/Nvidia-Megatron-LM-0.9.0
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py
```
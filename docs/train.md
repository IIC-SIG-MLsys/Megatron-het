# Docker

## nvidia
```
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel

docker run -itd \
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
  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/pytorch/pytorch:2.5.0-cuda12.4-cudnn9-devel \
  /bin/bash

docker exec -it megatron-het bash

docker stop megatron-het && docker rm megatron-het
```

```
// for transformer-engine[pytorch]
// pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
conda create -n megatron python=3.10 pytorch cudatoolkit=11.8 cudnn -c pytorch -c nvidia -c conda-forge
conda activate megatron
pip install -r requirements_0.9.0.txt
```

Prepare Data:
```
cd data && bash download_bert_data.sh
```

Train
```
cd thirdparty/nvidia_megatron_lm_0_9_0
torchrun --nproc_per_node=2 examples/run_simple_mcore_train_loop.py

CHECKPOINT_PATH="" #<Specify path>
TENSORBOARD_LOGS_PATH=""#<Specify path>
VOCAB_FILE="" #<Specify path to file>//bert-vocab.txt
DATA_PATH="" #<Specify path and file prefix>_text_document

bash examples/bert/train_bert_340m_distributed.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $DATA_PATH
```

## uccl
```
git submodule update --init --recursive thirdparty/uccl
cd thirdparty/uccl/p2p && bash build_and_install.sh [cuda|rocm] p2p
```

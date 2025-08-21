# Docker

## nvidia
```
docker pull swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvcr.io/nvidia/pytorch:24.07-py3

docker run -itd \
  --gpus '"device=0,1"' \
  --network host \
  --shm-size=8G \
  --device /dev/infiniband/rdma_cm \
  --device /dev/infiniband/uverbs0 \
  -v /sys/class/infiniband:/sys/class/infiniband:ro \
  -v $(pwd):/workspace \
  -w /workspace \
  --name megatron-het \
  --cap-add IPC_LOCK \
  --cap-add SYS_NICE \
  --security-opt seccomp=unconfined \
  swr.cn-north-4.myhuaweicloud.com/ddn-k8s/docker.io/nvcr.io/nvidia/pytorch:24.07-py3 \
  /bin/bash

docker exec -it megatron-het bash

docker stop megatron-het && docker rm megatron-het
```

test ib
```
apt-get update && apt-get install -y infiniband-diags libibverbs-dev
// ibstatus
```

```
// for transformer-engine[pytorch]
// pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple
// conda create -n megatron python=3.10 pytorch cudatoolkit=11.8 cudnn -c pytorch -c nvidia -c conda-forge
// conda activate megatron
python -m pip install --upgrade pip
pip install -r requirements_0.9.0.txt
```

Prepare Data:
```
cd data && bash download_bert_data.sh && bash progress_bert_data.sh
```

Train
```
export CHECKPOINT_PATH="/workspace/data/checkpoints" && \
export TENSORBOARD_LOGS_PATH="/workspace/data/tensorboard" && \
export VOCAB_FILE="/workspace/data/gpt2-vocab.json" && \
export MERGE_FILE="/workspace/data/gpt2-merges.txt" && \
export DATA_PATH="/workspace/data/wikidata/output/my-gpt2_text_document"

bash ./examples/gpt3/train_gpt3_345m_distributed.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH
```

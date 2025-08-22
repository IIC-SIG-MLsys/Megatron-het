# Megatron-Het
Support training with heterogeneous GPUs using Megatron-LM
- Pipeline (v0.9.0)
- TODO: MOE

## Init submodule on demand
Due to different Megatron-LM versions maintained by various GPU vendors, it is necessary to ensure version alignment by using a unified version, organized via submodules.
```
git submodule update --init --recursive thirdparty/ascend_megatron_lm
git submodule update --init --recursive thirdparty/nvidia_megatron_lm_0_9_0
```
For example: 
- When using MLU GPUs and NVIDIA GPUs, since MLU provides Megatron-LM up to version v0.9.0, we need to use the corresponding version of Megatron-LM for NVIDIA.


## Train gpt3
prepare dataset first
```
cd data && bash download_wiki_data.sh && bash progress_wiki_data.sh
```

```
export MEGATRON_PROVIDER=nvidia (nvidia, mlu, ascend)

export CHECKPOINT_PATH="/workspace/data/checkpoints" && \
export TENSORBOARD_LOGS_PATH="/workspace/data/tensorboard" && \
export VOCAB_FILE="/workspace/data/gpt2-vocab.json" && \
export MERGE_FILE="/workspace/data/gpt2-merges.txt" && \
export DATA_PATH="/workspace/data/wikidata/output/my-gpt2_text_document"

bash ./examples/gpt3/train_gpt3_345m_distributed.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH
```

## For Cambircon MLU
The software package needs to be obtained from the vendor and saved under the thirdparty directory, name it with mlu_megatron_lm.

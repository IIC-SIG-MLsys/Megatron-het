
## init submodule on demand
git submodule update --init --recursive thirdparty/ascend_megatron_lm
git submodule update --init --recursive thirdparty/nvidia_megatron_lm_0_9_0
git submodule update --init --recursive thirdparty/uccl
git submodule update --init --recursive thirdparty/HMC


# train gpt
export CHECKPOINT_PATH="/workspace/data/checkpoints" && \
export TENSORBOARD_LOGS_PATH="/workspace/data/tensorboard" && \
export VOCAB_FILE="/workspace/data/gpt2-vocab.json" && \
export MERGE_FILE="/workspace/data/gpt2-merges.txt" && \
export DATA_PATH="/workspace/data/wikidata/output/my-gpt2_text_document"

bash ./examples/gpt3/train_gpt3_345m_distributed.sh $CHECKPOINT_PATH $TENSORBOARD_LOGS_PATH $VOCAB_FILE $MERGE_FILE $DATA_PATH

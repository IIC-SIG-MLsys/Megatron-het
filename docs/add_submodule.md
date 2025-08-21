# How add new submodule
```
git submodule add https://github.com/NVIDIA/Megatron-LM.git thirdparty/ascend_megatron_lm

cd thirdparty/ascend_megatron_lm
git checkout core_v0.12.1
cd ../..

git add thirdparty/ascend_megatron_lm
git commit -m "add ascend_megatron_lm @ tag core_v0.12.1"
```

```
git submodule add https://github.com/NVIDIA/Megatron-LM.git thirdparty/nvidia_megatron_lm_0_9_0

cd thirdparty/nvidia_megatron_lm_0_9_0
git checkout core_v0.9.0
cd ../..

git add thirdparty/nvidia_megatron_lm_0_9_0
git commit -m "add nvidia_megatron_lm_0_9_0 @ tag core_v0.9.0"
```
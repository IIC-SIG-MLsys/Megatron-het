# How add new submodule
```
git submodule add https://github.com/NVIDIA/Megatron-LM.git thirdparty/Ascend-Megatron-LM

cd thirdparty/Ascend-Megatron-LM
git checkout core_v0.12.1
cd ../..

git add thirdparty/Ascend-Megatron-LM
git commit -m "build: add Ascend-Megatron-LM @ tag core_v0.12.1"
```
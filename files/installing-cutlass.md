
# Installing dependencies 

## Cutlass (on V100 GPUs)

```
git submodule update --init
mkdir ext/cutlass/build
cd ext/cutlass/build
cmake .. -DCUTLASS_NVCC_ARCHS=70
make -j12
```
-----


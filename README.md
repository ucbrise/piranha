
# Piranha: A GPU Platform for Secure Computation
-----

Piranha is a C++-based platform for accelerating secure multi-party computation (MPC) protocols on the GPU in a protocol-independent manner. It is designed both for MPC developers, providing a modular structure for easily adding new protocol implementations, and secure application developers, allowing execution on any Piranha-implemented protocols.

Piranha is described in more detail in our USENIX Security '22 paper: {Link coming soon}.

This repo currently includes a secure ML inference and training application, which you can find in `/nn`.

## Build

> (Note; some of these instructions are fairly incomplete and will be updated in the near future :) ).

Assumes that you have CUDA drivers/toolkit installed.

1. Checkout external modules

```
git submodule update --init --recursive
```

1. Build CUTLASS

```
cd ext/cutlass
mkdir build
cmake .. -DCUTLASS_NVCC_ARCHS=70 -DCMAKE_CUDA_COMPILER_WORKS=1 -DCMAKE_CUDA_COMPILER=<YOUR NVCC PATH>
make -j
```

1. Install GTest. We use it for unit testing.

```
sudo apt install libgtest-dev libssl-dev
cd /usr/src/gtest
sudo mkdir build
cd build
sudo cmake ..
sudo make
sudo make install
```

1. Create some necessary directories

```
mkdir output; mkdir files/MNIST; mkdir files/CIFAR10
```

1. Download the MNIST/CIFAR10 datasets, if using. This step might take a while

```
cd scripts
sudo pip install torch torchvision
python download_{mnist, cifar10}.py
```

1. Build Piranha at a specific fixed point precision and for a particular protocol:

```
make -j8 PIRANHA_FLAGS="-DFLOAT_PRECISION=<NBITS> -D{TWOPC,FOURPC}"
```

## Run

1. Copy and set up a run configuration from one of the examples in `/deploy`.

1. Run Piranha with a party number:

```
./piranha -p <PARTY NUM> -c <CONFIG>
```


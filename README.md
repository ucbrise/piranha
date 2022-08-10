
# Piranha: A GPU Platform for Secure Computation

<p align="center">
    <img src="https://github.com/ucbrise/piranha/blob/main/files/piranha-fish.png?raw=true" alt="cute cuddly PIRANHA >:D courtesy of Vivian Fang @ vivi.sh" width=20% height=20%/>
</p>

Piranha is a C++-based platform for accelerating secure multi-party computation (MPC) protocols on the GPU in a protocol-independent manner. It is designed both for MPC developers, providing a modular structure for easily adding new protocol implementations, and secure application developers, allowing execution on any Piranha-implemented protocols. This repo currently includes a secure ML inference and training application, which you can find in `/nn`.

Piranha is described in more detail in our [USENIX Security '22 paper](https://eprint.iacr.org/2022/892)! If you have questions, please create git issues; for eventual replies, you can also reach out to `jlw@berkeley.edu`.

<p align="center">
    <img src="https://github.com/ucbrise/piranha/blob/main/files/images/usenixbadges-available.png?raw=true" alt="usenix-available" width=10% height=10%/>
    <img src="https://github.com/ucbrise/piranha/blob/main/files/images/usenixbadges-functional.png?raw=true" alt="usenix-functional" width=10% height=10%/>
    <img src="https://github.com/ucbrise/piranha/blob/main/files/images/usenixbadges-reproduced.png?raw=true" alt="usenix-reproduced" width=10% height=10%/>
</p>

**Warning**: This is an academic proof-of-concept prototype and has not received careful code review. This implementation is NOT ready for production use.

## Build

This project requires an NVIDIA GPU, and assumes you have your GPU drivers and the [NVIDIA CUDA Toolkit](https://docs.nvidia.com/cuda/) already installed. The following has been tested on AWS with the `Deep Learning Base AMI (Ubuntu 18.04 ) Version 53.5` AMI.

1. Checkout external modules
```
git submodule update --init --recursive
```

1. Build CUTLASS

```
cd ext/cutlass
mkdir build
cmake .. -DCUTLASS_NVCC_ARCHS=<YOUR_GPU_ARCH_HERE> -DCMAKE_CUDA_COMPILER_WORKS=1 -DCMAKE_CUDA_COMPILER=<YOUR NVCC PATH HERE>
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

1. Build Piranha at a specific fixed point precision and for a particular protocol. 3-party replicated secret sharing is the default and doesn't require a command-line flag.

```
make -j8 PIRANHA_FLAGS="-DFLOAT_PRECISION=<NBITS> -D{TWOPC,FOURPC}"
```

## Run

1. Copy and set up a run configuration using `config.json` as a base. It is already set up to perform a 10-epoch SecureML training run; simply specify party IPs in the configuration.

1. Run Piranha on each machine with a party number (0 -> n_parties - 1):

```
./piranha -p <PARTY NUM> -c <CONFIG FILE>
```

## Citation

You can cite the paper using the following BibTeX entry (the paper links to this repo):

```
@inproceedings {watson22piranha,
    author = {Watson, Jean-Luc and Wagh, Sameer and Popa, Raluca Ada},
    title = {Piranha: A {GPU} Platform for Secure Computation},
    booktitle = {31st USENIX Security Symposium (USENIX Security 22)},
    year = {2022},
    isbn = {978-1-939133-31-1},
    address = {Boston, MA},
    pages = {827--844},
    url = {https://www.usenix.org/conference/usenixsecurity22/presentation/watson},
    publisher = {USENIX Association},
    month = aug,
}
```

## Artifact Evaluation

For our experiments, we use a cluser of AWS GPU-provisioned machines. Reviewers should have credentials to access the environment, but due to resource limits, we can only support one reviewer evaluating at a time. You can run Piranha to regenerate Figures 4, 5, 6, and 7, as well as Tables 2, 3, and 4.

Evaluation runs through `experiments/run_experiment.py`, which should be executed on the control instance we provide with the required dependencies. Here are the relevant options:

```
usage: run_experiment.py [-h] [--start] [--stop] [--figure FIGURE] [--table TABLE] [--generate] [--fast] [--verbose]

Run artifact evaluation!

optional arguments:
  -h, --help       show this help message and exit
  --start          Provision cluster for experiments. _Please suspend the cluster while not running experiments :)_
  --stop           Suspend evaluation machines.
  --figure FIGURE  Figure # to run.
  --table TABLE    Table # to run.
  --generate       Generate figure/table images.
  --fast           Run all the (relatively) fast runs, see README for more information
  --verbose        Display verbose run commands, helpful for debugging
```

* You can start and stop the cluster with `--start` and `--stop`, respectively. Please use these if you're not running evaluation! GPU instances are not cheap and cost about $450/day to keep running.

* Use the `--figure` and `--table` flags to run data generation for each of the paper's figures/tables. They're fairly automatic and should run without intervention. 

* Generate each figure/table with the `--generate` flag. You can run the evaluation script on partial results and the results will reflect those partial values. Figures generate `.png` files in `artifact_figures/artifact` while table replication generates JSON. You can compare to the paper figures/tables generated into `artifact_figures/paper` from hardcoded data.

* **Very important note on timing.** Unfortunately, MPC still requires a significant amount of time (~30 hrs/training run) on a larger network like VGG16. A conservative estimate is that for Figure 5 alone, > 270 computation-hours are required to replicate the full figure. We've included a `--fast` flag if you'd like to replicate every other datapoint first (will still require a number of compute-hours), then come back to the VGG-based values.

* Use `--verbose` if something isn't working and you want to take a look at the raw output or need an error message. In the backend, we use Ansible to communicate with each of the machines in the cluster.



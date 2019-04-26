# Maliciously secure protocol based on [SecureNN]()

A framework for efficient secure 3-party protocols tailored for neural networks with malicious security. This work builds off [SecureNN](), [ABY3]() and other prior works.  


### Table of Contents

- [Requirements](#requirements)
- [Source Code](#source-code)
    - [Repository Structure](#repository-structure)
    - [Building the code](#building)
    - [Running the code](#running)
- [Additional Resources](#additional-resources)
    - [Neural Networks](#neural-networks)
    - [Debugging](#debugging)


### Requirements
---
* The code should work on any Linux distribution of your choice (It has been developed and tested with [Ubuntu](http://www.ubuntu.com/) 16.04 and 18.04).

* **Required packages for SecureNN:**
  * [`g++`](https://packages.debian.org/testing/g++)
  * [`make`](https://packages.debian.org/testing/make)
  * [`libssl-dev`](https://packages.debian.org/testing/libssl-dev)

  Install these packages with your favorite package manager, e.g, `sudo apt-get install <package-name>`.

* For installation on Mac OS, contact [swagh@princeton.edu](swagh@princeton.edu).


### Source Code
---

#### Repository Structure

* `files/`    - Shared keys, IP addresses and data files.
* `lib_eigen/`    - [Eigen library](http://eigen.tuxfamily.org/) for faster matrix multiplication.
* `src/`    - Source code.
* `utils/` - Dependencies for AES randomness.

#### Building the code

To build SecureNN, run the following commands:

```
git clone https://github.com/snwagh/SecureNN.git
cd SecureNN
make
```

#### Running the code

SecureNN can be run either as a single party (to verify correctness) or as a 3 (or 4) party protocol. It can be run on a single machine (localhost) or over a network. Finally, the output can be written to the terminal or to a file (from Party *P_0*). The makefile contains the promts for each. To run SecureNN, run the appropriate command after building (a few examples given below). 

```
make standalone
make abcTerminal
make abcFile
```



### Additional Resources
---
#### General Information

* Derivative of ReLU is actually the MSB in the entire code i.e., 0 for positives, 1 for negatives.

#### Neural Networks

#### Debugging

#### Fixes

* Remove size arugment from all functions (generate it inside functions)
* Clean-up tools and functionalities file -- move reconstruction functions to tools
* Pointers to layer configurations are never deleted --> needs to be fixed

---
Report any bugs to [swagh@princeton.edu](swagh@princeton.edu)


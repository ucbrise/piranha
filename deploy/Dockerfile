# syntax=docker/dockerfile:1

# Build piranha binary
FROM nvidia/cuda:11.6.0-devel-ubuntu20.04 AS builder
ARG flags

WORKDIR /

ENV TZ=America/Los_Angeles
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update
RUN apt install build-essential libssl-dev libgtest-dev cmake -y

RUN mkdir -p /usr/src/gtest/build
WORKDIR /usr/src/gtest/build
RUN cmake ..
RUN make

WORKDIR /piranha
RUN mkdir -p build
COPY ./Makefile ./Makefile
COPY ./ext ./ext
COPY ./include ./include
COPY ./src ./src

RUN make clean
RUN make -j8 PIRANHA_FLAGS="$flags"

# Build image for running piranha
FROM nvidia/cuda:11.6.0-base-ubuntu20.04
WORKDIR /piranha

RUN apt update
RUN apt install libcublas-11-6 -y

COPY --from=builder /piranha/piranha ./piranha
COPY ./files ./files
COPY ./scripts ./scripts

RUN mkdir -p /piranha/output
CMD ["./piranha"]


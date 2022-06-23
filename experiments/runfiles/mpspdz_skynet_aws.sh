#!/bin/bash
USERNAME=ubuntu

IP1=34.223.42.20
IP2=35.86.215.164
IP3=
IP4=

SSH_KEY=~/.ssh/aws-jlw-berkeley.pem
FILE=./benchmark_piranha.mpc

#########################################################################################
RUN_TYPE=2PC # RUN_TYPE {2PC, 3PC, or 4PC}
#########################################################################################

kill_n_build () {
    scp -i $SSH_KEY $FILE $USERNAME@$1:/home/ubuntu/MP-SPDZ/Programs/Source/ >> /dev/null
    scp -i $SSH_KEY ip_aws $USERNAME@$1:/home/ubuntu/MP-SPDZ/ >> /dev/null
    local CMD="sudo pkill -f semi2k-party.x; sudo pkill -f replicated-ring-party.x; sudo pkill -f rep4-ring-party.x; cd MP-SPDZ; rm -f Programs/Bytecode/benchmark_piranha*; rm -f Programs/Schedules/benchmark_piranha*; ./compile.py -R 64 benchmark_piranha"
    local BASH_CMD="bash -ic '$CMD'"
    #echo ssh -i $SSH_KEY $USERNAME@$1 $BASH_CMD
    ssh -i $SSH_KEY -o ServerAliveInterval=30 $USERNAME@$1 $BASH_CMD
} >> /dev/null 

run_experiment () {
    local CMD="cd MP-SPDZ; $2"
    local BASH_CMD="bash -ic '$CMD'"
    #echo ssh -i $SSH_KEY $USERNAME@$1 $BASH_CMD
    ssh -i $SSH_KEY -o ServerAliveInterval=30 $USERNAME@$1 $BASH_CMD
}

if [[ $RUN_TYPE = 2PC ]]; then
    kill_n_build $IP1 &
    kill_n_build $IP2

    run_experiment $IP2 "./semi2k-party.x 1 benchmark_piranha -ip ip_aws -N 2 > ./time.txt" &> /dev/null &
    run_experiment $IP1 "./semi2k-party.x 0 benchmark_piranha -ip ip_aws -N 2 > ./time.txt; less time.txt"

elif [[ $RUN_TYPE = 3PC ]]; then
    kill_n_build $IP1 &
    kill_n_build $IP2 &
    kill_n_build $IP3

    run_experiment $IP3 "./replicated-ring-party.x 2 benchmark_piranha -ip ip_lan > ./time.txt" &> /dev/null &
    run_experiment $IP2 "./replicated-ring-party.x 1 benchmark_piranha -ip ip_lan > ./time.txt" &> /dev/null &
    run_experiment $IP1 "./replicated-ring-party.x 0 benchmark_piranha -ip ip_lan > ./time.txt; less time.txt"

elif [[ $RUN_TYPE = 4PC ]]; then

    kill_n_build $IP1 &
    kill_n_build $IP2 &
    kill_n_build $IP3 &
    kill_n_build $IP4

    run_experiment $IP4 "./rep4-ring-party.x 3 benchmark_piranha -ip ip_lan > ./time.txt" &> /dev/null &
    run_experiment $IP3 "./rep4-ring-party.x 2 benchmark_piranha -ip ip_lan > ./time.txt" &> /dev/null &
    run_experiment $IP2 "./rep4-ring-party.x 1 benchmark_piranha -ip ip_lan > ./time.txt" &> /dev/null &
    run_experiment $IP1 "./rep4-ring-party.x 0 benchmark_piranha -ip ip_lan > ./time.txt; less time.txt"

else
    echo "RUN_TYPE error" 
fi


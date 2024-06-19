#!/bin/bash

source activate tf_env

cd /home/server/rtl/pyrtl

python3 aim.py -d /mnt/nas/tests/capture/ -c 1024



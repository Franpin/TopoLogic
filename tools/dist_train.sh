#!/usr/bin/env bash
set -x

timestamp=`date +"%y%m%d.%H%M%S"`

WORK_DIR=work_dirs/$2
CONFIG=projects/configs/topologic_r50_8x1_24e_olv2_subset_B.py


GPUS=$1
PORT=${PORT:-28510}

python -m torch.distributed.run  --nproc_per_node=$GPUS --master_port=$PORT\
    tools/train.py $CONFIG --launcher pytorch --work-dir ${WORK_DIR} --deterministic ${@:3} \
    2>&1 | tee ${WORK_DIR}/train.${timestamp}.log


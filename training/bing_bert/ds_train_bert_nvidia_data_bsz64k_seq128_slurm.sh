#!/bin/bash

base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Where should we save checkpoints and tensorboard events?
JOB_NAME=lamb_nvidia_data_64k_seq128
OUTPUT_DIR=$(pwd)/bert_model_nvidia_data_outputs

DATA_PATH="/workspace/bert"

# disable progressbar in non interactive jobs
export TQDM_DISABLE=1

export RANK=$SLURM_PROCID
export LOCAL_RANK=$SLURM_LOCALID
export WORLD_SIZE=$SLURM_NTASKS

export MASTER_PORT=$(expr 10000 + $(echo -n $SLURM_JOBID | tail -c 4))
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr


mkdir -p "$OUTPUT_DIR"

NCCL_TREE_THRESHOLD=0 python "${base_dir}/deepspeed_train.py" \
--cf "${base_dir}/bert_large_lamb_nvidia_data.json" \
--max_seq_length 128 \
--output_dir "$OUTPUT_DIR" \
--deepspeed \
--deepspeed_transformer_kernel \
--print_steps 100 \
--lr_schedule "EE" \
--lr_offset 10e-4 \
--job_name "$JOB_NAME" \
--deepspeed_config "${base_dir}/deepspeed_bsz64k_lamb_config_seq128.json" \
--data_path_prefix "$DATA_PATH" \
--use_nvidia_dataset 

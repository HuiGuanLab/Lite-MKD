#!/bin/bash
#SBATCH --job-name=strm
#SBATCH --gres gpu:4
#SBATCH --nodes 1
#SBATCH --cpus-per-task 24
#SBATCH --time=2-00:00:00

NEW_HOME="./strm_hmdb_tempset_2"
COMMAND_TO_RUN="python3 run.py -c checkpoint_dir_hmdb_skeleton/ \
--query_per_class 4 --shot 5 --way 5 --trans_linear_out_dim 1152 --dataset hmdb \
--split 3 -lr 0.0001 --img_size 224 --scratch new --num_gpus 4 --method resnet50 --save_freq 10000 --print_freq 1 \
--training_iterations 40010 --temp_set 2"

echo ""
echo "Date = $(date)"
echo "Hostname = $(hostname -s)"
echo "Working Directory = $NEW_HOME"
echo "Command = $COMMAND_TO_RUN"
echo ""


$COMMAND_TO_RUN

python3 multi_fusion.py -c work/checkpoint_dir_hmdb_multi_fusion/ --query_per_class 4 --shot 5 --way 5 \
    --trans_linear_out_dim 1152 --tasks_per_batch 16 --test_iters 75000 --dataset hmdb --split 3 -lr 0.001 \
    --method resnet50 --img_size 224 --scratch new --num_gpus 1 --print_freq 1 --save_freq 75000  \
    --training_iterations 75010 \
    --temp_set 2 
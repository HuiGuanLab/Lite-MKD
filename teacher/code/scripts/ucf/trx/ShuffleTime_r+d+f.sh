export DEVICES=2   # CUDA_Visiable_Devices
export dataset="ucf"  # dataset name
export checkpoint_dir="work/ucf/fusion/trx/ThuffleTime_r+d+f" # save dir
export model="ThreeTRXShuffleTime"  # model name
export m2='depth'
export m3='flow'
export m4='skeleton'
sh scripts/base_fusion.sh


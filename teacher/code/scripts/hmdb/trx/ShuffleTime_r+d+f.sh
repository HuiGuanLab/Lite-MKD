export DEVICES=0   # CUDA_Visiable_Devices
export dataset="hmdb"  # dataset name
export checkpoint_dir="work/hmdb/fusion/trx/ShuffleTime_r+d+f" # save dir
export model="ThreeTRXShuffleTime"  # model name
export m2='depth'
export m3='flow'
export m4='skeleton'
sh scripts/base_fusion.sh


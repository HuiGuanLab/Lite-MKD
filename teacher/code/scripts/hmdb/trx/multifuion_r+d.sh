export DEVICES=2   # CUDA_Visiable_Devices
export dataset="hmdb"  # dataset name
export checkpoint_dir="work/hmdb/fusion/trx/r+d" # save dir
export model="TwoTRX"  # model name
export m2='depth'
export m3='skeleton'
export m4='flow'
sh scripts/base_fusion.sh
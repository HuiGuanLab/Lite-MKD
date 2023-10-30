export DEVICES=0   # CUDA_Visiable_Devices
export dataset="hmdb"  # dataset name
export checkpoint_dir="work/hmdb/fusion/trx/r+s" # save dir
export model="TwoTRX"  # model name
export m2='skeleton'
export m3='flow'
export m4='depth'
sh scripts/base_fusion.sh
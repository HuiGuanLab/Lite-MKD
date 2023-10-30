export DEVICES=2   # CUDA_Visiable_Devices
export dataset="hmdb"  # dataset name
export checkpoint_dir="work/hmdb/fusion/trx/r+f" # save dir
export model="TwoTRX"  # model name
export m2='flow'
export m3='skeleton'
export m4='depth'
sh scripts/base_fusion.sh
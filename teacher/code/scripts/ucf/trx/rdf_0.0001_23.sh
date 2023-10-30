export DEVICES=0  # CUDA_Visiable_Devices
export dataset="ucf"  # dataset name
export lr=0.0001
export checkpoint_dir=work/${dataset}/fusion/trx/final/lr${lr}_tempset23 # save dir
export model="ThreeTRXShiftLoopTime"  # model name
export m2='depth'
export m3='flow'
export m4='skeleton'
export shirt_num=1


CUDA_VISIBLE_DEVICES=${DEVICES} python3 multi_fusion.py -c ${checkpoint_dir} --query_per_class 4 --shot 5 --way 5 \
    --trans_linear_out_dim 1152 --tasks_per_batch 16 --test_iters 5000 10000 15000 20000 25000 30000 35000 40000 45000 50010 --dataset ${dataset} --split 3 \
    --method resnet50 --img_size 224 --scratch new --num_gpus 1 --print_freq 50 --save_freq 5000  \
    --training_iterations 50015 \
    -lr ${lr} --model ${model} --m2 ${m2} --m3 ${m3} --m4 ${m4} --shirt_num ${shirt_num} --trans_num 2


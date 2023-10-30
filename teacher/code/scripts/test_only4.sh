# dataset="kinetics"
# CUDA_VISIBLE_DEVICES=2 python3 multi_fusion.py -c work/kinetics/fusion/trx_r+s+f+d_lr_0.0008 --query_per_class 4 --shot 5 --way 5 \
#     --trans_linear_out_dim 1152 --tasks_per_batch 16 --test_iters 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 --dataset ${dataset} --split 3 \
#     --method resnet50 --img_size 224 --scratch new --num_gpus 1 --print_freq 50 --save_freq 5000  \
#     --training_iterations 50010 \
#     --temp_set 2 -lr 0.0008

# # 使用舞蹈数据集数据集的参数来测试Kinetics
# dataset="kinetics"
# CUDA_VISIBLE_DEVICES=1 python3 multi_fusion.py -r -c work/kinetics/fusion/trx/1_ShuffleLoopTime_r+d+f --query_per_class 4 --shot 5 --way 5 \
#     --trans_linear_out_dim 1152 --tasks_per_batch 16 --test_iters 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 --dataset kinetics --split 3 \
#     --method resnet50 --img_size 224 --scratch new --num_gpus 1 --print_freq 50 --save_freq 5000  \
#     --training_iterations 50010 --model ThreeTRXShiftLoopTime\
#     --temp_set 2 -lr 0.0001 --test_model_only True --test_model_path  work/dance/fusion/trx/ThreeTRXShiftLoopTime_lr0.00002/checkpoint50000.pt --trans_num 2

# # 使用hmdb数据集的参数来测试舞蹈数据集
# dataset="kinetics"
# CUDA_VISIBLE_DEVICES=3 python3 multi_fusion.py -r -c work/kinetics/fusion/trx/1_ShuffleLoopTime_r+d+f --query_per_class 4 --shot 5 --way 5 \
#     --trans_linear_out_dim 1152 --tasks_per_batch 16 --test_iters 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 --dataset kinetics --split 3 \
#     --method resnet50 --img_size 224 --scratch new --num_gpus 1 --print_freq 50 --save_freq 5000  \
#     --training_iterations 50010 --model ThreeTRXShiftLoopTime\
#     --temp_set 2 -lr 0.0001 --test_model_only True --test_model_path  work/hmdb/fusion/trx/TwoTRX_r+f/checkpoint10000.pt --trans_num 2

# 使用kinetics数据集的参数来测试hmdb数据集
dataset="hmdb"
CUDA_VISIBLE_DEVICES=2 python3 multi_fusion.py -r -c work/kinetics/fusion/trx/1_ShuffleLoopTime_r+d+f --query_per_class 4 --shot 5 --way 5 \
    --trans_linear_out_dim 1152 --tasks_per_batch 16 --test_iters 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 --dataset kinetics --split 3 \
    --method resnet50 --img_size 224 --scratch new --num_gpus 1 --print_freq 50 --save_freq 5000  \
    --training_iterations 50010 --model ThreeTRXShiftLoopTime\
    --temp_set 2 -lr 0.0001 --test_model_only True --test_model_path  work/kinetics/fusion/trx/ThreeTRXShiftLoopTime_trans2_r+s+f_lr0.00002/checkpoint50000.pt --trans_num 2
dataset="kinetics"
test_model_path="work/kinetics/fusion/params/ThreeTRXShiftLoopTime_trans1_r+d+f_lr0.00002/checkpoint25000.pt"
trans_num=1
CUDA_VISIBLE_DEVICES=1 python3 multi_fusion.py -r -c work/kinetics/fusion/params/ThreeTRXShiftLoopTime_trans1_r+d+f_lr0.00002 --query_per_class 4 --shot 5 --way 5 \
    --trans_linear_out_dim 1152 --tasks_per_batch 16 --test_iters 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 --dataset kinetics --split 3 \
    --method resnet50 --img_size 224 --scratch new --num_gpus 1 --print_freq 50 --save_freq 5000  \
    --training_iterations 50010 --model ThreeTRXShiftLoopTime\
    --temp_set 2 -lr 0.0001 \
    --test_model_only True --test_model_path ${test_model_path} --trans_num ${trans_num} \
    --demo True



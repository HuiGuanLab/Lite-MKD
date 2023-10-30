CUDA_VISIBLE_DEVICES=0,1 python3 score_fusion_run.py -c work/kinetics/score_fusion_TRX_1_0.3_0.3 --query_per_class 4 --shot 5 --way 5 \
    --trans_linear_out_dim 1152 --tasks_per_batch 16 --test_iters 30000 --dataset kinetics --split 3 -lr 0.001 \
    --method resnet50 --img_size 224 --scratch new --num_gpus 2 --print_freq 1 --save_freq 75000  \
    --training_iterations 30010 \
    --temp_set 2 --test_model_only True \
    --rgb_test_model_path work/kinetics/rgb_trx/checkpoint50000.pt \
    --skeleton_test_model_path work/kinetics/skeleton_trx/checkpoint45000.pt \
    --flow_test_model_path work/kinetics/flow_trx/checkpoint40000.pt --b 0.3 --c 0.3
    
    
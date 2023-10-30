modality=rgb 
dateset=hmdb 
model=TRX
shot=1
CUDA_VISIBLE_DEVICES=2,3 python3 run.py -c work/${dateset}/single/$shot-shot/${modality}_$model_23 --query_per_class 4 --shot $shot --way 5 \
--trans_linear_out_dim 1152 --test_iters 10000 20000 30000 40000 50000 --dataset ${dateset} --split 3 \
-lr 0.0001 --img_size 224 --scratch new --num_gpus 2 --method resnet50 --save_freq 5000 \
--print_freq 50  --training_iterations 50010 --temp_set 2 3 --modality ${modality} --model ${model}
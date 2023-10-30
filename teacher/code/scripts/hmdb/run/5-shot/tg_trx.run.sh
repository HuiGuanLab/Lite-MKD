modality=TG 
dateset=hmdb 
CUDA_VISIBLE_DEVICES=2,3 python3 run.py  -c work/${dateset}/single/${modality}_trm --query_per_class 4 --shot 5 --way 5 \
--trans_linear_out_dim 1152 --test_iters 10000 20000 30000 40000 50000 60000 70000 80000 --dataset ${dateset} --split 3 \
-lr 0.0001 --img_size 224 --scratch new --num_gpus 2 --method resnet50 --save_freq 5000 \
--print_freq 50  --training_iterations 80010 --temp_set 2 --modality ${modality}
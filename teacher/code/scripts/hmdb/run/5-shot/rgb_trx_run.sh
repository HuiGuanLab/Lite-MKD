modality=rgb 
dateset=hmdb 
CUDA_VISIBLE_DEVICES=1,2,3,0 python3 run.py -c work/${dateset}/single/${modality}_trx --query_per_class 4 --shot 5 --way 5 \
--trans_linear_out_dim 1152 --test_iters 10000 20000 30000 40000 50000 --dataset ${dateset} --split 3 \
-lr 0.0001 --img_size 224 --scratch new --num_gpus 4 --method resnet50 --save_freq 5000 \
--print_freq 50  --training_iterations 50010 --temp_set 2 --modality ${modality} --test_model_only True --test_model_path  work/hmdb/single/rgb_trx/checkpoint30000.pt
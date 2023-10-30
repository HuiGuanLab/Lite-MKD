modality=rgb 
model=CNN_STRM
dateset=kinetics 
CUDA_VISIBLE_DEVICES=0,3,2 python3 run.py -c work/${dateset}/${modality}_${model}_lr0.0001 --query_per_class 4 --shot 5 --way 5 \
--trans_linear_out_dim 1152 --test_iters 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 --dataset ${dateset} --split 3 \
-lr 0.0001 --img_size 224 --scratch new --num_gpus 3 --method resnet50 --save_freq 5000 \
--print_freq 50  --training_iterations 50010 --modality ${modality} --model ${model}
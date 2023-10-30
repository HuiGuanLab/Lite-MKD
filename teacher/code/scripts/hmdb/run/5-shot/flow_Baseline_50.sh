modality=flow
dateset=hmdb
lr=0.0003
model=Baseline
loss=CELoss
CUDA_VISIBLE_DEVICES=1,0 python3 run.py -r -c work/${dateset}/single/${modality}/${model} --query_per_class 4 --shot 5 --way 5 \
--trans_linear_out_dim 1152 --test_iters 10010 20000 30000 40000 50000 60000 70011 --dataset ${dateset} --split 3 \
-lr $lr --img_size 224 --scratch new --num_gpus 2 --method resnet50 --save_freq 5000 \
--print_freq 50  --training_iterations 70020 --temp_set 2 --modality ${modality} --model ${model}  --loss $loss \

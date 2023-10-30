dataset=ucf
CUDA_VISIBLE_DEVICES=3 python extract_multi_feature.py --path imp_datasets/video_datasets/data/$dataset/rgb_l8 --mode test \
--checkpoint_dir work/ucf/fusion/Baseline/ThreeTRXShiftLoopTime_r+d+f_lr0.00002/checkpoint25000.pt --trans_num 2 \
--getitem_name get_multi_fea --dataset $dataset --base_model Baseline \
--traintestlist imp_datasets/video_datasets/splits/ucfTrainTestlist

CUDA_VISIBLE_DEVICES=3 python extract_multi_feature.py --path imp_datasets/video_datasets/data/$dataset/rgb_l8 --mode train \
--checkpoint_dir work/ucf/fusion/Baseline/ThreeTRXShiftLoopTime_r+d+f_lr0.00002/checkpoint25000.pt --trans_num 2 \
--getitem_name get_multi_fea --dataset $dataset --base_model Baseline \
--traintestlist imp_datasets/video_datasets/splits/ucfTrainTestlist
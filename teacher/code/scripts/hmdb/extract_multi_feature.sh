dataset=hmdb
CUDA_VISIBLE_DEVICES=3 python extract_multi_feature.py --path imp_datasets/video_datasets/data/$dataset/rgb_l8 --mode test \
--checkpoint_dirwork/hmdb/fusion/Baseline/ThreeTRXShiftLoopTime_r+d+f_lr0.00002/checkpoint15000.pt --trans_num 2 \
--getitem_name get_multi_fea --dataset $dataset --base_model Baseline\
--traintestlist imp_datasets/video_datasets/splits/hmdb51TrainTestlist

CUDA_VISIBLE_DEVICES=3 python extract_multi_feature.py --path imp_datasets/video_datasets/data/$dataset/rgb_l8 --mode train \
--checkpoint_dir work/hmdb/fusion/Baseline/ThreeTRXShiftLoopTime_r+d+f_lr0.00002/checkpoint15000.pt --trans_num 2 \
--getitem_name get_multi_fea --dataset $dataset --base_model Baseline\
--traintestlist imp_datasets/video_datasets/splits/hmdb51TrainTestlist
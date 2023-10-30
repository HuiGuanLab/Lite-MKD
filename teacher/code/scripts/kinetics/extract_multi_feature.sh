CUDA_VISIBLE_DEVICES=3 python extract_multi_feature.py --path imp_datasets/video_datasets/data/kinetics/rgb_l8 --mode test \
--checkpoint_dir work/kinetics/fusion/Baseline/ThreeTRXShiftLoopTime_r+d+f_lr0.00002/checkpoint20000.pt --trans_num 2 --base_model Baseline

CUDA_VISIBLE_DEVICES=3 python extract_multi_feature.py --path imp_datasets/video_datasets/data/kinetics/rgb_l8 --mode train \
--checkpoint_dir work/kinetics/fusion/Baseline/ThreeTRXShiftLoopTime_r+d+f_lr0.00002/checkpoint20000.pt --trans_num 2 --base_model Baseline
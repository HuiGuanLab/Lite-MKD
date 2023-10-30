modality=flow

CUDA_VISIBLE_DEVICES=0,1 python pretrain/pretrain.py --path imp_datasets/video_datasets/data/kinetics/rgb_l8 \
--checkpoint_dir work/kinetics/single/$modality --model Action_Recognition_Resnet50 --num_gpus 2 \
--getitem_name get_video_with_label --modality $modality

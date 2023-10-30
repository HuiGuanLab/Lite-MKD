declare -A modality_dict
modality_dict[skeleton]="work/hmdb/single/skeleton_trx/checkpoint50000.pt"

declare -A traintestlist_dict
traintestlist_dict[ucf]="imp_datasets/video_datasets/splits/ucfTrainTestlist"
traintestlist_dict[hmdb]="imp_datasets/video_datasets/splits/hmdb51TrainTestlist"
traintestlist_dict[dance]="imp_datasets/video_datasets/splits/danceTrainTestlist"
traintestlist_dict[kinetics]="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

declare -A num_classes_dict
num_classes_dict[dance]=101
dataset="ucf"

for modality in skeleton;do
    for mode in train test; do
        CUDA_VISIBLE_DEVICES=0,1 python extract_feature.py --num_gpus=2 --modality=${modality} --checkpoint_dir=${modality_dict[$modality]} \
        --path "imp_datasets/video_datasets/data/${dataset}/${modality}_l8" \
        --mode=$mode --dataset=${dataset} --model TRX \
        --traintestlist="imp_datasets/video_datasets/splits/ucfTrainTestlist" \
        --getitem_name get_video --num_classes 101
    done
done

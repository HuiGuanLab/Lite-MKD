declare -A modality_dict
modality_dict[rgb]="work/kinetics/single/rgb/Baseline/checkpoint70000.pt"
modality_dict[depth]="work/kinetics/single/depth/Baseline/checkpoint70000.pt"
modality_dict[flow]="work/kinetics/single/flow/Baseline/checkpoint70000.pt"

declare -A traintestlist_dict
traintestlist_dict[ucf]="imp_datasets/video_datasets/splits/ucfTrainTestlist"
traintestlist_dict[hmdb]="imp_datasets/video_datasets/splits/hmdb51TrainTestlist"
traintestlist_dict[dance]="imp_datasets/video_datasets/splits/danceTrainTestlist"
traintestlist_dict[kinetics]="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

declare -A num_classes_dict
num_classes_dict[dance]=64
dataset="kinetics"

for modality in rgb flow depth;do
    for mode in train test; do
        CUDA_VISIBLE_DEVICES=0,1 python extract_feature.py --num_gpus=2 --modality=${modality} --checkpoint_dir=${modality_dict[$modality]} \
        --path "imp_datasets/video_datasets/data/${dataset}/${modality}_l8" \
        --mode=$mode --dataset=${dataset} --model Baseline \
        --traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist" \
        --getitem_name get_video --num_classes 64
    done
done

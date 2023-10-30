declare -A modality_dict
modality_dict[rgb]="work/hmdb/single/rgb_trx/checkpoint30000.pt"
modality_dict[depth]="work/hmdb/single/depth_trx/checkpoint35000.pt"
modality_dict[flow]="work/hmdb/single/flow_trx/checkpoint40000.pt"
modality_dict[skeleton]="work/hmdb/single/TG_trx/checkpoint75000.pt"
modality_dict[TG]="work/hmdb/single/TG_trx/checkpoint70000.pt"

declare -A traintestlist_dict
traintestlist_dict[ucf]="imp_datasets/video_datasets/splits/ucfTrainTestlist"
traintestlist_dict[hmdb]="imp_datasets/video_datasets/splits/hmdb51TrainTestlist"
traintestlist_dict[dance]="imp_datasets/video_datasets/splits/danceTrainTestlist"
traintestlist_dict[kinetics]="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

declare -A num_classes_dict
num_classes_dict[dance]=51
dataset="hmdb"

for modality in rgb depth flow;do
    for mode in train test; do
        CUDA_VISIBLE_DEVICES=0,1 python extract_feature.py --num_gpus=2 --modality=${modality} --checkpoint_dir=${modality_dict[$modality]} \
        --path "imp_datasets/video_datasets/data/${dataset}/${modality}_l8" \
        --mode=$mode --dataset=${dataset} --model TRX \
        --traintestlist="imp_datasets/video_datasets/splits/hmdb51TrainTestlist" \
        --getitem_name get_video --num_classes 51
    done
done

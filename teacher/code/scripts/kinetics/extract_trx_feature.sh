# modality=rgb
# python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/rgb_trx/checkpoint45000.pt \
# --path /media/star/zp/dataset/kinetics/${modality}_l8 \
# --mode="train" --dataset="kinetics" \
# --traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

# modality=rgb
# python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/rgb_trx/checkpoint45000.pt \
# --path /media/star/zp/dataset/kinetics/${modality}_l8 \
# --mode="test" --dataset="kinetics" \
# --traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

# modality=skeleton
# python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/skeleton_trx/checkpoint50000.pt \
# --path /media/star/zp/dataset/kinetics/${modality}_l8 \
# --mode="train" --dataset="kinetics" \
# --traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

# modality=skeleton
# python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/skeleton_trx/checkpoint50000.pt \
# --path /media/star/zp/dataset/kinetics/${modality}_l8 \
# --mode="test" --dataset="kinetics" \
# --traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

# modality=flow
# python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/flow_trx/checkpoint50000.pt \
# --path /media/star/zp/dataset/kinetics/${modality}_l8 \
# --mode="train" --dataset="kinetics" \
# --traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

# modality=flow
# python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/flow_trx/checkpoint50000.pt \
# --path /media/star/zp/dataset/kinetics/${modality}_l8 \
# --mode="test" --dataset="kinetics" \
# --traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

# modality=depth
# python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/depth_trx/checkpoint35000.pt \
# --path /media/star/zp/dataset/kinetics/${modality}_l8 \
# --mode="train" --dataset="kinetics" \
# --traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

# modality=depth
# python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/depth_trx/checkpoint35000.pt \
# --path /media/star/zp/dataset/kinetics/${modality}_l8 \
# --mode="test" --dataset="kinetics" \
# --traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

modality=TG
python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/single/TG/trm/checkpoint40000.pt \
--path /media/star/zp/dataset/kinetics/${modality}_l8 \
--mode="train" --dataset="kinetics" \
--traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist" --model="TRX" --getitem_name="get_video" 

modality=TG
python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/single/TG/trm/checkpoint40000.pt \
--path /media/star/zp/dataset/kinetics/${modality}_l8 \
--mode="test" --dataset="kinetics" \
--traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist" --model="TRX" --getitem_name="get_video" 


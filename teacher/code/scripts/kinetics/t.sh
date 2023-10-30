modality=TG
python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/single/TG_trx/checkpoint25000.pt \
--path /media/star/zp/dataset/kinetics/${modality}_l8 \
--mode="train" --dataset="kinetics" \
--traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist"

modality=TG
python extract_feature.py --num_gpus=4 --modality=${modality} --checkpoint_dir=work/kinetics/single/TG_trx/checkpoint25000.pt \
--path /media/star/zp/dataset/kinetics/${modality}_l8 \
--mode="test" --dataset="kinetics" \
--traintestlist="imp_datasets/video_datasets/splits/kineticsTrainTestlist"
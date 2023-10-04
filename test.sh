export mode="test_student"
export debug=False   #debug模式
export DEVICES=3   # 指定GPU
export num_gpus=1

#data
export dataset="hmdb"  # dataset name

#test model
export test_model="teacher" 
export num_test_tasks=10000  

#student param
export model_backbone="resnet50_2fc" 
export model_classifier="TRX_2fc"  
export test_model_path="/home/zty/baseline3.0/model_save/202302170343ucf19999.pt"  

#teacher param
export model_teacher="test_teacher"  
export teacher_checkpoint="/home/zty/baseline3.0/data/hmdb_teacher/new_teacher/checkpoint15000.pt" 



CUDA_VISIBLE_DEVICES=${DEVICES} python3 test.py --dataset ${dataset} --num_gpus ${num_gpus} --mode ${mode} --test_model ${test_model} --num_test_tasks ${num_test_tasks} --model_backbone ${model_backbone} --model_classifier ${model_classifier} --test_model_path ${test_model_path} --model_teacher ${model_teacher} --teacher_checkpoint ${teacher_checkpoint} --temp_set 2 --method resnet18 
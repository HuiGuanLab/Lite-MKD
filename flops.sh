export mode="测试hmdb学生"
export debug=False  
export DEVICES=3   
export num_gpus=1

#数据
export dataset="hmdb"  # dataset name

#测试模型
export test_model="teacher"  #test_model
export num_test_tasks=10000  

#学生参数
export model_backbone="strm18_student" 
export model_classifier="strmclassifiers"  
export test_model_path="/home/zty/204/model_save/202303302308ucf39999.pt"  

#教师参数
export model_teacher="test_teacher"  #teacher model
export teacher_checkpoint="/home/zty/204/data/hmdb/teacher/checkpoint35000.pt" 



CUDA_VISIBLE_DEVICES=${DEVICES} python3 flops.py --dataset ${dataset} --num_gpus ${num_gpus} --mode ${mode} --test_model ${test_model} --num_test_tasks ${num_test_tasks} --model_backbone ${model_backbone} --model_classifier ${model_classifier} --test_model_path ${test_model_path} --model_teacher ${model_teacher} --teacher_checkpoint ${teacher_checkpoint} --temp_set 2 --method resnet18 
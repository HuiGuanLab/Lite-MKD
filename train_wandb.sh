export mode="hmdb"
export debug=False  
export DEVICES="3" 
export num_gpus=1

#data                                                                                                                                                 
export dataset="hmdb"  # dataset name

#test_model
export test_model="student"  
export num_test_tasks=10000  


#train_model
export learning_rate=0.0001  
export checkpoint_dir="hmdb_cheakpoint_50双流/"   
export training_iterations=70010  

#student
export model_backbone="resnet18_2fc"
export model_classifier="TRX_2fcsup"  
export shot=5  

#distillers
export distill_name="fc_2_sup_dist"  
                                        

#teacher model  
export model_teacher="test_teacher_TRX_2fcsup_fixed"  
export teacher_checkpoint="/home/zty/204/data/hmdb/teacher/checkpoint35000.pt" 


CUDA_VISIBLE_DEVICES=${DEVICES} python3 trainwandb.py --dataset ${dataset} --shot ${shot} --debug ${debug} --num_gpus ${num_gpus} --mode ${mode} --test_model ${test_model} --num_test_tasks ${num_test_tasks} --learning_rate ${learning_rate} --checkpoint_dir ${checkpoint_dir} --training_iterations ${training_iterations} --model_backbone ${model_backbone} --model_classifier ${model_classifier} --model_teacher ${model_teacher} --teacher_checkpoint ${teacher_checkpoint} --distill_name ${distill_name}  --temp_set 2 --trans_linear_in_dim 2048 
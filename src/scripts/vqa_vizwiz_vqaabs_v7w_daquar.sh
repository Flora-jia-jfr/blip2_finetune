export WANDB__SERVICE_WAIT=300

accelerate launch CL_train.py \
    --model_config_file /home/florajia/blip2_finetune/src/configs/model_configs/blip2flant5xl.yaml \
    --training_config_file /home/florajia/blip2_finetune/src/configs/training_configs/cl_train_blip2.yaml \
    --task_list vqa,vizwiz,vqaabs,v7w,daquar \
    --mode q2a \

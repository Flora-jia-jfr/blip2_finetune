accelerate launch train.py \
    --model_config_file /home/florajia/blip2_finetune/src/configs/model_configs/blip2flant5xl.yaml \
    --training_config_file /home/florajia/blip2_finetune/src/configs/training_configs/aokvqa_train_blip2.yaml \
    --task_name aokvqa \
    --mode q2a
#
export PYTHONUNBUFFERED=1

OUTPUT_PATH=result/e/1.0

mkdir -p ${OUTPUT_PATH}

export CUDA_VISIBLE_DEVICES=0,1,2,3

python masked_run_glue.py --output_dir ${OUTPUT_PATH} --data_dir /data/GlueData/QQP \
--task_name qqp --do_train --do_eval --do_lower_case --model_type bert --model_name_or_path bert-base-uncased \
--per_gpu_train_batch_size 32 --overwrite_output_dir --warmup_steps 5000 --num_train_epochs 10 \
--max_seq_length 128 --learning_rate 12e-05 --mask_scores_learning_rate 4e-2 \
--evaluate_during_training --logging_steps 500 --save_steps 500 --fp16 \
--final_threshold 1.0  --head_pruning --final_lambda 3000 --pruning_method topK --mask_init constant \
--mask_scale 0. | tee -a ${OUTPUT_PATH}/training_log.txt 
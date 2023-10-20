#!/bin/bash

export MODEL_NAME=distilbert
export MODEL_PATH=distilbert-base-cased
export TASK=binary_abuse
export PROMPT='{"placeholder":"text_a"} Does this text contain abuse? {"mask"}'
export EVAL_BATCH=64
export PROMPT_ID='Prompt2'

echo "prompt learning: serialised training with different training size"
python src/prompt_engineering.py --n_train 0 --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID

python src/prompt_engineering.py --n_train 16  --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID
python src/prompt_engineering.py --n_train 16 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID

python src/prompt_engineering.py --n_train 32  --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID
python src/prompt_engineering.py --n_train 32 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID

python src/prompt_engineering.py --n_train 64  --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID
python src/prompt_engineering.py --n_train 64 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID

python src/prompt_engineering.py --n_train 128  --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID
python src/prompt_engineering.py --n_train 128 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID

python src/prompt_engineering.py --n_train 256 --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID
python src/prompt_engineering.py --n_train 256 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID

python src/prompt_engineering.py --n_train 512 --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID
python src/prompt_engineering.py --n_train 512 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID

python src/prompt_engineering.py --n_train 1024 --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID
python src/prompt_engineering.py --n_train 1024 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH --balanced_eval --eval_set "balanced_dev_sample" --prompt_id $PROMPT_ID
#!/bin/bash

export MODEL_NAME=gpt2
export MODEL_PATH=gpt2
export TASK=binary_abuse
export PROMPT='{"placeholder":"text_a"} It was? {"mask"}'
export EVAL_BATCH=64

echo "prompt learning: serialised training with different training size"
python src/prompt_engineering.py --n_train 0 --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH
python src/prompt_engineering.py --n_train 16  --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH
python src/prompt_engineering.py --n_train 16 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH

python src/prompt_engineering.py --n_train 32  --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH
python src/prompt_engineering.py --n_train 32 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH

python src/prompt_engineering.py --n_train 64  --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH
python src/prompt_engineering.py --n_train 64 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH

python src/prompt_engineering.py --n_train 128  --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH
python src/prompt_engineering.py --n_train 128 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH

python src/prompt_engineering.py --n_train 256 --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH
python src/prompt_engineering.py --n_train 256 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH

python src/prompt_engineering.py --n_train 512 --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH
python src/prompt_engineering.py --n_train 512 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH

python src/prompt_engineering.py --n_train 1024 --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH
python src/prompt_engineering.py --n_train 1024 --balanced_train --model_name $MODEL_NAME --model_path $MODEL_PATH --task $TASK --prompt "$PROMPT" --eval_batch_size $EVAL_BATCH
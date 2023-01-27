#!/bin/bash

echo "prompt learning: serialised training with different training size"
# python src/prompt_engineering.py --n_train 0 --balanced_train 0 --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"
# python src/prompt_engineering.py --n_train 16  --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"
# python src/prompt_engineering.py --n_train 16 --balanced_train --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"

# python src/prompt_engineering.py --n_train 32  --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"
# python src/prompt_engineering.py --n_train 32 --balanced_train --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"

# python src/prompt_engineering.py --n_train 64  --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"
# python src/prompt_engineering.py --n_train 64 --balanced_train --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"

# python src/prompt_engineering.py --n_train 128  --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"
# python src/prompt_engineering.py --n_train 128 --balanced_train --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"

# python src/prompt_engineering.py --n_train 256 --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"
# python src/prompt_engineering.py --n_train 256 --balanced_train --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"

# python src/prompt_engineering.py --n_train 512 --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"
# python src/prompt_engineering.py --n_train 512 --balanced_train --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"

# python src/prompt_engineering.py --n_train 1024 --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"
# python src/prompt_engineering.py --n_train 1024 --balanced_train --model_name distilbert --model_path "distilbert-base-cased" --task "binary_abuse"
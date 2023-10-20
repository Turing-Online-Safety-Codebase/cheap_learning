#!/bin/bash

echo "transfer learning: serialised training with different training size on abuse detection"
python src/transfer_learning.py --n_train 0 --balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"
python src/transfer_learning.py --n_train 0 --no-balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"

python src/transfer_learning.py --n_train 16 --balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"
python src/transfer_learning.py --n_train 16 --no-balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"

python src/transfer_learning.py --n_train 32 --balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"
python src/transfer_learning.py --n_train 32 --no-balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"

python src/transfer_learning.py --n_train 64 --balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"
python src/transfer_learning.py --n_train 64 --no-balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"

python src/transfer_learning.py --n_train 128 --balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"
python src/transfer_learning.py --n_train 128 --no-balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"

python src/transfer_learning.py --n_train 256  --balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"
python src/transfer_learning.py --n_train 256  --no-balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"

python src/transfer_learning.py --n_train 512  --balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"
python src/transfer_learning.py --n_train 512  --no-balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"

python src/transfer_learning.py --n_train 1024  --balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"
python src/transfer_learning.py --n_train 1024  --no-balanced_train --model_name "microsoft/deberta-v3-base" --task "binary_abuse" --balanced_eval --eval_set "balanced_dev_sample"

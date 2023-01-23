#!/bin/bash

echo "transfer learning: serialised training with different training size"
#python src/transfer_learning.py --n_train 16 --balanced_train 0 --model_name "distilbert-base-cased" --task "binary_abuse"
python src/transfer_learning.py --n_train 32 --balanced_train 0 --model_name "distilbert-base-cased" --task "binary_abuse"
python src/transfer_learning.py --n_train 64 --balanced_train 0 --model_name "distilbert-base-cased" --task "binary_abuse"
python src/transfer_learning.py --n_train 128 --balanced_train 0 --model_name "distilbert-base-cased" --task "binary_abuse"
python src/transfer_learning.py --n_train 256  --balanced_train 0 --model_name "distilbert-base-cased" --task "binary_abuse"

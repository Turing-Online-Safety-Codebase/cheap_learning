#!/bin/bash

echo "transfer learning: serialised training with different training size"
python src/transfer_learning.py --n_train_values "16,32" --balanced_train 0 --model_name "distilbert-base-cased" --task "binary_abuse" 
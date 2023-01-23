#!/bin/bash

echo "transfer learning: serialised training with different training size"
python src/transfer_learning.py --n_train 16 --balanced_train --model_name "distilbert-base-cased" --task "binary_abuse"
python src/transfer_learning.py --n_train 16 --no-balanced_train --model_name "distilbert-base-cased" --task "binary_abuse"


python src/transfer_learning.py --n_train 32 --balanced_train --model_name "distilbert-base-cased" --task "binary_abuse"
python src/transfer_learning.py --n_train 32 --no-balanced_train --model_name "distilbert-base-cased" --task "binary_abuse"


python src/transfer_learning.py --n_train 64 --balanced_train --model_name "distilbert-base-cased" --task "binary_abuse"
python src/transfer_learning.py --n_train 64 --no-balanced_train --model_name "distilbert-base-cased" --task "binary_abuse"


python src/transfer_learning.py --n_train 128 --balanced_train --model_name "distilbert-base-cased" --task "binary_abuse"
python src/transfer_learning.py --n_train 128 --no-balanced_train --model_name "distilbert-base-cased" --task "binary_abuse"


# python src/transfer_learning.py --n_train 256  --balanced_train --model_name "distilbert-base-cased" --task "binary_abuse"
# python src/transfer_learning.py --n_train 256  --no-balanced_train --model_name "distilbert-base-cased" --task "binary_abuse"

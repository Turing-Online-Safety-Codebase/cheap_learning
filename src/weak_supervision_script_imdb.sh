#!/bin/bash

export TASK=binary_movie_sentiment

echo "weak supervision: serialised training with different training size"

export EVAL_SET=balanced_dev_sample
echo "Balanced evaluation"

python src/weak_supervision.py --n_train 16 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 16 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 32 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 32 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 64 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 64 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 128 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 128 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 256 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 256 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 512 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 512 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 1024 --balanced_train --task $TASK  --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 1024 --no-balanced_train --task $TASK --eval_set $EVAL_SET


export EVAL_SET=unbalanced_dev_sample
echo "Unbalanced evaluation"

python src/weak_supervision.py --n_train 16 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 16 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 32 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 32 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 64 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 64 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 128 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 128 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 256 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 256 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 512 --balanced_train --task $TASK --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 512 --no-balanced_train --task $TASK --eval_set $EVAL_SET

python src/weak_supervision.py --n_train 1024 --balanced_train --task $TASK  --eval_set $EVAL_SET
python src/weak_supervision.py --n_train 1024 --no-balanced_train --task $TASK --eval_set $EVAL_SET

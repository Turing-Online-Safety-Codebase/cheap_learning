#!/bin/bash

export TASK=binary_movie_sentiment

echo "weak labeling: serialised training with different training size"

python src/weak_supervision.py --n_train 16 --balanced_train --task $TASK
python src/weak_supervision.py --n_train 32 --balanced_train --task $TASK
python src/weak_supervision.py --n_train 64 --balanced_train --task $TASK
python src/weak_supervision.py --n_train 128 --balanced_train --task $TASK
python src/weak_supervision.py --n_train 256 --balanced_train --task $TASK
python src/weak_supervision.py --n_train 512 --balanced_train --task $TASK
python src/weak_supervision.py --n_train 1024 --balanced_train --task $TASK 

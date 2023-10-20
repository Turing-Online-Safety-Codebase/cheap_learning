#!/bin/bash

export MODEL_NAME=NB
export TASK=binary_movie_sentiment

echo "Naive bayes: serialised training with different training size"
python src/naive_bayes_classifier.py --n_train 16 --model_name $MODEL_NAME --task $TASK --eval_set "unbalanced_dev_sample"
python src/naive_bayes_classifier.py --n_train 16 --model_name $MODEL_NAME --task $TASK --balanced_train --eval_set "unbalanced_dev_sample"

python src/naive_bayes_classifier.py --n_train 32 --model_name $MODEL_NAME --task $TASK --eval_set "unbalanced_dev_sample"
python src/naive_bayes_classifier.py --n_train 32 --model_name $MODEL_NAME --task $TASK --balanced_train --eval_set "unbalanced_dev_sample"

python src/naive_bayes_classifier.py --n_train 64 --model_name $MODEL_NAME --task $TASK --eval_set "unbalanced_dev_sample"
python src/naive_bayes_classifier.py --n_train 64 --model_name $MODEL_NAME --task $TASK --balanced_train --eval_set "unbalanced_dev_sample"

python src/naive_bayes_classifier.py --n_train 128 --model_name $MODEL_NAME --task $TASK --eval_set "unbalanced_dev_sample"
python src/naive_bayes_classifier.py --n_train 128 --model_name $MODEL_NAME --task $TASK --balanced_train --eval_set "unbalanced_dev_sample"

python src/naive_bayes_classifier.py --n_train 256 --model_name $MODEL_NAME --task $TASK --eval_set "unbalanced_dev_sample"
python src/naive_bayes_classifier.py --n_train 256 --model_name $MODEL_NAME --task $TASK --balanced_train --eval_set "unbalanced_dev_sample"

python src/naive_bayes_classifier.py --n_train 512 --model_name $MODEL_NAME --task $TASK --eval_set "unbalanced_dev_sample"
python src/naive_bayes_classifier.py --n_train 512 --model_name $MODEL_NAME --task $TASK --balanced_train --eval_set "unbalanced_dev_sample"

python src/naive_bayes_classifier.py --n_train 1024 --model_name $MODEL_NAME --task $TASK --eval_set "unbalanced_dev_sample"
python src/naive_bayes_classifier.py --n_train 1024 --model_name $MODEL_NAME --task $TASK --balanced_train --eval_set "unbalanced_dev_sample"

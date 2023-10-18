#!/bin/bash

export TASK=binary_abuse
export ANNOTATIONS=attack_annotations.tsv
export KEYWORDS=keywords.csv

echo "weak supervision: serialised training with different training size"

export EVAL_SET=balanced_dev_sample
echo "balanced evaluation"

python src/weak_supervision.py --n_train 16 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 16 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 32 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 32 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 64 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 64 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 128 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 128 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 256 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 256 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 512 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 512 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 1024 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 1024 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS


#export EVAL_SET=unbalanced_dev_sample
#echo "unbalanced evaluation"

#python src/weak_supervision.py --n_train 16 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
#python src/weak_supervision.py --n_train 16 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

#python src/weak_supervision.py --n_train 32 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
#python src/weak_supervision.py --n_train 32 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

#python src/weak_supervision.py --n_train 64 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
#python src/weak_supervision.py --n_train 64 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

#python src/weak_supervision.py --n_train 128 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
#python src/weak_supervision.py --n_train 128 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

#python src/weak_supervision.py --n_train 256 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
#python src/weak_supervision.py --n_train 256 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

#python src/weak_supervision.py --n_train 512 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
#python src/weak_supervision.py --n_train 512 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

#python src/weak_supervision.py --n_train 1024 --balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
#python src/weak_supervision.py --n_train 1024 --no-balanced_train --task $TASK --eval_set $EVAL_SET --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
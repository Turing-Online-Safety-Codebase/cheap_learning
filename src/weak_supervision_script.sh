#!/bin/bash

export TASK=binary_abuse
export ANNOTATIONS=attack_annotations.tsv
export KEYWORDS=keywords.csv

echo "weak labeling: serialised training with different training size"

python src/weak_supervision.py --n_train 2 --balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 2 --no-balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 16 --balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 16 --no-balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 32 --balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 32 --no-balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 64 --balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 64 --no-balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 128 --balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 128 --no-balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 256 --balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 256 --no-balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 512 --balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 512 --no-balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

python src/weak_supervision.py --n_train 1024 --balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS
python src/weak_supervision.py --n_train 1024 --no-balanced_train --task $TASK --filename_annotations $ANNOTATIONS --filename_keywords $KEYWORDS

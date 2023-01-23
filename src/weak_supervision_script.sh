#!/bin/bash

echo "weak labeling: serialised training with different training size"

python src/weak_labeling.py --n_train 16 --task 'binary_abuse' --filename_annotations 'attack_annotations.tsv' --filename_keywords 'keywords.csv'
python src/weak_labeling.py --n_train 32 --task 'binary_abuse' --filename_annotations 'attack_annotations.tsv' --filename_keywords 'keywords.csv'
python src/weak_labeling.py --n_train 64 --task 'binary_abuse' --filename_annotations 'attack_annotations.tsv' --filename_keywords 'keywords.csv'
python src/weak_labeling.py --n_train 128 --task 'binary_abuse' --filename_annotations 'attack_annotations.tsv' --filename_keywords 'keywords.csv'
python src/weak_labeling.py --n_train 256 --task 'binary_abuse' --filename_annotations 'attack_annotations.tsv' --filename_keywords 'keywords.csv'
python src/weak_labeling.py --n_train 512 --task 'binary_abuse' --filename_annotations 'attack_annotations.tsv' --filename_keywords 'keywords.csv'
python src/weak_labeling.py --n_train 1024 --task 'binary_abuse' --filename_annotations 'attack_annotations.tsv' --filename_keywords 'keywords.csv'
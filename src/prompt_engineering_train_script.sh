#!/bin/bash

echo "serialised training with different training size"
# python src/prompt_engineering.py --n_examples 0
# python src/prompt_engineering.py --n_examples 16
# python src/prompt_engineering.py --n_examples 32
python src/prompt_engineering.py --n_examples 64
python src/prompt_engineering.py --n_examples 128
python src/prompt_engineering.py --n_examples 256
python src/prompt_engineering.py --n_examples 512
# python src/prompt_engineering.py --n_examples 1024
python src/prompt_engineering.py --n_examples 2048
# python src/prompt_engineering.py --n_examples 4096
# python src/prompt_engineering.py --n_examples 8192
# python src/prompt_engineering.py --n_examples 16384
# python src/prompt_engineering.py --n_examples 32768
# python src/prompt_engineering.py --n_examples 65536
# python src/prompt_engineering.py --n_examples 69523
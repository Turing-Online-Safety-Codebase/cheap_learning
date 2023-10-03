#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Runs Prompt engineering experiments.
"""

import os
import argparse
import time
import datetime
import logging
import numpy
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from evaluation import get_results_dict, save_results
from helper_functions import check_dir_exists, load_n_samples, load_balanced_n_samples, convert_labels

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Prompt learning")
    parser.add_argument('--task', type=str, default='binary_abuse', help = 'name of task')
    parser.add_argument('--n_train', type=int, default=16, help='num of training points tested (seperated by ",", e.g., "15,20")')
    parser.add_argument('--n_eval', type=int, default=-1, help='num of eval entries. Set to -1 to take all entries.')
    parser.add_argument('--eval_set', type=str, default="dev_sample", help='name of eval set')
    parser.add_argument('--model_name', type=str, default='NB', help='name of the model')
    parser.add_argument('--balanced_train', action='store_true', help='If training entries are balanced by class label. Default to False.')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

def main(data_dir, n_train, n_eval, eval_set, seed, output_dir, model_name, balanced_train):
    datetime_str = str(datetime.datetime.now())

    # Setup logging
    logger.setLevel(logging.DEBUG)
    log_dir = f'{output_dir}/logs'
    check_dir_exists(log_dir)
    handler = logging.FileHandler(f"{log_dir}/{datetime_str}.log")
    logger.addHandler(handler)

    # Measure training run time, run_time will be 0 if n_train = 0 (i.e. no training)
    run_time = 0

    # Prepare dataset
    raw_dataset = {}
    if balanced_train is False:
        raw_dataset['train'], n_classes_train = convert_labels(load_n_samples(data_dir, TASK, 'train', n_train))
    else:
        raw_dataset['train'], n_classes_train = convert_labels(load_balanced_n_samples(data_dir, TASK, 'train', n_train))
    raw_dataset['eval'], n_classes_eval = convert_labels(load_n_samples(data_dir, TASK, eval_set, n_eval))

    # Build the model
    model = make_pipeline(TfidfVectorizer(), MultinomialNB())

    start = time.time()
    # Train the model using the training data
    model.fit(raw_dataset['train']['text'], raw_dataset['train']['label'])
    end = time.time()
    run_time = end - start
    
    # Inference and evalute
    logger.info("--Model Evaluation--")
    predicted_labels = model.predict(raw_dataset['eval']['text'])

    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dict = get_results_dict(TASK, TECH, model_name, run_time,
                    raw_dataset['eval']['label'], predicted_labels, eval_set,
                    n_train, n_eval, balanced_train, seed, datetime_str)
    
    # save results
    save_str = f'mod={model_name}_n={n_train}_bal={balanced_train}_s={seed}'
    save_results(output_dir, save_str, results_dict)

if __name__ == '__main__':
    args = parse_args()

    # Set global vars
    TECH = 'naive_bayes'
    TASK = args.task

    # Set dirs
    main_dir = os.getcwd()
    data_dir = f"data"
    output_dir = f'results/{TASK}/{TECH}'

    # Run for multiple training batch sizes and multiple seeds
    for SEED in [1,2,3]:
        print(f'RUNNING for SEED={SEED}')
        main(data_dir, args.n_train, args.n_eval, args.eval_set, SEED, output_dir, args.model_name, args.balanced_train)
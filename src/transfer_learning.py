#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Runs Transfer Learning experiments.
"""

import argparse
import os
from datetime import datetime
import logging
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from datasets import Dataset, DatasetDict
from evaluation import get_results_dict, save_results
from helper_functions import check_dir_exists, load_n_samples, load_balanced_n_samples, convert_labels

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer learning")
    parser.add_argument('--n_train_values', type=str, default='15,20,32,50,70,100,150', help='str list of num of training entries (seperated by ",", e.g., "15,20")')
    parser.add_argument('--balanced_train', type=bool, default=False, help='if training entries are balanced by class label')
    parser.add_argument('--n_test', type=int, default=3000, help='num of testing entries')
    parser.add_argument('--n_dev', type=int, default=1000, help='num of dev entries')
    parser.add_argument('--model_name', type=str, default='bert', help='name of the model')
    parser.add_argument('--task', type=str, defaulthelp='target task')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

def convert_data_format(train_df, dev_df, test_df):
    tf_dataset = DatasetDict()
    for df, split in zip([train_df, dev_df, test_df], ['train', 'dev', 'test']):
        split_ds = Dataset.from_pandas(df[['text', 'label']])
        tf_dataset[split] = split_ds
    return tf_dataset


def get_pred_labels(trainer, tokenized_datasets, split):
    logits = trainer.predict(tokenized_datasets[split])
    y_pred = np.argmax(logits.predictions, axis=-1)
    y_true = np.array(tokenized_datasets[split]['label'])
    return y_pred, y_true

def get_runtime(train_output):
    metrics = train_output.metrics
    return metrics['train_runtime']

def main(TASK, TECH, data_dir, output_dir, n_train, balanced_train, n_test, n_dev, model_name):
    datetime_str = str(datetime.now())

    # Setup logging
    logger.setLevel(logging.DEBUG)
    log_dir = f'{output_dir}/logs'
    check_dir_exists(log_dir)
    handler = logging.FileHandler(f"{log_dir}/{datetime_str}.log")
    # format = logging.Formatter('%(asctime)s  %(name)s %(levelname)s: %(message)s')
    # handler.setFormatter(format)
    logger.addHandler(handler)

    # Load data
    if balanced_train is False:
        train_df, n_classes_train = convert_labels(load_n_samples(data_dir, TASK, "train", n_train)) 
    else:
        train_df, n_classes_train = convert_labels(load_balanced_n_samples(data_dir, TASK, "train", n_train))
    dev_df, n_classes_dev = convert_labels(load_n_samples(data_dir, TASK, "dev", n_dev))
    test_df, n_classes_test = convert_labels(load_n_samples(data_dir, TASK, "test", n_test))
    if n_classes_train == n_classes_dev == n_classes_test:
        n_classes = n_classes_train
    else:
        print("Error: train, test or dev have different number of classes")

    for df, split in zip([train_df, dev_df, test_df], ['train', 'dev', 'test']):
        logger.info(f'--{len(df)} examples in {split} set--\n')
        logger.info(f"--label distribution for {split} set--\n{df['label'].value_counts()}")
    # Convert data format
    dataset = convert_data_format(train_df, dev_df, test_df)

    # Tokenize
    logger.info("--Tokenization--")
    tokenizer= AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Model Training
    logger.info("--Model Training--")
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)
    training_args = TrainingArguments(output_dir=output_dir, evaluation_strategy="epoch")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],)
    train_output = trainer.train()

    # Model Evaluation
    logger.info("--Model Evaluation--")
    runtime = get_runtime(train_output)
    dev_pred, dev_true = get_pred_labels(trainer, tokenized_datasets, "dev")
    test_pred, test_true = get_pred_labels(trainer, tokenized_datasets, "test")
    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dict = get_results_dict(TASK, TECH, model_name, runtime,
                                    test_true, test_pred,
                                    dev_true, dev_pred,
                                    n_train, n_dev, n_test, balanced_train,
                                    datetime_str)
    # Save output
    save_results(output_dir, datetime_str, results_dict)

if __name__ == '__main__':
    args = parse_args()

    # Set global vars
    SEED = 123
    TECH = 'transfer_learning'
    TASK = args.task

    # Set dirs
    path = os.getcwd()
    main_dir = os.path.split(path)[0]
    data_dir = f"{main_dir}/data"
    output_dir = f'{main_dir}/results/{TASK}/{TECH}'

    # Run for multiple training batch sizes
    n_train_list = args.n_train_values.split(',')
    for n_train in n_train_list:
        main(TASK, TECH, data_dir, output_dir, n_train, args.balanced_train, args.n_test, args.n_dev, args.model_name)

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
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, pipeline
from datasets import Dataset, DatasetDict
from evaluation import get_results_dict, save_results
from helper_functions import check_dir_exists, load_n_samples, load_balanced_n_samples, convert_labels

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Transfer learning")
    parser.add_argument('--n_train', type=int, default=16, help='num of training entries')
    parser.add_argument('--balanced_train', type=bool, action=argparse.BooleanOptionalAction, help='if training entries are balanced by class label')
    parser.add_argument('--eval_set', type=str, default="dev_sample", help='name of eval set')
    parser.add_argument('--n_eval', type=int, default=-1, help='num of eval entries. Set to -1 to take all entries.')
    parser.add_argument('--model_name', type=str, default='bert', help='name of the model')
    parser.add_argument('--task', type=str, default='target task', help = 'name of task')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

def convert_data_format(train_df, eval_df):
    tf_dataset = DatasetDict()
    for df, split in zip([train_df, eval_df], ['train', 'eval']):
        split_ds = Dataset.from_pandas(df[['text', 'label']])
        tf_dataset[split] = split_ds
    return tf_dataset


def get_pred_labels(trainer, tokenized_datasets, split):
    logits = trainer.predict(tokenized_datasets[split])
    y_pred = np.argmax(logits.predictions, axis=-1)
    y_true = np.array(tokenized_datasets[split]['label'])
    return y_pred, y_true

def get_pred_labels_zero_shot(tokenized_datasets, split, labels, model_name, tokenizer):
    classifier = pipeline(task="zero-shot-classification", model=model_name, tokenizer=tokenizer)
    # Below does not seem to work in batches, so loop through the dataset for prediction
    # preds = classifier(tokenized_datasets[split]['text'], batch_size=16, candidate_labels=labels)
    # y_pred = [pred["labels"][0] for pred in preds]
    y_pred = []
    for text in tokenized_datasets[split]['text']:            
        preds = classifier(text, candidate_labels=labels)
        print(preds["labels"][0], preds["labels"])
        y_pred.append(preds["labels"][0])
    y_true = np.array(tokenized_datasets[split]['label'])
    return y_pred, y_true

def get_runtime(train_output):
    metrics = train_output.metrics
    return metrics['train_runtime']

def main(SEED, TASK, TECH, data_dir, output_dir, n_train, balanced_train, eval_set, n_eval, model_name):
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
    eval_df, n_classes_eval = convert_labels(load_n_samples(data_dir, TASK, eval_set, n_eval))
    if n_classes_train == n_classes_eval or n_train == 0:
        n_classes = n_classes_train
        labels = eval_df['label'].unique()
    else:
        print("Error: train and eval have different number of classes")
    for df, split in zip([train_df, eval_df], ['train', 'dev', 'test']):
        logger.info(f'--{len(df)} examples in {split} set--\n')
        logger.info(f"--label distribution for {split} set--\n{df['label'].value_counts()}")
    # Convert data format
    dataset = convert_data_format(train_df, eval_df)

    # Tokenize
    logger.info("--Tokenization--")
    tokenizer= AutoTokenizer.from_pretrained(model_name)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True)
    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=n_classes)

    training_args = TrainingArguments(output_dir=output_dir, 
                                        do_predict = False, do_eval = False,
                                        seed=SEED)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"])

    # Model Training
    if n_train > 0:
        logger.info("--Model Training--")
        train_output = trainer.train()
        runtime = get_runtime(train_output)

        # Model Evaluation
        logger.info("--Model Evaluation--")
        eval_pred, eval_true = get_pred_labels(trainer, tokenizer, tokenized_datasets, "eval", n_train, labels, model_name)
    else:
        runtime = 0
        # Zero-Shot Model Evaluation
        logger.info("--Zero-Shot Model Evaluation--")
        eval_pred, eval_true = get_pred_labels_zero_shot(tokenized_datasets, "eval", labels, model_name, tokenizer)

    datetime_str = datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dict = get_results_dict(TASK, TECH, 
                                    model_name, runtime,
                                    eval_true, eval_pred, eval_set,
                                    n_train, n_eval, balanced_train, 
                                    SEED, datetime_str)
    # Save output
    if model_name == "microsoft/deberta-v3-base":
        model_name = "deberta-v3-base"
    save_str = f'mod={model_name}_n={n_train}_bal={balanced_train}_s={SEED}'
    save_results(output_dir, save_str, results_dict)

if __name__ == '__main__':
    args = parse_args()

    # Set global vars
    TECH = 'transfer_learning'
    TASK = args.task

    # Set dirs
    main_dir = os.getcwd()
    data_dir = f"{main_dir}/data"
    output_dir = f'{main_dir}/results/{TASK}/{TECH}'

    # Run for multiple training batch sizes and multiple seeds
    for SEED in [1,2,3]:
        print(f'RUNNING for SEED={SEED}')
        main(SEED, TASK, TECH, data_dir, output_dir, args.n_train, args.balanced_train, args.eval_set, args.n_eval, args.model_name)

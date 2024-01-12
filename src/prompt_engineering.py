#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Runs Prompt engineering experiments.
"""

import os
import argparse
import torch
import time
import datetime
import logging
import numpy
import pandas as pd
from openprompt.data_utils import InputExample
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from openprompt.utils.reproduciblity import set_seed
from transformers import AdamW
from evaluation import evaluate, get_results_dict, save_results
from helper_functions import check_dir_exists, load_n_samples, load_balanced_n_samples, convert_labels
from load_lm import load_plm

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Prompt engineering")
    parser.add_argument('--task', type=str, default='binary_abuse', help = 'name of task')
    parser.add_argument('--prompt', type=str, default='{"placeholder":"text_a"} It was? {"mask"}', help = 'prompt')
    parser.add_argument('--prompt_id', type=str, default='Prompt1', help = 'prompt id indicating the folder to store results')
    parser.add_argument('--n_train', type=int, default=16, help='num of training points tested (seperated by ",", e.g., "15,20")')
    parser.add_argument('--n_eval', type=int, default=-1, help='num of eval entries. Set to -1 to take all entries.')
    parser.add_argument('--eval_set', type=str, default="dev_sample", help='name of eval set')
    parser.add_argument('--model_name', type=str, default='bert', help='name of the model')
    parser.add_argument('--model_path', type=str, default='bert-base-cased', help='path to the model')
    parser.add_argument('--use_cuda', action='store_false', help='If using cuda. Default to True')
    parser.add_argument('--balanced_train', action='store_true', help='If training entries are balanced by class label. Default to False.')
    parser.add_argument('--balanced_eval', action='store_true', help='If test entries are balanced by class label. Default to False.')
    parser.add_argument('--eval_batch_size', type=int, default=128, help='num of examples evaluated at once')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

def main(data_dir, n_train, n_eval, eval_set, template, seed, output_dir, model_name, model_path, balanced_train, use_cuda, eval_batch_size, balanced_eval):
    datetime_str = str(datetime.datetime.now())

    set_seed(seed)

    # Setup logging
    logger.setLevel(logging.DEBUG)
    log_dir = f'{output_dir}/logs'
    check_dir_exists(log_dir)
    handler = logging.FileHandler(f"{log_dir}/{datetime_str}.log")
    logger.addHandler(handler)

    # Measure training run time, run_time will be 0 if n_train = 0 (i.e. no training)
    run_time = 0

    # Load a Pre-trained Language Model (PLM).
    plm, tokenizer, model_config, WrapperClass = load_plm(model_name, model_path)
    
    # Define a Template.
    promptTemplate = ManualTemplate(
        text=template,
        tokenizer=tokenizer,
    )

    # Define a Verbalizer
    promptVerbalizer = ManualVerbalizer(
        classes= [0, 1],
        label_words={
            0: ["good", "positive", "great"],
            1: ["bad", "negative"],
        },
        tokenizer=tokenizer,
    )

    # Combine them into a PromptModel
    promptModel = PromptForClassification(
        template=promptTemplate,
        plm=plm,
        verbalizer=promptVerbalizer,
    )
    if use_cuda:
        promptModel = promptModel.cuda()

    # Set up an optimizer
    loss_func = torch.nn.CrossEntropyLoss()
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in promptModel.named_parameters() if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in promptModel.named_parameters() if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=1e-4)

    def inference(model, dataset):
        model.eval()
        preds = []
        gold_labels = []
        with torch.no_grad():
            for step, inputs in enumerate(dataset):
                if use_cuda:
                    inputs = inputs.cuda()
                logits = model(inputs)
                labels = inputs['label']
                gold_labels.extend(labels.cpu().tolist()) 
                preds.extend(torch.argmax(logits, dim=-1).cpu().tolist())

        # Compute scores
        results = evaluate(gold_labels, preds)
        return gold_labels, preds, results

    # Prepare train dataset
    raw_dataset = {}
    if balanced_train is False:
        if TASK == 'binary_movie_sentiment' :
            df = pd.read_csv(f'{data_dir}/{TASK}/unbalanced_data/{TASK}_train.csv', nrows =  n_train)
            raw_dataset['train'], n_classes_train = convert_labels(df)
        else:
            raw_dataset['train'], n_classes_train = convert_labels(load_n_samples(data_dir, TASK, 'train', n_train))
    else:
        raw_dataset['train'], n_classes_train = convert_labels(load_balanced_n_samples(data_dir, TASK, 'train', n_train))
    raw_dataset['eval'], n_classes_eval = convert_labels(load_n_samples(data_dir, TASK, eval_set, n_eval))

    dataset = {}
    for split in ['train', 'eval']:
        logger.info(f'--{len(raw_dataset[split])} examples in {split} set--\n')
        logger.info(f"--label distribution for {split} set--\n{raw_dataset[split]['label'].value_counts()}")
        dataset[split] = []
        for index, row in raw_dataset[split].iterrows():
            input_example = InputExample(text_a=row['text'], label=int(row['label']), guid=row['id'])
            dataset[split].append(input_example)

    eval_dataloader = PromptDataLoader(
        dataset=dataset["eval"],
        template=promptTemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=512,
        batch_size=eval_batch_size,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail")

    if int(n_train) > 0:
        # If n_train > 0, do train.
        # Otherwise, do zero-shot evaluation on dev and test set
        train_dataloader = PromptDataLoader(
            dataset=dataset['train'],
            template=promptTemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=512,
            batch_size=8,
            shuffle=False,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="tail")

        logger.info("--Model Training--")
        start = time.time()
        for epoch in range(1, 4):
            promptModel.train()
            tot_loss = 0
            for step, inputs in enumerate(train_dataloader):
                if use_cuda:
                    inputs = inputs.cuda()
                optimizer.zero_grad()
                labels = inputs['label']
                logits = promptModel(inputs)
                loss = loss_func(logits, labels)
                loss.backward()
                tot_loss += loss.item()
                optimizer.step()
                logger.info(f"Epoch {epoch}, step {step}, average loss: {tot_loss/(step+1)}")
        end = time.time()
        run_time = end - start

    # Inference and evalute
    logger.info("--Model Evaluation--")
    eval_gold_labels, eval_preds, eval_result = inference(promptModel, eval_dataloader)

    datetime_str = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    results_dict = get_results_dict(TASK, TECH, model_name, run_time,
                    eval_gold_labels, eval_preds, eval_set,
                    n_train, n_eval, balanced_train, balanced_eval, seed, datetime_str, template)
    # add test_result to results_dict
    results_dict.update(eval_result)
    save_str = f'mod={model_name}_n={n_train}_bal={balanced_train}_s={seed}_balEval={balanced_eval}'
    save_results(output_dir, save_str, results_dict)

if __name__ == '__main__':
    args = parse_args()

    # Set global vars
    TECH = 'prompt_engineering'
    TASK = args.task
    PROMPT_ID = args.prompt_id

    # Set dirs
    main_dir = os.getcwd()
    data_dir = f"data"
    output_dir = f'results/{TASK}/{TECH}/{PROMPT_ID}'

    # Run for multiple training batch sizes and multiple seeds
    for SEED in [1, 2, 3]:
        print(f'RUNNING for SEED={SEED}')
        main(data_dir, args.n_train, args.n_eval, args.eval_set, args.prompt, SEED, output_dir, args.model_name, args.model_path, args.balanced_train, args.use_cuda, args.eval_batch_size, args.balanced_eval)
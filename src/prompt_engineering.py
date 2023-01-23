#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Runs Prompt engineering experiments. (in progress)
"""

import argparse
import torch
import pandas
import time
import datetime
import logging
import numpy
from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate, ManualVerbalizer
from openprompt import PromptForClassification, PromptDataLoader
from openprompt.utils.reproduciblity import set_seed
from transformers import AdamW
from evaluation import evaluate, get_results_dict, save_results
from helper_functions import load_n_samples, load_balanced_n_samples, convert_labels

logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="Prompt learning")
    parser.add_argument('--n_examples', type=str, default='15,20,32,50,70,100,150', help='num of training points tested (seperated by ",", e.g., "15,20")')
    parser.add_argument('--model_name', type=str, default='bert', help='name of the model')
    parser.add_argument('--model_path', type=str, default='bert-base-cased', help='path to the model')
    parser.add_argument('--use_cuda', type=bool, default=True, help='if using cuda')
    parser.add_argument('--balanced_train', type=bool, action=argparse.BooleanOptionalAction, help='if training entries are balanced by class label')
    parser.add_argument('--output_dir', type=str, default='results', help='directory to the results')
    parser.add_argument('--eval_steps', type=int, default='4', help='num of update steps between two evaluations')
    pars_args = parser.parse_args()

    print("the inputs are:")
    for arg in vars(pars_args):
        print(f"{arg} is {getattr(pars_args, arg)}")
    return pars_args

def main(n_examples, template, seed, output_dir, model_name, model_path, balanced_train, use_cuda):
    datetime_str = str(datetime.datetime.now())

    set_seed(seed)

    # Setup logging
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(f"{output_dir}/{datetime_str}.log")
    # format = logging.Formatter('%(asctime)s  %(name)s %(levelname)s: %(message)s')
    # handler.setFormatter(format)
    logger.addHandler(handler)

    # Measure training run time, run_time will be 0 if n = 0 (i.e. no training)
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
        classes= [0, 1],    # Two classes: 0 for not abusive and 1 for abusive
        label_words={
            0: ["love"],
            1: ["hate"],
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
        preds = numpy.array([])
        gold_labels = numpy.array([])
        with torch.no_grad():
            for step, inputs in enumerate(dataset):
                if use_cuda:
                    inputs = inputs.cuda()
                logits = model(inputs)
                labels = inputs['label']
                gold_labels = numpy.concatenate((gold_labels, labels.cpu()), axis=0)
                preds = numpy.concatenate((preds, torch.argmax(logits, dim=-1).cpu()), axis=0)

        # logger.info(f"num of true and prediction: {sum(gold_labels)} and {sum(preds)}")
        # Compute scores
        results = evaluate(gold_labels.tolist(), preds.tolist())
        return gold_labels, preds, results

    # Prepare dataset
    raw_dataset = {}
    if balanced_train is False:
        raw_dataset['train'] = load_n_samples('data', 'binary_abuse', 'train', int(n_examples))
    else:
        raw_dataset['train'] = load_balanced_n_samples('data', 'binary_abuse', 'train', int(n_examples))
    
    raw_dataset['val'] = pandas.read_csv("data/binary_abuse/clean_data/binary_abuse_dev.csv")
    raw_dataset['test'] = pandas.read_csv("data/binary_abuse/clean_data/binary_abuse_test.csv")
    raw_dataset['train'], n_classes = convert_labels(raw_dataset['train'])
    raw_dataset['val'], n_classes = convert_labels(raw_dataset['val'])
    raw_dataset['test'], n_classes = convert_labels(raw_dataset['test'])
    logger.info(f'--{num} number of training examples--\n')
    logger.info(f"--label distribution for train set--\n{raw_dataset['train']['label'].value_counts()}")
    n_dev, n_test = len(raw_dataset['val']), len(raw_dataset['test'])

    dataset = {}
    for split in ['train', 'val', 'test']:
        dataset[split] = []
        for index, row in raw_dataset[split].iterrows():
            input_example = InputExample(text_a=row['text'], label=int(row['label']), guid=row['rev_id'])
            dataset[split].append(input_example)

    test_dataloader = PromptDataLoader(
        dataset=dataset["test"],
        template=promptTemplate,
        tokenizer=tokenizer,
        tokenizer_wrapper_class=WrapperClass,
        max_seq_length=256,
        batch_size=128,
        shuffle=False,
        teacher_forcing=False,
        predict_eos_token=False,
        truncate_method="tail")

    validation_dataloader = PromptDataLoader(
            dataset=dataset["val"],
            template=promptTemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=256,
            batch_size=128,
            shuffle=False,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="tail")

    if int(n_examples) > 0:
        # If n_examples > 0, do train.
        # Otherwise, do zero-shot evaluation on dev and test set
        train_dataloader = PromptDataLoader(
            dataset=dataset['train'],
            template=promptTemplate,
            tokenizer=tokenizer,
            tokenizer_wrapper_class=WrapperClass,
            max_seq_length=256,
            batch_size=8,
            shuffle=False,
            teacher_forcing=False,
            predict_eos_token=False,
            truncate_method="tail")

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
            logger.info("--Perform validation--")
        end = time.time()
        run_time = end - start
        val_gold_labels, val_preds, val_result = inference(promptModel, validation_dataloader)
    else:        
        logger.info("--Perform validation--")
        val_gold_labels, val_preds, val_result = inference(promptModel, validation_dataloader)

    # Inference and evalute
    logger.info("--Perform testing--")
    test_gold_labels, test_preds, test_result = inference(promptModel, test_dataloader)

    datetime_str = str(datetime.datetime.now())
    results_dict = get_results_dict(TASK, TECH, model_name, run_time,
                    test_gold_labels, test_preds,
                    val_gold_labels, val_preds,
                    int(n_examples), n_dev, n_test, balanced_train, seed, datetime_str, template)
    # add test_result to results_dict
    results_dict.update(test_result)
    save_results(output_dir, datetime_str, results_dict)

if __name__ == '__main__':
    args = parse_args()

    num_examples = args.n_examples.split(',')
    # templates_list = ['{"placeholder":"text_a"} Is this text abusive? {"mask"}', 
    #              '{"placeholder":"text_a"} Does this text contain abuse? {"mask"}',
    #              '{"placeholder":"text_a"} It was? {"mask"}',
    #              ]
    templates = '{"placeholder":"text_a"} It was? {"mask"}'

    for num in num_examples:
        for SEED in [1, 2, 3]:
            main(num, templates, SEED, args.output_dir, args.model_name, args.model_path, args.balanced_train, args.use_cuda)

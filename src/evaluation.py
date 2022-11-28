#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Evaluates prediction array.
"""
import json
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    )
from helper_functions import check_dir_exists

# # example
# y_true = [1, 1, 0, 1]
# y_pred = [1, 1, 0, 0]
# result = evaluate(y_true, y_pred)
# print(result)

def evaluate(true, pred, target_names=["not_hateful", "hateful"]):
    """Computes the number of votes received for each class labelled for an entry.
        Three classes are included: disagree, agree, other.
    Args:
        true (list): ground truth (correct) target values
        pred (list): model prediction
        target_names (list): display names matching the labels (same order)
    Returns:
        a dictionary of the evaluation results including accuracy, f1, precision, recall,
        and confusion metric
    """
    results = {}
    results['test_accuracy'] = accuracy_score(true, pred)
    results['test_precision'] = precision_score(true, pred, average='weighted', zero_division=0)
    results['test_recall'] = recall_score(true, pred, average='weighted', zero_division=0)
    results['test_f1'] = f1_score(true, pred, average='weighted', zero_division=0)
    print(f'--confusion metric-- \n {confusion_matrix(true, pred)}')
    results['test_cm (tn, fp, fn, tp)'] = confusion_matrix(true, pred, normalize='true').ravel().tolist()
    print("\n--full report--")
    print(classification_report(true, pred, output_dict=False, target_names=target_names))
    return results

def evaluate_dataframe(s):
    """Function to be applied to dataframe with prediction lists as columns. 
    E.g. results_df[['acc', 'f1', 'prec', 'recall']] = results_df.apply(evaluate_dataframe, axis=1, result_type="expand")

    Args:
        s (pd.DataFrame): Results dataframe.

    Returns:
        float: list of floats for eval scores.
    """
    true = s['test_true']
    pred = s['test_pred']
    acc = accuracy_score(true, pred)
    f1 = f1_score(true, pred, average='macro', zero_division=0)
    prec = precision_score(true, pred, average='weighted', zero_division=0)
    recall = recall_score(true, pred, average='weighted', zero_division=0)
    return acc,f1, prec, recall


def get_results_dict(task, technique, model_name, runtime,
                      test_true, test_pred,
                      dev_true, dev_pred,
                      n_train, n_dev, n_test, balanced_train, seed, datetime_str, template=''):
    """Standardizes results dictionary.

    Args:
        task (str): The current task e.g., binary_abuse.
        technique (str): The learning technique e.g., transfer_learning.
        model_name (str): The model name (if applicable).
        runtime (str): The training runtime of the technique in seconds.
        test_true (np.array): True labels for test set.
        test_pred (np.array): Pred labels for test set.
        dev_true (np.array): True labels for dev set.
        dev_pred (np.array): Pred labels for dev set.
        n_train (int): Number of training entries.
        n_dev (int): Number of dev entries.
        n_test (int): Number of test entries.
        balanced_train (bool): Whether training data is balanced by class label.
        seed (int): Random seed for experiment run.
        datetime_str (str): Current datetime.
        template(str) (optional): prompt template.

    Returns:
        dict: Dictionary of results.
    """
    results_dict = {}
    results_dict['task'] = task
    results_dict['technique'] = technique
    results_dict['model'] = model_name
    results_dict['train_runtime'] = runtime
    results_dict['n_train'] = n_train
    results_dict['n_dev'] = n_dev
    results_dict['n_test'] = n_test
    results_dict['balanced_train'] = balanced_train
    results_dict['datetime'] = datetime_str
    results_dict['test_true'] = test_true.tolist()
    results_dict['test_pred'] = test_pred.tolist()
    results_dict['dev_true'] = dev_true.tolist()
    results_dict['dev_pred'] = dev_pred.tolist()
    results_dict['seed'] = seed
    if technique == 'prompt_engineering':
        results_dict['template'] = template
    return results_dict


def save_results(output_dir, datetime_str, results_dict):
    """Saves results dictionary as a json.

    Args:
        output_dir (str): Filepath to store results.
        datetime_str (str): Current datetime for filename.
        results_dict (dict): Dictionary of results
    """
    check_dir_exists(output_dir)
    with open(f'{output_dir}/{datetime_str}.json', 'w', encoding="utf-8") as file:
        json.dump(results_dict, file)

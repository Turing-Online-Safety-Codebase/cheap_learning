#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

"""
Evaluates prediction array.
"""

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    )

# # example
# y_true = [1, 1, 0, 1]
# y_pred = [1, 1, 0, 0]
# result = evaluation(y_true, y_pred)
# print(result)

def evaluation(true, pred, target_names=["hateful", "not_hateful"]):
    """Compute the number of votes received for each class labelled for an entry.
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
    results['accuracy'] = accuracy_score(true, pred)
    results['precision'] = precision_score(true, pred, average='weighted', zero_division=0)
    results['recall'] = recall_score(true, pred, average='weighted', zero_division=0)
    results['f1'] = f1_score(true, pred, average='weighted', zero_division=0)
    print(f'--confusion metric-- \n {confusion_matrix(true, pred)}')
    results['tn, fp, fn, tp'] = confusion_matrix(true, pred, normalize='true').ravel()
    print("\n--full report--")
    print(classification_report(true, pred, output_dict=False, target_names=target_names))
    return results
